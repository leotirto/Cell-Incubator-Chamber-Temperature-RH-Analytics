"""Incubator controller log ingestion, engineering metrics and Plotly visualizations.

No mutation of original CSVs. Derived/aggregated views are explicitly separated from
observations. CO2 and STC31 units are intentionally NOT assumed. All timestamps are
interpreted as naive device-local clock times unless a caller supplies a conversion.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TIME = "Timestamp"
TEMP_COLS = ["T Steam", "T Chamber", "T Bot", "T Door", "T Port"]
CTRL_COLS = ["Ctrl Steam", "Ctrl Chamber", "Ctrl Bot", "Ctrl Door", "Ctrl Port", "Ctrl RH", "Ctrl CO2"]
PROCESS_COLS = TEMP_COLS + ["RH", "CO2", "STC31"]
MEASURED_COLS = TEMP_COLS + CTRL_COLS + ["RH", "CO2", "STC31"]
CONTROL_PAIRS = {
    "T Steam": "Ctrl Steam", "T Chamber": "Ctrl Chamber", "T Bot": "Ctrl Bot",
    "T Door": "Ctrl Door", "T Port": "Ctrl Port", "RH": "Ctrl RH", "CO2": "Ctrl CO2"
}
TEMP_GROUP = ["T Chamber", "T Bot", "T Door", "T Port"]
COLOURS = {
    "T Steam": "#e05d43", "T Chamber": "#2563eb", "T Bot": "#0d9488",
    "T Door": "#8b5cf6", "T Port": "#d97706", "RH": "#0891b2",
    "CO2": "#b45309", "STC31": "#64748b"
}


@dataclass
class LoadedLog:
    name: str
    frame: pd.DataFrame
    quality: dict


def _time_column(columns: Iterable[str]) -> str:
    """Handle unusual CSV with first header equal to data logger start time."""
    columns = list(columns)
    for c in columns:
        if c.strip().casefold() in {"timestamp", "datetime", "date", "time", "date time"}:
            return c
    if not columns:
        raise ValueError("CSV has no columns")
    first = columns[0]
    if pd.notna(pd.to_datetime(first, errors="coerce")):
        return first
    raise ValueError("No recognized timestamp column; first header is not a timestamp")


def _numeric(series: pd.Series) -> pd.Series:
    # Accept '36,5', '36.5 °C' and '90.0 %'; never silently delete mixed text.
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    s = series.astype("string").str.strip()
    s = s.str.replace(r"\s*(?:°\s*C|℃|%|ppm)\s*$", "", case=False, regex=True)
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def read_controller_log(path: str | Path, *, name: str | None = None) -> LoadedLog:
    """Read a log, preserve source row numbers, and report data issues without correction.

    Trailing empty CSV column is removed; rows with invalid times remain counted
    in quality and are omitted from chronological analysis (not from source file).
    Duplicate timestamps remain in the cleaned frame, but are explicitly flagged.
    """
    path = Path(path)
    errors = []
    for enc in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            raw = pd.read_csv(path, encoding=enc, low_memory=False)
            break
        except (UnicodeError, UnicodeDecodeError, pd.errors.ParserError) as e:
            errors.append(f"{enc}: {e}")
    else:
        raise ValueError("Unable to load CSV: " + "; ".join(errors))
    if raw.empty:
        raise ValueError(f"Empty log: {path}")
    raw.columns = [str(c).strip().lstrip("\ufeff") for c in raw.columns]
    discarded = [c for c in raw if c.startswith("Unnamed:") and raw[c].isna().all()]
    raw = raw.drop(columns=discarded)
    stamp = _time_column(raw.columns)
    raw = raw.rename(columns={stamp: TIME})
    # Recognize variants without overwriting a preexisting matching column.
    canonical = {re.sub(r"[^a-z0-9]+", "", c.lower()): c for c in MEASURED_COLS}
    renames = {}
    for c in raw.columns:
        alias = re.sub(r"[^a-z0-9]+", "", c.lower())
        if alias in canonical and c != canonical[alias] and canonical[alias] not in raw.columns:
            renames[c] = canonical[alias]
    raw = raw.rename(columns=renames)
    raw.insert(0, "SourceRow", np.arange(2, len(raw) + 2, dtype=int))
    raw[TIME] = pd.to_datetime(raw[TIME], errors="coerce")
    missing = [c for c in MEASURED_COLS if c not in raw.columns]
    available = [c for c in MEASURED_COLS if c in raw.columns]
    if not available or not any(c in raw.columns for c in CTRL_COLS):
        raise ValueError("No signed Ctrl channels found; this is not a compatible controller log")
    nonnumeric = {}
    for c in available:
        before = raw[c].notna().sum()
        raw[c] = _numeric(raw[c])
        nonnumeric[c] = int(before - raw[c].notna().sum())
    invalid_time = int(raw[TIME].isna().sum())
    raw = raw.dropna(subset=[TIME]).copy()
    duplicated = int(raw[TIME].duplicated().sum())
    out_of_order = int((raw[TIME].diff().dt.total_seconds() < 0).sum())
    raw = raw.sort_values([TIME, "SourceRow"], kind="stable").reset_index(drop=True)
    gaps = raw[TIME].diff().dt.total_seconds()
    sample_median = float(gaps[gaps > 0].median()) if (gaps > 0).any() else np.nan
    qa = {
        "Source": path.name, "Original rows": int(len(raw) + invalid_time),
        "Valid timestamp rows": len(raw), "Invalid timestamp rows": invalid_time,
        "Duplicate timestamps": duplicated, "Out-of-order timestamps": out_of_order,
        "Median sample period (s)": sample_median,
        "Gaps > 5s": int((gaps > 5).sum()),
        "RH exact zeros": int((raw["RH"] == 0).sum()) if "RH" in raw else None,
        "RH outside 0-100%": int(((raw["RH"] < 0) | (raw["RH"] > 100)).sum()) if "RH" in raw else None,
        "Control values outside [-1, 1]": int(sum(((raw[c] < -1) | (raw[c] > 1)).sum() for c in CTRL_COLS if c in raw)),
        "Nonnumeric values": {c: n for c, n in nonnumeric.items() if n},
        "Missing channels": missing,
        "Empty CSV columns dropped": discarded,
        "Log header timestamp": stamp if stamp != "Timestamp" else "",
    }
    return LoadedLog(name or path.stem, raw, qa)


def discover_logs(folder: str | Path, *, recursive: bool = False) -> list[Path]:
    folder = Path(folder).expanduser()
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Log directory not found: {folder}")
    matches = folder.rglob("*.csv") if recursive else folder.glob("*.csv")
    return sorted(p for p in matches if p.is_file())


def load_folder(folder: str | Path, *, recursive: bool = False) -> tuple[dict[str, LoadedLog], dict[str, str]]:
    """Loads all matching CSVs, reports non-controller CSVs instead of failing entire batch."""
    logs: dict[str, LoadedLog] = {}
    failures: dict[str, str] = {}
    for path in discover_logs(folder, recursive=recursive):
        try:
            log = read_controller_log(path)
            logs[path.name] = log
        except (ValueError, OSError, pd.errors.ParserError) as exc:
            failures[path.name] = str(exc)
    return logs, failures


def select_window(log: LoadedLog | pd.DataFrame, *, start_row: int | None = None,
                  end_row: int | None = None, start_time: str | pd.Timestamp | None = None,
                  end_time: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """Inclusive source-CSV row range and inclusive absolute timestamp filter."""
    df = (log.frame if isinstance(log, LoadedLog) else log).copy()
    if start_row is not None:
        if start_row < 2:
            raise ValueError("CSV header is row 1; first data row is row 2")
        df = df.loc[df.SourceRow >= start_row]
    if end_row is not None:
        if end_row < 2 or (start_row is not None and end_row < start_row):
            raise ValueError("End row must be >= start row and >= 2")
        df = df.loc[df.SourceRow <= end_row]
    if start_time not in (None, ""):
        ts = pd.to_datetime(start_time, errors="coerce")
        if pd.isna(ts):
            raise ValueError("Invalid start time; use YYYY-MM-DD HH:MM:SS")
        df = df.loc[df[TIME] >= ts]
    if end_time not in (None, ""):
        ts = pd.to_datetime(end_time, errors="coerce")
        if pd.isna(ts):
            raise ValueError("Invalid end time; use YYYY-MM-DD HH:MM:SS")
        df = df.loc[df[TIME] <= ts]
    return df.sort_values([TIME, "SourceRow"], kind="stable").reset_index(drop=True)


def augment(df: pd.DataFrame, *, mask_rh_zeros: bool = False,
            smooth_seconds: int = 60, max_gap_seconds: float = 5) -> pd.DataFrame:
    """Create loss-aware derived features. No interpolation or outlier clipping.

    RH=0 is retained by default. Optional masking applies only to derived analysis;
    original RH stays available as RH_Original. Time derivatives respect gaps.
    """
    if max_gap_seconds <= 0 or smooth_seconds < 0:
        raise ValueError("max_gap_seconds must be positive; smoothing cannot be negative")
    d = df.copy().sort_values([TIME, "SourceRow"], kind="stable")
    if d.empty:
        return d
    d["Elapsed_min"] = (d[TIME] - d[TIME].iloc[0]).dt.total_seconds() / 60
    d["Delta_t_s"] = d[TIME].diff().dt.total_seconds()
    d["LargeGapBefore"] = d["Delta_t_s"] > max_gap_seconds
    if "RH" in d:
        d["RH_Original"] = d["RH"]
        d["RH_Flag_Zero"] = d["RH"].eq(0)
        d["RH_Flag_OutOfRange"] = ~(d["RH"].between(0, 100) | d["RH"].isna())
        if mask_rh_zeros:
            d.loc[d["RH_Flag_Zero"], "RH"] = np.nan
        d.loc[d["RH_Flag_OutOfRange"], "RH"] = np.nan
    for c in PROCESS_COLS:
        if c not in d:
            continue
        dt = d["Delta_t_s"]
        delta = d[c].diff()
        d[f"{c} rate/min"] = (delta / dt * 60).where((dt > 0) & (dt <= max_gap_seconds))
    # Dew point and absolute humidity represent chamber air only if the RH probe
    # samples that same air. This assumption is documented, not a calibration.
    if "T Chamber" in d and "RH" in d:
        t, rh = d["T Chamber"], d["RH"]
        valid = t.between(-40, 80) & rh.gt(0) & rh.le(100)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            gamma = np.log(rh / 100) + (17.62 * t) / (243.12 + t)
            d["DewPoint_C"] = ((243.12 * gamma) / (17.62 - gamma)).where(valid)
            d["AbsoluteHumidity_g_m3"] = (
                6.112 * np.exp((17.67 * t) / (t + 243.5)) * rh * 2.1674 / (273.15 + t)
            ).where(valid)
    available = [c for c in TEMP_GROUP if c in d]
    if len(available) >= 2:
        # Components reside in different thermal zones; not spatial chamber uniformity.
        d["TemperatureZoneRange_C"] = d[available].max(axis=1) - d[available].min(axis=1)
    if smooth_seconds > 0:
        # Group duplicate timestamps for rolling view only (original d remains intact).
        indexed = d.set_index(TIME)
        for c in PROCESS_COLS:
            if c in d:
                unique = indexed[c].groupby(level=0).mean().sort_index()
                rolling = unique.rolling(f"{smooth_seconds}s", min_periods=2).mean()
                d[f"{c} MA_{smooth_seconds}s"] = d[TIME].map(rolling)
    return d.reset_index(drop=True)


def observed_weights(df: pd.DataFrame, max_gap_seconds: float = 5) -> pd.Series:
    """Weight sample i by interval to sample i+1; discard uncovered long gaps."""
    next_dt = (df[TIME].shift(-1) - df[TIME]).dt.total_seconds()
    return next_dt.where((next_dt > 0) & (next_dt <= max_gap_seconds), 0).fillna(0)


def descriptive_statistics(df: pd.DataFrame, *, max_gap_seconds: float = 5,
                           targets: Mapping[str, tuple[float, float]] | None = None) -> pd.DataFrame:
    """Sample statistics; time-in-band is weighted only across observed intervals."""
    weights = observed_weights(df, max_gap_seconds)
    rows = []
    for c in PROCESS_COLS:
        if c not in df:
            continue
        s = df[c].dropna()
        target, tol = (targets or {}).get(c, (None, None))
        covered = float(weights.where(df[c].notna(), 0).sum())
        in_band = np.nan
        if target is not None and tol is not None and tol >= 0 and covered > 0:
            in_band = float(weights.where(df[c].between(target - tol, target + tol), 0).sum() / covered * 100)
        rows.append({
            "Channel": c, "Samples": int(s.size), "Min": s.min() if len(s) else np.nan,
            "Mean": s.mean() if len(s) else np.nan,
            "Median": s.median() if len(s) else np.nan,
            "Max": s.max() if len(s) else np.nan,
            "StdDev": s.std() if len(s) else np.nan,
            "P05": s.quantile(.05) if len(s) else np.nan,
            "P95": s.quantile(.95) if len(s) else np.nan,
            "Range": s.max() - s.min() if len(s) else np.nan,
            "Covered_min": covered / 60, "Reference": target,
            "Tolerance": tol, "Time_in_band_pct": in_band,
        })
    return pd.DataFrame(rows)


def control_statistics(df: pd.DataFrame, *, neutral_band: float = 0.02,
                       saturation_level: float = .98, max_gap_seconds: float = 5) -> pd.DataFrame:
    """Positive/negative controls are sign classes, not assumed heater/cooler actions."""
    if neutral_band < 0 or not (0 < saturation_level <= 1):
        raise ValueError("neutral_band >= 0 and 0 < saturation_level <= 1 required")
    w = observed_weights(df, max_gap_seconds)
    rows = []
    for c in CTRL_COLS:
        if c not in df:
            continue
        v = df[c]
        usable = w.where(v.notna(), 0)
        total = float(usable.sum())
        positive = v > neutral_band
        negative = v < -neutral_band
        neutral = v.notna() & ~(positive | negative)
        state = pd.Series(np.select([positive, negative], [1, -1], default=0), index=df.index)
        nz = state[state != 0]
        flips = int((nz != nz.shift()).sum() - int(len(nz) > 0))
        rows.append({
            "Controller": c, "Samples": int(v.notna().sum()),
            "Mean": v.mean(), "Min": v.min(), "Max": v.max(),
            "Mean_abs": v.abs().mean(),
            "Positive_time_pct": float(usable.where(positive, 0).sum() / total * 100) if total else np.nan,
            "Negative_time_pct": float(usable.where(negative, 0).sum() / total * 100) if total else np.nan,
            "Neutral_time_pct": float(usable.where(neutral, 0).sum() / total * 100) if total else np.nan,
            "Positive_saturation_pct": float(usable.where(v >= saturation_level, 0).sum() / total * 100) if total else np.nan,
            "Negative_saturation_pct": float(usable.where(v <= -saturation_level, 0).sum() / total * 100) if total else np.nan,
            "Sign_changes": max(flips, 0), "Covered_min": total / 60,
            "Signed_exposure_min": float((v.fillna(0) * usable).sum() / 60),
        })
    return pd.DataFrame(rows)


def _runs_of_true(df: pd.DataFrame, mask: pd.Series, max_gap_seconds: float) -> list[tuple[int, int]]:
    """Contiguous observed runs; split when time gap exceeds tolerated coverage."""
    valid = mask.fillna(False).to_numpy(dtype=bool)
    timestamps = df[TIME].to_numpy()
    runs = []
    start = None
    for i, yes in enumerate(valid):
        if start is not None and i > 0:
            dt = (timestamps[i] - timestamps[i-1]) / np.timedelta64(1, 's')
            if not np.isfinite(dt) or dt > max_gap_seconds or dt <= 0:
                runs.append((start, i - 1))
                start = None
        if yes and start is None:
            start = i
        if not yes and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(df) - 1))
    return runs


def first_sustained_threshold(df: pd.DataFrame, *, temp_threshold: float = 36.5,
                              rh_threshold: float = 85.0, hold_s: float = 30,
                              max_gap_seconds: float = 5) -> dict:
    """First observed crossing and first observed run lasting hold_s seconds."""
    if hold_s < 0:
        raise ValueError("hold_s must not be negative")
    if df.empty:
        return {}
    start = df[TIME].iloc[0]
    t = df["T Chamber"].ge(temp_threshold) if "T Chamber" in df else pd.Series(False, index=df.index)
    rh = df["RH"].ge(rh_threshold) if "RH" in df else pd.Series(False, index=df.index)
    conditions = {"Temperature": t, "RH": rh, "Both": t & rh}
    out = {"Temperature threshold": temp_threshold, "RH threshold": rh_threshold, "Hold requirement (s)": hold_s}
    for key, mask in conditions.items():
        hits = df.loc[mask]
        out[f"{key}: first crossing (min)"] = float((hits[TIME].iloc[0] - start).total_seconds() / 60) if not hits.empty else np.nan
        sustained = np.nan
        confirmed = np.nan
        for lo, hi in _runs_of_true(df, mask, max_gap_seconds):
            if (df[TIME].iloc[hi] - df[TIME].iloc[lo]).total_seconds() >= hold_s:
                sustained = (df[TIME].iloc[lo] - start).total_seconds() / 60
                end_target = df[TIME].iloc[lo] + pd.Timedelta(seconds=hold_s)
                confirmed_idx = df[TIME].iloc[lo:hi+1].searchsorted(end_target, side="left") + lo
                confirmed = (df[TIME].iloc[confirmed_idx] - start).total_seconds() / 60
                break
        out[f"{key}: sustained start (min)"] = sustained
        out[f"{key}: hold confirmed (min)"] = confirmed
    return out


def find_excursions(df: pd.DataFrame, channel: str, target: float, tolerance: float,
                    *, min_duration_s: float = 5, recovery_hold_s: float = 30,
                    max_gap_seconds: float = 5) -> pd.DataFrame:
    """Identify recorded out-of-band runs and require observed in-band recovery hold.

    The event duration is last-to-first outside sample (lower bound on excursion).
    If a sufficiently long in-band run immediately follows, recovery is the first
    in-band observation; confirmed timestamp is when the hold has been observed.
    """
    columns = ["Channel", "Start", "Last_outside", "Outside_observed_s", "Samples",
               "Direction", "Worst_reading", "Worst_at", "Peak_abs_error", "Recovered", "Recovery_start",
               "Recovery_confirmed", "Recovery_from_start_min"]
    if df.empty or channel not in df:
        return pd.DataFrame(columns=columns)
    if tolerance < 0 or min_duration_s < 0 or recovery_hold_s < 0:
        raise ValueError("Tolerances and hold lengths cannot be negative")
    series = df[channel]
    outside = series.notna() & ~series.between(target - tolerance, target + tolerance)
    inside = series.between(target - tolerance, target + tolerance)
    inside_runs = {lo: hi for lo, hi in _runs_of_true(df, inside, max_gap_seconds)}
    rows = []
    for lo, hi in _runs_of_true(df, outside, max_gap_seconds):
        duration = (df[TIME].iloc[hi] - df[TIME].iloc[lo]).total_seconds()
        if duration < min_duration_s:
            continue
        v = series.iloc[lo:hi + 1]
        errors = (v - target).abs()
        worst = float(v.loc[errors.idxmax()])
        recovered = False
        recovery_start = pd.NaT
        recovery_confirmed = pd.NaT
        recovery_from_start_min = np.nan
        nidx = hi + 1
        if nidx in inside_runs:
            close_gap = (df[TIME].iloc[nidx] - df[TIME].iloc[hi]).total_seconds()
            end_idx = inside_runs[nidx]
            if close_gap > 0 and close_gap <= max_gap_seconds and (df[TIME].iloc[end_idx] - df[TIME].iloc[nidx]).total_seconds() >= recovery_hold_s:
                recovered = True
                recovery_start = df[TIME].iloc[nidx]
                stop = df[TIME].iloc[nidx] + pd.Timedelta(seconds=recovery_hold_s)
                at = df[TIME].iloc[nidx:end_idx + 1].searchsorted(stop, side="left") + nidx
                recovery_confirmed = df[TIME].iloc[at]
                recovery_from_start_min = (recovery_start - df[TIME].iloc[lo]).total_seconds() / 60
        rows.append({"Channel": channel, "Start": df[TIME].iloc[lo],
                     "Last_outside": df[TIME].iloc[hi], "Outside_observed_s": duration,
                     "Samples": hi - lo + 1, "Direction": "low" if worst < target else "high",
                     "Worst_reading": worst, "Worst_at": df.loc[errors.idxmax(), TIME], "Peak_abs_error": float(errors.max()),
                     "Recovered": recovered, "Recovery_start": recovery_start,
                     "Recovery_confirmed": recovery_confirmed,
                     "Recovery_from_start_min": recovery_from_start_min})
    return pd.DataFrame(rows, columns=columns)


def detect_drop_recovery(df: pd.DataFrame, *, thresholds: Mapping[str, float] | None = None,
                         hold_s: float = 30, max_gap_seconds: float = 5,
                         min_drop_duration_s: float = 0,
                         include_combined: bool = True) -> pd.DataFrame:
    """Observed BELOW-threshold episodes and subsequent sustained threshold recovery.

    An event begins at the first sampled low value; it ends only after an above-
    threshold run lasting hold_s. Short temporary returns are part of the same
    event. No interpolation is used. Long sampling gaps and null measurements
    break episodes; unresolved events cannot be assumed to recover across gaps.
    A low value at the left edge is marked as left-censored (onset unknown).
    """
    thresholds = dict(thresholds or {"T Chamber": 36.5, "RH": 85.0, "CO2": 47000.0})
    columns = ["Channel", "Threshold", "Drop_start", "Last_below", "Worst_value",
               "Worst_at", "Depth_below_threshold", "Below_samples",
               "Below_observed_span_s", "Left_censored", "Recovered",
               "Recovery_start", "Recovery_confirmed", "Recovery_min",
               "Confirmation_min", "Status"]
    if hold_s < 0 or max_gap_seconds <= 0 or min_drop_duration_s < 0:
        raise ValueError("Hold, maximum gap and minimum drop duration must be valid")
    if df.empty:
        return pd.DataFrame(columns=columns)
    work = df.sort_values(TIME, kind="stable").drop_duplicates(subset=[TIME], keep="first").reset_index(drop=True)
    channels = [(key, val) for key, val in thresholds.items() if key in work.columns]
    if include_combined and len(channels) >= 2:
        channels.append(("Any below threshold", np.nan))
    rows = []
    timestamps = work[TIME]
    for channel, threshold in channels:
        combined = channel == "Any below threshold"
        used = [key for key, _ in channels if key != "Any below threshold"] if combined else [channel]
        valid = work[used].notna().all(axis=1).to_numpy(dtype=bool)
        if combined:
            below = np.logical_or.reduce([work[key].lt(value).fillna(False).to_numpy(dtype=bool)
                                          for key, value in thresholds.items() if key in used])
        else:
            below = work[channel].lt(threshold).fillna(False).to_numpy(dtype=bool)
        n = len(work)
        segments = []
        first = None
        for i in range(n):
            gap = i > 0 and ((timestamps.iloc[i]-timestamps.iloc[i-1]).total_seconds() <= 0 or
                            (timestamps.iloc[i]-timestamps.iloc[i-1]).total_seconds() > max_gap_seconds)
            if (gap or not valid[i]) and first is not None:
                segments.append((first, i-1))
                first = None
            if valid[i] and first is None:
                first = i
        if first is not None:
            segments.append((first,n-1))
        for segment_start, segment_end in segments:
            i = segment_start
            while i <= segment_end:
                if not below[i]:
                    i += 1
                    continue
                onset = i
                left_censored = onset == segment_start
                last_low = i
                recovery = None
                confirmed = None
                j = i+1
                while j <= segment_end:
                    if below[j]:
                        last_low = j
                        j += 1
                        continue
                    good_start = j
                    while j <= segment_end and not below[j]:
                        j += 1
                    if (timestamps.iloc[j-1] - timestamps.iloc[good_start]).total_seconds() >= hold_s:
                        recovery = good_start
                        goal = timestamps.iloc[good_start]+pd.Timedelta(seconds=hold_s)
                        confirmed = int(timestamps.iloc[good_start:j].searchsorted(goal,side="left"))+good_start
                        break
                    # An unconfirmed above-threshold run is not a recovery.
                low_idxs = np.flatnonzero(below[onset:last_low+1]) + onset
                span = (timestamps.iloc[last_low]-timestamps.iloc[onset]).total_seconds()
                if span >= min_drop_duration_s:
                    if combined:
                        normalized = pd.DataFrame({key: thresholds[key]-work[key].iloc[onset:last_low+1]
                                                   for key in used})
                        depth = normalized.max(axis=1)
                        worst_local = depth.idxmax()
                        worst = np.nan
                        worst_depth = float(depth.max())
                    else:
                        series=work[channel].iloc[onset:last_low+1]
                        worst_local = series.idxmin()
                        worst = float(series.min())
                        worst_depth = float(threshold-worst)
                    recovered = recovery is not None
                    start_time = timestamps.iloc[onset]
                    recovery_start = timestamps.iloc[recovery] if recovered else pd.NaT
                    recovery_confirmed = timestamps.iloc[confirmed] if recovered else pd.NaT
                    rows.append({"Channel": channel, "Threshold": threshold,
                        "Drop_start": start_time, "Last_below": timestamps.iloc[last_low],
                        "Worst_value": worst, "Worst_at": timestamps.iloc[worst_local],
                        "Depth_below_threshold": worst_depth, "Below_samples": int(len(low_idxs)),
                        "Below_observed_span_s": span, "Left_censored": bool(left_censored),
                        "Recovered": recovered, "Recovery_start": recovery_start,
                        "Recovery_confirmed": recovery_confirmed,
                        "Recovery_min": (recovery_start-start_time).total_seconds()/60 if recovered else np.nan,
                        "Confirmation_min": (recovery_confirmed-start_time).total_seconds()/60 if recovered else np.nan,
                        "Status": ("left-censored; " if left_censored else "") + ("recovered" if recovered else "not confirmed")})
                i = confirmed+1 if recovery is not None else segment_end+1
    return pd.DataFrame(rows,columns=columns)


def plot_drop_recovery(df: pd.DataFrame, events: pd.DataFrame,
                       thresholds: Mapping[str, float] | None = None) -> go.Figure:
    """Three synchronized measured profiles and markers for observed drops/recovery."""
    thresholds = dict(thresholds or {"T Chamber": 36.5,"RH": 85,"CO2": 47000})
    channels=[c for c in ("T Chamber","RH","CO2") if c in df and c in thresholds]
    fig=make_subplots(rows=len(channels),cols=1,shared_xaxes=True,
                      subplot_titles=[f"{c} (threshold {thresholds[c]:g})" for c in channels],
                      vertical_spacing=.09)
    for r,c in enumerate(channels,1):
        fig.add_trace(go.Scatter(x=df[TIME],y=df[c],name=c,mode="lines",
                                 line=dict(color=COLOURS.get(c))),row=r,col=1)
        fig.add_hline(y=thresholds[c],row=r,col=1,line_dash="dash",line_color="#dc2626")
        if not events.empty:
            subset=events[events.Channel.eq(c)]
            fig.add_trace(go.Scatter(x=subset.Drop_start,y=[thresholds[c]]*len(subset),
                                     mode="markers",name=f"{c} drops",legendgroup=f"{c} drops",
                                     marker=dict(color="#dc2626",size=9,symbol="triangle-down")),row=r,col=1)
            recovered=subset[subset.Recovered]
            fig.add_trace(go.Scatter(x=recovered.Recovery_start,y=[thresholds[c]]*len(recovered),
                                     mode="markers",name=f"{c} recovery",legendgroup=f"{c} recovery",
                                     marker=dict(color="#16a34a",size=9,symbol="circle")),row=r,col=1)
    fig.update_xaxes(title_text="Logged timestamp",row=len(channels),col=1)
    return _layout(fig,"Below-threshold drops and observed recovery",height=300*len(channels)+100)


def plot_recovery_bars(events: pd.DataFrame) -> go.Figure:
    """Recovered episode times in minutes; unresolved episodes omitted, not zeroed."""
    fig=go.Figure()
    if not events.empty:
        recovered=events[events.Recovered & events.Channel.ne("Any below threshold")].copy()
        if not recovered.empty:
            recovered["Event"] = recovered.Channel + " • " + recovered.Drop_start.dt.strftime("%H:%M:%S")
            fig.add_trace(go.Bar(x=recovered.Event,y=recovered.Recovery_min,
                                 marker_color=[COLOURS.get(c,"#64748b") for c in recovered.Channel],
                                 text=recovered.Recovery_min.round(2),textposition="outside",
                                 hovertemplate="%{x}<br>Observed recovery: %{y:.3f} min<extra></extra>"))
    fig.update_yaxes(title_text="Observed recovery time (minutes)")
    fig.update_xaxes(title_text="Channel and drop onset")
    return _layout(fig,"Recovery durations by drop episode (confirmed only)",height=520)


def lag_correlation(df: pd.DataFrame, *, measurement: str = "RH", controller: str = "Ctrl RH",
                    bin_seconds: int = 10, max_lag_seconds: int = 300,
                    min_overlaps: int = 15) -> pd.DataFrame:
    """Exploratory Pearson corr(u(t), y(t+lag)); no causal inference."""
    if measurement not in df or controller not in df:
        raise ValueError("Requested channels not present")
    if bin_seconds <= 0 or max_lag_seconds < 0:
        raise ValueError("Invalid bin/lag interval")
    d = (df.set_index(TIME)[[controller, measurement]]
           .groupby(level=0).mean().resample(f"{bin_seconds}s").mean())
    if len(d) > 500000:
        raise ValueError("Too many resampled bins for lag analysis; increase bin_seconds")
    limit = max_lag_seconds // bin_seconds
    rows = []
    for k in range(-limit, limit + 1):
        a = d[controller]
        b = d[measurement].shift(-k)
        valid = a.notna() & b.notna()
        n = int(valid.sum())
        corr = float(a.loc[valid].corr(b.loc[valid])) if n >= min_overlaps and a.loc[valid].std() > 0 and b.loc[valid].std() > 0 else np.nan
        rows.append({"Lag_s": k * bin_seconds, "Pearson_r": corr, "Overlapping_bins": n})
    return pd.DataFrame(rows)


def _layout(fig: go.Figure, title: str, *, height: int = 580) -> go.Figure:
    fig.update_layout(template="plotly_white", title=dict(text=title, x=.02),
                      height=height, margin=dict(l=55, r=25, t=80, b=60),
                      hovermode="x unified", legend=dict(orientation="h", y=-.12),
                      font=dict(family="Arial, sans-serif", size=12))
    return fig


def _plot_frame(df: pd.DataFrame, resample: str | None) -> pd.DataFrame:
    if not resample or resample.lower() == "none":
        return df
    cols = [c for c in MEASURED_COLS if c in df]
    # Resample for display only; no filling/interpolation.
    return (df.set_index(TIME)[cols].groupby(level=0).mean()
            .resample(resample).mean().reset_index())


def plot_environment(df: pd.DataFrame, *, resample: str | None = None,
                     show_stc31: bool = True, overlay_moving_average: bool = False,
                     targets: Mapping[str, tuple[float, float]] | None = None) -> go.Figure:
    """Separate axes by incompatible physical measurement and temperature scale."""
    d = _plot_frame(df, resample)
    groups = [["T Steam"], TEMP_GROUP, ["RH"], ["CO2"]]
    if show_stc31:
        groups.append(["STC31"])
    captions = ["Steam temperature", "Chamber / component temperatures", "Relative humidity",
                "CO2 — raw logged unit"] + (["STC31 — raw logged unit"] if show_stc31 else [])
    fig = make_subplots(rows=len(groups), cols=1, shared_xaxes=True, vertical_spacing=.055,
                        subplot_titles=captions)
    for r, group in enumerate(groups, start=1):
        for c in group:
            if c not in d:
                continue
            fig.add_trace(go.Scatter(x=d[TIME], y=d[c], mode="lines", name=c,
                legendgroup=c, line=dict(color=COLOURS.get(c), width=1.6),
                connectgaps=False), row=r, col=1)
            if overlay_moving_average:
                matches = [key for key in df.columns if key.startswith(f"{c} MA_")]
                if matches:
                    smoothed = df[[TIME, matches[-1]]].groupby(TIME).mean().sort_index()
                    if resample:
                        smoothed = smoothed.resample(resample).mean()
                    fig.add_trace(go.Scatter(x=smoothed.index, y=smoothed[matches[-1]],
                         name=f"{c}: moving mean", mode="lines", connectgaps=False,
                         line=dict(color=COLOURS.get(c), dash="dot", width=2)), row=r, col=1)
            if targets and c in targets:
                ref, tol = targets[c]
                fig.add_hline(y=ref - tol, row=r, col=1, line_width=1,
                              line_dash="dot", line_color="#64748b")
                fig.add_hline(y=ref + tol, row=r, col=1, line_width=1,
                              line_dash="dot", line_color="#64748b")
    fig.update_xaxes(title_text="Logger timestamp (device local)", row=len(groups), col=1)
    return _layout(fig, "Environmental & component profiles — measured readings", height=280 * len(groups) + 130)


def plot_controller_signals(df: pd.DataFrame, *, resample: str | None = None,
                           show_controls: Iterable[str] | None = None) -> go.Figure:
    d = _plot_frame(df, resample)
    chosen = [c for c in (show_controls or CTRL_COLS) if c in d]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.12,
                        subplot_titles=["Thermal controllers (signed output)", "RH and CO2 controllers (signed output)"])
    palette = ["#c2410c", "#2563eb", "#0d9488", "#8b5cf6", "#d97706", "#0891b2", "#b45309"]
    for c in chosen:
        r = 2 if c in {"Ctrl RH", "Ctrl CO2"} else 1
        fig.add_trace(go.Scatter(x=d[TIME], y=d[c], name=c, mode="lines", connectgaps=False,
                                 line=dict(width=1.3, color=palette[CTRL_COLS.index(c)])), row=r, col=1)
    for r in [1, 2]:
        fig.add_hline(y=0, line_color="#475569", line_width=1, row=r, col=1)
        fig.update_yaxes(title_text="Signed PID output", row=r, col=1)
    fig.update_xaxes(title_text="Logger timestamp (device local)", row=2, col=1)
    return _layout(fig, "PID commands — positive / negative / saturation", height=750)


def plot_pair(df: pd.DataFrame, measurement: str, *, resample: str | None = None,
              target: tuple[float, float] | None = None) -> go.Figure:
    """Two aligned plots: physical response above, matching PID command below."""
    if measurement not in CONTROL_PAIRS:
        raise ValueError("Choose from: " + ", ".join(CONTROL_PAIRS))
    ctrl = CONTROL_PAIRS[measurement]
    d = _plot_frame(df, resample)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.12,
                        subplot_titles=[f"Measured: {measurement}", f"Controller: {ctrl}"])
    if measurement in d:
        fig.add_trace(go.Scatter(x=d[TIME], y=d[measurement], name=measurement,
                                 line=dict(color=COLOURS.get(measurement, "#2563eb"), width=1.8)), row=1, col=1)
    if target:
        center, tolerance = target
        for level in [center - tolerance, center + tolerance]:
            fig.add_hline(y=level, row=1, col=1, line_color="#94a3b8", line_dash="dot")
    if ctrl in d:
        fig.add_trace(go.Scatter(x=d[TIME], y=d[ctrl], name=ctrl,
                                 line=dict(color="#f97316", width=1.5)), row=2, col=1)
        fig.add_hline(y=0, row=2, col=1, line_color="#475569", line_width=1)
    fig.update_xaxes(title_text="Logger timestamp (device local)", row=2, col=1)
    fig.update_yaxes(title_text="Measured value", row=1, col=1)
    fig.update_yaxes(title_text="Signed control", row=2, col=1)
    return _layout(fig, f"Measured response and PID control: {measurement}", height=720)


def plot_statistics(stats: pd.DataFrame, *, group: str = "Temperature") -> go.Figure:
    if group == "Temperature":
        subset = stats[stats.Channel.isin(TEMP_COLS)]
        ytitle = "Temperature (°C, assumed logged unit)"
    elif group == "RH":
        subset = stats[stats.Channel.eq("RH")]
        ytitle = "Relative humidity (%)"
    elif group == "CO2":
        subset = stats[stats.Channel.eq("CO2")]
        ytitle = "CO2 (raw logged unit)"
    elif group == "STC31":
        subset = stats[stats.Channel.eq("STC31")]
        ytitle = "STC31 (raw logged unit)"
    else:
        raise ValueError("group must be Temperature, RH, CO2 or STC31")
    fig = go.Figure()
    for metric, color in [("Min", "#4f7cac"), ("Mean", "#0d9488"), ("Max", "#ef8354")]:
        fig.add_bar(name=metric, x=subset.Channel, y=subset[metric], marker_color=color,
                    text=subset[metric].round(2), textposition="outside")
    fig.update_layout(barmode="group", xaxis_title="Channel", yaxis_title=ytitle)
    # Important: no zero baseline for narrow temperature/RH bars (allows differences to be read).
    return _layout(fig, f"Min / mean / max — {group}", height=480)


def plot_control_summary(summary: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for c, label, color in [
        ("Positive_time_pct", "+ high", "#0d9488"),
        ("Negative_time_pct", "- low", "#ea580c"),
        ("Neutral_time_pct", "Near zero", "#94a3b8"),
    ]:
        fig.add_bar(x=summary.Controller, y=summary[c], name=label, marker_color=color)
    fig.update_layout(barmode="stack", yaxis_title="% of covered observation time",
                      xaxis_title="Controller", yaxis=dict(range=[0, 100]))
    return _layout(fig, "Controller sign distribution (time weighted)", height=500)


def plot_correlation(df: pd.DataFrame) -> go.Figure:
    cols = [c for c in MEASURED_COLS if c in df and df[c].notna().sum() >= 3 and df[c].nunique() > 1]
    if not cols:
        return _layout(go.Figure(), "No valid channels for correlation")
    corr = df[cols].corr()
    fig = go.Figure(go.Heatmap(z=corr.to_numpy(), x=cols, y=cols,
                               colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
                               colorbar=dict(title="Pearson r"),
                               hovertemplate="%{y} vs %{x}: %{z:.3f}<extra></extra>"))
    fig.update_layout(xaxis=dict(tickangle=-45), yaxis=dict(autorange="reversed"))
    return _layout(fig, "Exploratory correlation (not evidence of causation)", height=760)


def plot_lag_correlation(lags: pd.DataFrame, *, measurement: str, controller: str) -> go.Figure:
    fig = go.Figure(go.Scatter(x=lags.Lag_s, y=lags.Pearson_r, mode="lines+markers",
                              line=dict(color="#2563eb"), name="Pearson r"))
    fig.add_vline(x=0, line_dash="dot", line_color="#94a3b8")
    fig.update_xaxes(title_text="Lag (seconds): positive = response AFTER control")
    fig.update_yaxes(title_text="Pearson r", range=[-1, 1])
    return _layout(fig, f"Exploratory lag — {controller} vs {measurement}", height=460)


def plot_excursions(df: pd.DataFrame, channel: str, target: float, tolerance: float,
                   events: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[TIME], y=df[channel], name=channel,
                             mode="lines", line=dict(color=COLOURS.get(channel,"#2563eb"))))
    fig.add_hrect(y0=target-tolerance, y1=target+tolerance, fillcolor="#10b981",
                  opacity=.10, line_width=0)
    if not events.empty:
        fig.add_trace(go.Scatter(x=events.Worst_at, y=events.Worst_reading,
                                 mode="markers", name="Out-of-band events",
                                 marker=dict(size=9, symbol="x", color="#dc2626")))
    fig.update_xaxes(title_text="Logger timestamp (device local)")
    fig.update_yaxes(title_text=channel)
    return _layout(fig, f"{channel} excursions; reference {target:g} ± {tolerance:g}", height=460)


def build_report(log: LoadedLog, *, start_row: int | None = None, end_row: int | None = None,
                 start_time: str | None = None, end_time: str | None = None,
                 mask_rh_zeros: bool = False, smooth_seconds: int = 60,
                 max_gap_seconds: float = 5, neutral_band: float = 0.02,
                 saturation_level: float = .98, chamber_target: float = 37.0,
                 chamber_tolerance: float = .5, rh_target: float = 90.0,
                 rh_tolerance: float = 1.0, hold_seconds: float = 30,
                 excursion_min_seconds: float = 5,
                 co2_target: float | None = None, co2_tolerance: float | None = None,
                 temp_threshold: float = 36.5, rh_threshold: float = 85.0,
                 co2_threshold: float = 47000.0) -> dict:
    """Consistent single-window report. Reference values are operator inputs, NOT setpoints."""
    df = select_window(log, start_row=start_row, end_row=end_row,
                       start_time=start_time, end_time=end_time)
    if df.empty:
        raise ValueError("No records in the selected row/time window")
    clean = augment(df, mask_rh_zeros=mask_rh_zeros,
                    smooth_seconds=smooth_seconds, max_gap_seconds=max_gap_seconds)
    targets = {"T Chamber": (chamber_target, chamber_tolerance),
               "RH": (rh_target, rh_tolerance)}
    if co2_target is not None and co2_tolerance is not None:
        targets["CO2"] = (co2_target, co2_tolerance)
    summary = descriptive_statistics(clean, targets=targets, max_gap_seconds=max_gap_seconds)
    controls = control_statistics(clean, max_gap_seconds=max_gap_seconds,
                                  neutral_band=neutral_band, saturation_level=saturation_level)
    threshold = first_sustained_threshold(clean, temp_threshold=temp_threshold,
                                          rh_threshold=rh_threshold, hold_s=hold_seconds,
                                          max_gap_seconds=max_gap_seconds)
    event_frames = [find_excursions(clean, ch, target, tol,
                    min_duration_s=excursion_min_seconds, recovery_hold_s=hold_seconds,
                    max_gap_seconds=max_gap_seconds) for ch, (target, tol) in targets.items()]
    event_frames = [f for f in event_frames if not f.empty]
    events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    drops = detect_drop_recovery(clean, thresholds={"T Chamber": temp_threshold,
                                 "RH": rh_threshold, "CO2": co2_threshold},
                                 hold_s=hold_seconds, max_gap_seconds=max_gap_seconds)
    return {"source": log.name, "data": clean, "quality": log.quality, "stats": summary,
            "controllers": controls, "threshold": pd.DataFrame([threshold]), "events": events,
            "drop_recovery": drops,
            "targets": targets,
            "parameters": {"mask_rh_zeros": mask_rh_zeros, "max_gap_seconds": max_gap_seconds,
                           "smooth_seconds": smooth_seconds, "chamber_target": chamber_target,
                           "chamber_tolerance": chamber_tolerance, "rh_target": rh_target,
                           "rh_tolerance": rh_tolerance, "hold_seconds": hold_seconds,
                           "co2_target": co2_target, "co2_tolerance": co2_tolerance,
                           "temp_threshold": temp_threshold, "rh_threshold": rh_threshold,
                           "co2_threshold": co2_threshold}}


def export_report(report: dict, dest: str | Path, *, charts: Mapping[str, go.Figure] | None = None) -> Path:
    """Create named CSV outputs and optional offline interactive HTML."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for label, key in [("processed_readings", "data"), ("statistics", "stats"),
                       ("pid_statistics", "controllers"), ("threshold_reach", "threshold"),
                       ("excursions", "events"), ("drop_recovery", "drop_recovery")]:
        report[key].to_csv(dest / f"{label}.csv", index=False)
    (dest / "quality_report.json").write_text(
        pd.Series({"source": report["source"], "quality": report["quality"],
                   "parameters": report["parameters"]}).to_json(indent=2), encoding="utf-8"
    )
    for label, fig in (charts or {}).items():
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", label)
        fig.write_html(str(dest / f"{safe}.html"), include_plotlyjs=True)
    return dest


def _main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Batch process incubator environmental/controller logs")
    parser.add_argument("--folder", required=True, help="Directory containing CSV log(s)")
    parser.add_argument("--output", default="./controller_analysis_output")
    parser.add_argument("--start-row", type=int, default=None, help="Inclusive CSV row, header=1")
    parser.add_argument("--end-row", type=int, default=None, help="Inclusive CSV row")
    parser.add_argument("--start-time", default=None, help="YYYY-MM-DD HH:MM:SS")
    parser.add_argument("--end-time", default=None)
    parser.add_argument("--mask-rh-zeros", action="store_true", help="Exclude RH=0 from calculations, keep RH_Original")
    parser.add_argument("--chamber-reference", type=float, default=37.0)
    parser.add_argument("--rh-reference", type=float, default=90.0)
    parser.add_argument("--co2-threshold", type=float, default=47000.0, help="CO2 recovery threshold, raw logged units")
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()
    logs, failures = load_folder(args.folder, recursive=args.recursive)
    print(f"Found {len(logs)} compatible log(s); skipped {len(failures)}")
    for file_name, reason in failures.items():
        print(f"Skipped {file_name}: {reason}")
    for fname, log in logs.items():
        try:
            report = build_report(log, start_row=args.start_row, end_row=args.end_row,
                                  start_time=args.start_time, end_time=args.end_time,
                                  mask_rh_zeros=args.mask_rh_zeros,
                                  chamber_target=args.chamber_reference, rh_target=args.rh_reference,
                                  co2_threshold=args.co2_threshold)
            output = Path(args.output) / re.sub(r"[^A-Za-z0-9_-]", "_", fname)
            figs = {"profiles": plot_environment(report["data"], resample="10s", targets=report["targets"]),
                    "controllers": plot_controller_signals(report["data"], resample="10s"),
                    "comparison_bars": plot_statistics(report["stats"]),
                    "control_distribution": plot_control_summary(report["controllers"])}
            export_report(report, output, charts=figs)
            print(f"Processed {fname}: {len(report['data'])} rows -> {output}")
        except Exception as exc:
            print(f"Failed to process {fname}: {exc}")
            raise


if __name__ == "__main__":
    _main()