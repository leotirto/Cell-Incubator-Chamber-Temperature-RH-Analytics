"""Temperature/RH engineering analytics: loss-aware cleaning, time metrics and charts.

Designed for multi-sensor CSV exports and an interactive Colab notebook.
Requires pandas, numpy, plotly. This module itself does not use Google Drive.
"""
from __future__ import annotations

import csv
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DATE, TEMP, RH = 'Date', 'Temperature_C', 'RH_pct'
METRICS = {TEMP: 'Temperature (°C)', RH: 'Relative humidity (%)'}
STAGES = ('Entire selection', 'After temperature first reach',
          'After RH first reach', 'After both first reach')


def _norm(value):
    return re.sub(r'[^a-z0-9]', '', str(value).lower())


def _resolve_columns(columns, overrides=None):
    """Resolve without silently assuming that arbitrary first three columns are sensors."""
    overrides = overrides or {}
    names = list(columns)
    def choose(label, exact, loose):
        if label in overrides:
            if overrides[label] not in names:
                raise ValueError(f'{label} override {overrides[label]!r} missing from columns {names}')
            return overrides[label]
        perfect = [c for c in names if _norm(c) in exact]
        candidates = perfect if perfect else [c for c in names if loose(_norm(c))]
        if len(candidates) != 1:
            raise ValueError(f'{label}: expected one matching column, found {candidates}; '
                             f'CSV columns: {names}. Set COLUMN_OVERRIDES for ambiguous exports.')
        return candidates[0]
    date = choose('Date', {'date', 'datetime', 'timestamp', 'time', 'recordedat', 'datetimeutc'},
                  lambda s: 'timestamp' in s or s.startswith('datetime') or s.endswith('date'))
    temp = choose('Temperature', {'temperature', 'temperaturec', 'temperaturecelsius', 'temp',
                                  'tempc', 'temperaturef', 'tempf', 'temperaturefahrenheit'},
                  lambda s: (s.startswith('temperature') or s.startswith('tempc') or s == 'celsius')
                  and not any(x in s for x in ('setpoint', 'target', 'threshold', 'dew')))
    rh = choose('RH', {'rh', 'rhpct', 'rhpercent', 'humidity', 'humiditypct',
                       'relativehumidity', 'relativehumiditypct', 'relativehumiditypercent'},
                lambda s: (s.startswith('rh') or 'relativehumidity' in s or s.startswith('humidity'))
                and not any(x in s for x in ('absolute', 'target', 'threshold', 'setpoint')))
    if len({date, temp, rh}) < 3:
        raise ValueError('Date, Temperature and RH must be different CSV columns')
    return date, temp, rh


def _numbers(series):
    """Read 36.7°C, 85%, 36,7, 1,234.5, or 1.234,5 without unit strings."""
    s = series.astype('string').str.strip().str.replace(r'[^0-9,.+\-]', '', regex=True)
    # Decimal separator is the rightmost separator if both occur.
    mixed = s.str.contains(',', na=False) & s.str.contains(r'\.', na=False)
    eu = mixed & (s.str.rfind(',') > s.str.rfind('.'))
    us = mixed & ~eu
    s.loc[eu] = s.loc[eu].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    s.loc[us] = s.loc[us].str.replace(',', '', regex=False)
    comma_only = s.str.contains(',', na=False) & ~mixed
    s.loc[comma_only] = s.loc[comma_only].str.replace(',', '.', regex=False)
    return pd.to_numeric(s, errors='coerce').astype(float)


def _detect_separator(path, encoding):
    with open(path, 'r', encoding=encoding, errors='replace', newline='') as f:
        sample = f.read(8192)
    try:
        sep = csv.Sniffer().sniff(sample, delimiters=',;\t|').delimiter
    except csv.Error:
        sep = ','
    return sep


def load_one_csv(path, *, day_first=False, rh_format='auto', overrides=None,
                 temp_min=-40.0, temp_max=100.0):
    """Return (readings, QA). All missing/invalid readings stay NaN, never fabricated."""
    path = Path(path)
    raw = None
    last_error = None
    for encoding in ('utf-8-sig', 'cp1252'):
        try:
            sep = _detect_separator(path, encoding)
            raw = pd.read_csv(path, sep=sep, encoding=encoding, dtype=str,
                              keep_default_na=False, on_bad_lines='error')
            break
        except (UnicodeError, pd.errors.ParserError) as exc:
            last_error = exc
    if raw is None:
        raise ValueError(f'Cannot parse {path.name}: {last_error}')
    raw.columns = raw.columns.astype(str).str.strip()
    if raw.columns.duplicated().any():
        raise ValueError(f'Duplicate CSV columns in {path.name}: {list(raw.columns)}')
    date_col, temp_col, rh_col = _resolve_columns(raw.columns, overrides)
    raw['SourceRow'] = np.arange(2, 2 + len(raw))
    # format='mixed' supports multiple date-time representations in a file.
    dt = pd.to_datetime(raw[date_col].replace('', pd.NA), errors='coerce',
                        dayfirst=day_first, format='mixed')
    if isinstance(dt.dtype, pd.DatetimeTZDtype):
        dt = dt.dt.tz_localize(None)  # Preserve displayed clock time; warn via QA.
        time_note = 'TZ-aware timestamps: timezone removed without clock conversion'
    else:
        time_note = 'Naive clock time; check DAY_FIRST if calendar dates are ambiguous'
    t = _numbers(raw[temp_col])
    f = _norm(temp_col)
    if ('fahrenheit' in f or f.endswith('tempf') or f == 'temperaturef'
            or '°f' in temp_col.lower() or '(f)' in temp_col.lower()):
        t = (t - 32.0) * 5.0 / 9.0
        temp_units = 'Fahrenheit converted to °C'
    else:
        temp_units = 'Celsius assumed from CSV column'
    h = _numbers(raw[rh_col])
    format_chosen = rh_format
    if rh_format == 'auto':
        valid = h.dropna()
        # Explicitly log scaling: very low 0...1% RH is otherwise ambiguous.
        if len(valid) and valid.between(0, 1).all() and valid.median() >= 0.05:
            format_chosen = 'fraction'
        else:
            format_chosen = 'percent'
    if format_chosen not in ('fraction', 'percent'):
        raise ValueError('RH format must be auto, fraction, or percent')
    if format_chosen == 'fraction':
        h = h * 100.0
    bad_temp_range = t.notna() & ~t.between(temp_min, temp_max)
    bad_rh_range = h.notna() & ~h.between(0, 100)
    quality = {
        'Sensor': path.stem,
        'CSV rows': len(raw),
        'Invalid timestamps': int(dt.isna().sum()),
        'Unparsed temperature': int(t.isna().sum()),
        'Unparsed RH': int(h.isna().sum()),
        'Implausible temperature': int(bad_temp_range.sum()),
        'RH out of 0–100%': int(bad_rh_range.sum()),
        'Duplicate timestamp rows': int(dt.dropna().duplicated().sum()),
        'Time out-of-order rows': int((dt.diff().dt.total_seconds() < 0).sum()),
        'RH interpretation': format_chosen,
        'Temperature units': temp_units,
        'Time interpretation': time_note,
    }
    result = pd.DataFrame({DATE: dt, TEMP: t.mask(bad_temp_range), RH: h.mask(bad_rh_range),
                           'SourceRow': raw['SourceRow']})
    result = result.dropna(subset=[DATE]).sort_values([DATE, 'SourceRow']).reset_index(drop=True)
    return result, quality


def load_folder(folder, *, day_first=False, rh_format='auto', overrides=None):
    folder = Path(folder).expanduser()
    if not folder.exists():
        raise FileNotFoundError(f'Folder does not exist: {folder}')
    files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == '.csv')
    if not files:
        raise FileNotFoundError(f'No CSV files in {folder}')
    data, reports, failures = {}, [], []
    for file in files:
        try:
            frame, info = load_one_csv(file, day_first=day_first,
                                        rh_format=rh_format, overrides=overrides)
            if not frame.empty:
                data[file.stem] = frame
                reports.append(info)
            else:
                failures.append(f'{file.name}: no valid timestamps')
        except Exception as exc:
            failures.append(f'{file.name}: {exc}')
    if not data:
        raise ValueError('No usable sensor files. ' + ' | '.join(failures))
    return data, pd.DataFrame(reports), failures


def select_window(df, *, row_start=None, row_end=None, date_start=None, date_end=None,
                  drop_tail=0):
    """Row range is inclusive, referencing original CSV data rows (header is row 1)."""
    d = df
    if row_start is not None or row_end is not None:
        if row_start is not None and row_start < 2:
            raise ValueError('First data record is CSV row 2 (row 1 is header)')
        if row_end is not None and row_start is not None and row_end < row_start:
            raise ValueError('End row must be >= start row')
        if row_start is not None:
            d = d[d.SourceRow >= row_start]
        if row_end is not None:
            d = d[d.SourceRow <= row_end]
    if date_start is not None:
        d = d[d[DATE] >= pd.Timestamp(date_start)]
    if date_end is not None:
        d = d[d[DATE] <= pd.Timestamp(date_end)]
    d = d.copy().sort_values([DATE, 'SourceRow'])
    if drop_tail:
        if drop_tail < 0:
            raise ValueError('drop_tail must be >= 0')
        d = d.iloc[:-drop_tail].copy() if len(d) > drop_tail else d.iloc[:0].copy()
    return d


def prepare_series(d):
    """Collapse duplicate timestamps only AFTER row/time filtering."""
    if d.empty:
        return pd.DataFrame(columns=[DATE, TEMP, RH, 'ReadingsAtTime', 'FirstCSVRow', 'LastCSVRow'])
    return (d.groupby(DATE, as_index=False, sort=True)
            .agg(**{TEMP: (TEMP, 'mean'), RH: (RH, 'mean'),
                    'ReadingsAtTime': ('SourceRow', 'size'),
                    'FirstCSVRow': ('SourceRow', 'min'),
                    'LastCSVRow': ('SourceRow', 'max')}))


def with_derived(d, *, max_gap_min=5, rolling_min=5):
    """Add physically meaningful derived features, NOT synthetic measurements."""
    d = d.copy()
    if d.empty:
        return d
    temp, rh = d[TEMP], d[RH]
    # Magnus dew point valid approximately for common environmental conditions.
    valid_psych = temp.between(-40, 60) & rh.gt(0) & rh.le(100)
    gamma = np.log(rh.where(valid_psych) / 100) + 17.62 * temp.where(valid_psych) / (243.12 + temp.where(valid_psych))
    d['DewPoint_C'] = 243.12 * gamma / (17.62 - gamma)
    vapor_pressure = (rh.where(valid_psych) / 100) * 6.112 * np.exp(
        17.67 * temp.where(valid_psych) / (temp.where(valid_psych) + 243.5))
    d['AbsoluteHumidity_gm3'] = 216.7 * vapor_pressure / (temp.where(valid_psych) + 273.15)
    delta = d[DATE].diff().dt.total_seconds() / 60.0
    allowed = delta.gt(0) & delta.le(max_gap_min)
    d['dT_C_per_min'] = (temp.diff() / delta).where(allowed)
    d['dRH_pct_per_min'] = (rh.diff() / delta).where(allowed)
    if rolling_min and rolling_min > 0:
        indexed = d.set_index(DATE)
        d['TempRolling_C'] = indexed[TEMP].rolling(f'{rolling_min}min', min_periods=2).mean().to_numpy()
        d['RHRolling_pct'] = indexed[RH].rolling(f'{rolling_min}min', min_periods=2).mean().to_numpy()
    else:
        d['TempRolling_C'] = temp
        d['RHRolling_pct'] = rh
    return d


def first_reach(d, *, temp_threshold, rh_threshold, stage='both'):
    if d.empty:
        return pd.NaT
    m_t = d[TEMP].ge(temp_threshold)
    m_h = d[RH].ge(rh_threshold)
    matched = {'temp': m_t, 'rh': m_h, 'both': m_t & m_h}[stage]
    if not matched.any():
        return pd.NaT
    return d.loc[matched, DATE].iloc[0]


def first_sustained_reach(d, *, temp_threshold, rh_threshold, stage,
                           hold_minutes, max_gap_min):
    """Find first observed qualifying run spanning hold duration, without large gaps."""
    if d.empty:
        return pd.NaT
    if hold_minutes <= 0:
        return first_reach(d, temp_threshold=temp_threshold,
                           rh_threshold=rh_threshold, stage=stage)
    masks = {'temp': d[TEMP].ge(temp_threshold), 'rh': d[RH].ge(rh_threshold)}
    good = (masks['temp'] & masks['rh']) if stage == 'both' else masks[stage]
    previous = d[DATE].diff().dt.total_seconds().div(60)
    # New segment when condition fails or adjacent samples are too far apart.
    segment = ((~good) | (~good.shift(fill_value=False)) |
               previous.gt(max_gap_min)).cumsum()
    runs = d.loc[good].groupby(segment.loc[good])[DATE].agg(['first', 'last'])
    qualified = runs.loc[(runs['last'] - runs['first']).dt.total_seconds() >= 60 * hold_minutes]
    return qualified.iloc[0]['first'] if not qualified.empty else pd.NaT


def _minutes_from(start, reached):
    return float((reached - start).total_seconds() / 60) if pd.notna(reached) else np.nan


def time_weighted_fraction(d, mask, *, max_gap_min):
    """Observed eligible interval %, ignoring outages; last point adds no time."""
    if len(d) < 2:
        return np.nan, 0.0
    intervals = (d[DATE].shift(-1) - d[DATE]).dt.total_seconds().div(60)
    valid = intervals.gt(0) & intervals.le(max_gap_min) & mask.notna()
    eligible = intervals.where(valid, 0.0)
    covered = eligible.sum()
    if covered <= 0:
        return np.nan, 0.0
    return float(100.0 * eligible.where(mask.fillna(False), 0.0).sum() / covered), float(covered)


def stage_slice(d, stage, *, temp_threshold, rh_threshold, exempt=False):
    if stage not in STAGES:
        raise ValueError(f'Unknown stage: {stage}')
    if d.empty or stage == STAGES[0] or exempt:
        return d
    key = {STAGES[1]: 'temp', STAGES[2]: 'rh', STAGES[3]: 'both'}[stage]
    t = first_reach(d, temp_threshold=temp_threshold, rh_threshold=rh_threshold,
                    stage=key)
    return d.loc[d[DATE] >= t].copy() if pd.notna(t) else d.iloc[:0].copy()


def summarize(sensors, *, temp_threshold=36.5, rh_threshold=85.0,
              temp_setpoint=37.0, rh_setpoint=90.0, temp_tol=0.5, rh_tol=5.0,
              hold_minutes=10.0, max_gap_min=5.0, except_sensor_names=('govee-05',)):
    """Build overview + stage/variable statistics with numeric (never string) times."""
    overview, stages = [], []
    for name, d in sensors.items():
        if d.empty:
            continue
        start = d[DATE].iloc[0]
        end = d[DATE].iloc[-1]
        exempt = any(s.lower() in name.lower() for s in except_sensor_names)
        times = {s: first_reach(d, temp_threshold=temp_threshold,
                                rh_threshold=rh_threshold, stage=s)
                 for s in ('temp', 'rh', 'both')}
        sustained = {s: first_sustained_reach(d, temp_threshold=temp_threshold,
                                              rh_threshold=rh_threshold, stage=s,
                                              hold_minutes=hold_minutes,
                                              max_gap_min=max_gap_min)
                     for s in ('temp', 'rh', 'both')}
        both_valid = d[TEMP].notna() & d[RH].notna()
        above_temp = d[TEMP].ge(temp_threshold) & d[TEMP].notna()
        above_rh = d[RH].ge(rh_threshold) & d[RH].notna()
        above_both = above_temp & above_rh
        in_temp_band = (d[TEMP].sub(temp_setpoint).abs().le(temp_tol) & d[TEMP].notna())
        in_rh_band = (d[RH].sub(rh_setpoint).abs().le(rh_tol) & d[RH].notna())
        # For denominators, only intervals with an available measurement qualify.
        def pct(mask, observation):
            weights = (d[DATE].shift(-1) - d[DATE]).dt.total_seconds().div(60)
            active = weights.gt(0) & weights.le(max_gap_min) & observation
            covered = weights.where(active, 0).sum()
            passed = weights.where(active & mask, 0).sum()
            return (float(100 * passed / covered) if covered > 0 else np.nan,
                    float(covered))
        pt, tcov = pct(above_temp, d[TEMP].notna())
        ph, hcov = pct(above_rh, d[RH].notna())
        pb, bcov = pct(above_both, both_valid)
        pbt, _ = pct(in_temp_band, d[TEMP].notna())
        pbh, _ = pct(in_rh_band, d[RH].notna())
        pband, _ = pct(in_temp_band & in_rh_band, both_valid)
        rec = {
            'Sensor': name, 'Records': len(d), 'Start': start, 'End': end,
            'Elapsed hours': (end - start).total_seconds() / 3600,
            'Temp min °C': d[TEMP].min(), 'Temp mean °C': d[TEMP].mean(),
            'Temp max °C': d[TEMP].max(), 'Temp SD °C': d[TEMP].std(),
            'RH min %': d[RH].min(), 'RH mean %': d[RH].mean(),
            'RH max %': d[RH].max(), 'RH SD %': d[RH].std(),
            'Time to temp min': np.nan if exempt else _minutes_from(start, times['temp']),
            'Time to RH min': np.nan if exempt else _minutes_from(start, times['rh']),
            'Time to both min': np.nan if exempt else _minutes_from(start, times['both']),
            'Sustained temp min': np.nan if exempt else _minutes_from(start, sustained['temp']),
            'Sustained RH min': np.nan if exempt else _minutes_from(start, sustained['rh']),
            'Sustained both min': np.nan if exempt else _minutes_from(start, sustained['both']),
            'Time ≥ temp threshold %': pt, 'Time ≥ RH threshold %': ph,
            'Time ≥ both thresholds %': pb,
            'Time in temp band %': pbt, 'Time in RH band %': pbh,
            'Time in both bands %': pband,
            'Temp observed minutes': tcov, 'RH observed minutes': hcov,
            'Both observed minutes': bcov,
            'Temp threshold status': ('EXEMPT' if exempt else
                                       ('Reached' if pd.notna(times['temp']) else 'Never reached')),
            'RH threshold status': ('EXEMPT' if exempt else
                                     ('Reached' if pd.notna(times['rh']) else 'Never reached')),
            'Both threshold status': ('EXEMPT' if exempt else
                                       ('Reached' if pd.notna(times['both']) else 'Never reached')),
            'Missing temp': int(d[TEMP].isna().sum()),
            'Missing RH': int(d[RH].isna().sum()),
            'Collapsed duplicates': int((d['ReadingsAtTime'] - 1).clip(lower=0).sum()),
            'Gaps longer than limit': int(d[DATE].diff().dt.total_seconds().div(60).gt(max_gap_min).sum()),
        }
        overview.append(rec)
        for stage in STAGES:
            sliced = stage_slice(d, stage, temp_threshold=temp_threshold,
                                 rh_threshold=rh_threshold, exempt=exempt)
            for var, label in METRICS.items():
                v = sliced[var].dropna()
                stages.append({
                    'Sensor': name, 'Stage': stage, 'Variable': label, 'Units': '°C' if var == TEMP else '%',
                    'N': len(v), 'Min': v.min(), 'Mean': v.mean(), 'Median': v.median(),
                    'P05': v.quantile(.05), 'P95': v.quantile(.95),
                    'Max': v.max(), 'SD': v.std(), 'Variance': v.var(),
                    'Latest': v.iloc[-1] if not v.empty else np.nan,
                })
    return pd.DataFrame(overview), pd.DataFrame(stages)


def spread_series(sensors, resolution='1min'):
    """Spatial spread across aligned sensor recordings. No interpolation performed."""
    charts = []
    for name, d in sensors.items():
        if d.empty:
            continue
        aligned = d.set_index(DATE)[[TEMP, RH]].resample(resolution).mean()
        aligned.columns = pd.MultiIndex.from_product([[name], aligned.columns])
        charts.append(aligned)
    if len(charts) < 2:
        return pd.DataFrame(columns=[DATE, 'Sensors_Temp', 'TempSpread_C',
                                     'Sensors_RH', 'RHSpread_pct', 'TempSD_C', 'RHSD_pct'])
    wide = pd.concat(charts, axis=1)
    out = pd.DataFrame({DATE: wide.index})
    out.index = wide.index
    for var, out_prefix in ((TEMP, 'Temp'), (RH, 'RH')):
        vals = wide.xs(var, axis=1, level=1)
        count = vals.count(axis=1)
        out[f'Sensors_{out_prefix}'] = count
        out['TempSpread_C' if var == TEMP else 'RHSpread_pct'] = (vals.max(axis=1) - vals.min(axis=1)).where(count >= 2)
        out['TempSD_C' if var == TEMP else 'RHSD_pct'] = vals.std(axis=1).where(count >= 2)
    return out.reset_index(drop=True)


def plot_series(d, resolution='Raw', rolling_min=0, interpolation=False):
    """Plot-only resampling / interpolation; original analytics are always raw."""
    if d.empty:
        return d
    if resolution == 'Raw':
        out = d[[DATE, TEMP, RH, 'DewPoint_C', 'AbsoluteHumidity_gm3',
                 'dT_C_per_min', 'dRH_pct_per_min']].copy()
    else:
        vars_ = [TEMP, RH, 'DewPoint_C', 'AbsoluteHumidity_gm3',
                 'dT_C_per_min', 'dRH_pct_per_min']
        out = d.set_index(DATE)[vars_].resample(resolution).mean()
        if interpolation:
            out = out.interpolate(method='time', limit=2, limit_area='inside')
        out = out.reset_index()
    if rolling_min:
        for x in (TEMP, RH):
            out[x] = (out.set_index(DATE)[x].rolling(f'{rolling_min}min', min_periods=1)
                      .mean().to_numpy())
    return out


def _line_kwargs(name):
    controller = any(k in name.lower() for k in ('main', 'controller'))
    return dict(width=2.7 if controller else 1.55, dash='dash' if controller else 'solid')


def _x_values(df, origin, mode):
    if mode == 'Absolute timestamp':
        return df[DATE], 'Date / time'
    if mode == 'Elapsed from each sensor':
        return (df[DATE] - df[DATE].min()).dt.total_seconds() / 3600, 'Hours since each sensor started'
    return (df[DATE] - origin).dt.total_seconds() / 3600, 'Hours since earliest selected timestamp'


def _break_long_gaps(p, max_gap_min):
    """Insert missing plot values at long outages instead of drawing false bridges."""
    if len(p) < 2:
        return p
    dt = p[DATE].diff().dt.total_seconds().div(60)
    indices = np.flatnonzero(dt.to_numpy() > max_gap_min)
    if not len(indices):
        return p
    breaks = p.iloc[indices].copy()
    breaks[DATE] = p.iloc[indices - 1][DATE].to_numpy() + (
        p.iloc[indices][DATE].to_numpy() - p.iloc[indices - 1][DATE].to_numpy()) / 2
    for col in breaks.columns:
        if col != DATE:
            breaks[col] = np.nan
    return pd.concat([p, breaks], ignore_index=True).sort_values(DATE)


def build_profile(sensors, *, temp_threshold=36.5, rh_threshold=85.0,
                  resolution='Raw', smooth_minutes=0, interpolate=False,
                  x_mode='Absolute timestamp', show_points=False, max_gap_min=5):
    if not sensors:
        return go.Figure()
    origin = min(d[DATE].min() for d in sensors.values() if not d.empty)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.12,
                        subplot_titles=('Temperature profile', 'Relative humidity profile'))
    for name, d in sensors.items():
        if d.empty:
            continue
        p = plot_series(d, resolution, smooth_minutes, interpolate)
        p = _break_long_gaps(p, max_gap_min) if resolution == 'Raw' else p
        xx, xlabel = _x_values(p, origin, x_mode)
        for row, var, unit in ((1, TEMP, '°C'), (2, RH, '%')):
            fig.add_trace(go.Scatter(x=xx, y=p[var], mode='lines+markers' if show_points else 'lines',
                                     name=name, legendgroup=name, showlegend=row == 1,
                                     line=_line_kwargs(name),
                                     marker={'size': 4}, connectgaps=False,
                                     hovertemplate=f'<b>{name}</b><br>x=%{{x}}<br>%{{y:.2f}} {unit}<extra></extra>'),
                          row=row, col=1)
    fig.add_hline(y=temp_threshold, row=1, col=1, line_dash='dot', line_color='#bb4545',
                  annotation_text=f'T ≥ {temp_threshold:g} °C')
    fig.add_hline(y=rh_threshold, row=2, col=1, line_dash='dot', line_color='#377cbc',
                  annotation_text=f'RH ≥ {rh_threshold:g} %')
    fig.update_yaxes(title_text='Temperature (°C)', row=1, col=1)
    fig.update_yaxes(title_text='RH (%)', row=2, col=1)
    fig.update_xaxes(title_text=xlabel, row=2, col=1)
    fig.update_layout(height=760, template='plotly_white', hovermode='x unified',
                      legend={'orientation': 'h', 'y': 1.1}, margin={'t': 100},
                      title='Temperature and RH over the selected window')
    return fig


def build_bars(stage_stats, *, stage='Entire selection', variable='Temperature (°C)'):
    s = stage_stats.loc[(stage_stats.Stage == stage) & (stage_stats.Variable == variable)]
    fig = go.Figure()
    for metric, color in [('Min', '#5485bf'), ('Mean', '#41a58e'), ('Max', '#d88e4e')]:
        fig.add_bar(name=metric, x=s['Sensor'], y=s[metric], marker_color=color,
                    text=s[metric].map(lambda x: f'{x:.2f}' if pd.notna(x) else '—'),
                    textposition='outside', hovertemplate='%{x}<br>%{y:.2f}<extra></extra>')
    unit = '°C' if variable.startswith('Temperature') else '%'
    fig.update_layout(barmode='group', template='plotly_white', height=470,
                      title=f'{variable} · {stage}', xaxis_title='Sensor',
                      yaxis_title=f'{variable} ({unit})', margin={'t': 80, 'b': 60},
                      legend={'orientation': 'h', 'y': 1.1})
    return fig


def build_timing_bars(overview):
    fig = go.Figure()
    for col, label, color in [('Time to temp min', 'Temp', '#d07955'),
                              ('Time to RH min', 'RH', '#508ac9'),
                              ('Time to both min', 'Both simultaneously', '#4b9f80')]:
        fig.add_bar(name=label, x=overview['Sensor'], y=overview[col], marker_color=color)
    fig.update_layout(barmode='group', template='plotly_white', height=430,
                      title='Time to first threshold crossing (minutes)',
                      yaxis_title='Minutes after start of selected window',
                      legend={'orientation': 'h', 'y': 1.12})
    return fig


def build_spread_figure(spread):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=('Temperature spatial range', 'RH spatial range'),
                        vertical_spacing=.14)
    if spread.empty:
        fig.add_annotation(text='Select at least 2 overlapping sensors', x=.5, y=.5,
                           xref='paper', yref='paper', showarrow=False)
    else:
        for row, var, color in ((1, 'TempSpread_C', '#bd6f44'), (2, 'RHSpread_pct', '#327dab')):
            fig.add_trace(go.Scatter(x=spread[DATE], y=spread[var], mode='lines',
                                     line=dict(color=color, width=2), name=var,
                                     connectgaps=False), row=row, col=1)
    fig.update_yaxes(title_text='Max−min (°C)', row=1, col=1)
    fig.update_yaxes(title_text='Max−min (%RH)', row=2, col=1)
    fig.update_xaxes(title_text='Date / time', row=2, col=1)
    fig.update_layout(height=580, template='plotly_white', showlegend=False,
                      title='Cross-sensor environmental uniformity (1-minute bins)')
    return fig


def build_derived_figure(sensors, *, derived='DewPoint_C', resolution='Raw', max_gap_min=5):
    labels = {'DewPoint_C': 'Dew point (°C)',
              'AbsoluteHumidity_gm3': 'Absolute humidity (g/m³)',
              'dT_C_per_min': 'Temperature change rate (°C/min)',
              'dRH_pct_per_min': 'RH change rate (%RH/min)'}
    if derived not in labels:
        raise ValueError(f'Unknown derived metric {derived}')
    fig = go.Figure()
    for name, d in sensors.items():
        if d.empty:
            continue
        p = plot_series(d, resolution=resolution)
        p = _break_long_gaps(p, max_gap_min) if resolution == 'Raw' else p
        fig.add_scatter(x=p[DATE], y=p[derived], name=name, mode='lines',
                        line=_line_kwargs(name), connectgaps=False)
    fig.update_layout(template='plotly_white', height=440,
                      title=labels[derived] + ' over time', yaxis_title=labels[derived],
                      xaxis_title='Date / time', hovermode='x unified',
                      legend={'orientation': 'h', 'y': 1.14})
    return fig


def run_analysis(data, selected_names, *, row_start=None, row_end=None,
                 date_start=None, date_end=None, drop_tail=0,
                 temp_threshold=36.5, rh_threshold=85.0,
                 temp_setpoint=37, rh_setpoint=90, temp_tol=.5, rh_tol=5,
                 hold_minutes=10, max_gap_min=5, except_sensor_names=('govee-05',)):
    selected = {}
    for name in selected_names:
        if name not in data:
            raise KeyError(f'Sensor {name!r} not loaded')
        clipped = select_window(data[name], row_start=row_start, row_end=row_end,
                                date_start=date_start, date_end=date_end, drop_tail=drop_tail)
        prepared = prepare_series(clipped)
        if not prepared.empty:
            selected[name] = with_derived(prepared, max_gap_min=max_gap_min)
    if not selected:
        raise ValueError('No readings in the selected row/time window. Adjust your filters.')
    overview, stages = summarize(selected, temp_threshold=temp_threshold,
                                 rh_threshold=rh_threshold, temp_setpoint=temp_setpoint,
                                 rh_setpoint=rh_setpoint, temp_tol=temp_tol, rh_tol=rh_tol,
                                 hold_minutes=hold_minutes, max_gap_min=max_gap_min,
                                 except_sensor_names=except_sensor_names)
    spread = spread_series(selected)
    return selected, overview, stages, spread


def _main():
    """Optional headless batch mode; the .ipynb is the interactive user interface."""
    import argparse
    import json
    parser = argparse.ArgumentParser(description='Clean and analyze multi-sensor chamber CSV data')
    parser.add_argument('--folder', required=True, help='Folder containing source CSV exports')
    parser.add_argument('--output', default=None, help='Output folder (default: folder/analysis_output)')
    parser.add_argument('--start-row', type=int, default=None, help='Inclusive source CSV row; header=1')
    parser.add_argument('--end-row', type=int, default=None, help='Inclusive source CSV row')
    parser.add_argument('--start-time', type=str, default=None, help='ISO local timestamp')
    parser.add_argument('--end-time', type=str, default=None, help='ISO local timestamp')
    parser.add_argument('--temp-threshold', type=float, default=36.5)
    parser.add_argument('--rh-threshold', type=float, default=85.0)
    parser.add_argument('--day-first', action='store_true')
    parser.add_argument('--rh-format', choices=['auto', 'fraction', 'percent'], default='auto')
    parser.add_argument('--no-govee05-exception', action='store_true')
    args = parser.parse_args()
    data, qa, failures = load_folder(args.folder, day_first=args.day_first,
                                      rh_format=args.rh_format)
    selected, overview, stages, spread = run_analysis(
        data, list(data), row_start=args.start_row, row_end=args.end_row,
        date_start=args.start_time, date_end=args.end_time,
        temp_threshold=args.temp_threshold, rh_threshold=args.rh_threshold,
        except_sensor_names=() if args.no_govee05_exception else ('govee-05',))
    output = Path(args.output) if args.output else Path(args.folder)/'analysis_output'
    output.mkdir(parents=True, exist_ok=True)
    for name, d in selected.items():
        d.to_csv(output/f'{re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)}_processed.csv', index=False)
    overview.to_csv(output/'sensor_overview.csv', index=False)
    stages.to_csv(output/'stage_statistics.csv', index=False)
    spread.to_csv(output/'cross_sensor_uniformity.csv', index=False)
    qa.to_csv(output/'full_file_quality.csv', index=False)
    figures = {
        'profiles': build_profile(selected, temp_threshold=args.temp_threshold,
                                  rh_threshold=args.rh_threshold),
        'temperature_bars': build_bars(stages),
        'rh_bars': build_bars(stages, variable='Relative humidity (%)'),
        'timing_bars': build_timing_bars(overview),
        'sensor_uniformity': build_spread_figure(spread),
        'dew_point': build_derived_figure(selected),
    }
    for filename, figure in figures.items():
        figure.write_html(str(output/f'{filename}.html'), include_plotlyjs=True)
    (output/'run_info.json').write_text(json.dumps({
        'folder': str(args.folder), 'start_row': args.start_row, 'end_row': args.end_row,
        'files_skipped': failures,
        'note': 'Analysis from observed valid measurements; visualization never changes summaries.'
    }, indent=2), encoding='utf-8')
    print(f'Processed {len(selected)} sensor(s); results: {output.resolve()}')
    if failures:
        print('Failed/skipped files:', '\n'.join(failures))


if __name__ == '__main__':
    _main()