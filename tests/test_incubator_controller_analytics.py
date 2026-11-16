"""Synthetic, repository-safe tests. No real chamber measurements are included."""
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from incubator_controller_analytics import (
    TIME, augment, build_report, control_statistics, descriptive_statistics,
    discover_logs, export_report, find_excursions, first_sustained_threshold,
    lag_correlation, load_folder, plot_environment, plot_control_summary,
    plot_controller_signals, plot_correlation, plot_excursions, plot_lag_correlation,
    plot_pair, plot_statistics, read_controller_log, select_window,
)


def fake_log(path: Path):
    times = pd.date_range("2026-11-14 12:11:40", periods=130, freq="1200ms")
    t = np.arange(len(times))
    d = pd.DataFrame({
        "header-time": times.astype(str), "T Steam": 118 + np.sin(t / 10),
        "Ctrl Steam": np.sin(t / 7),
        "T Chamber": 36.5 + .8 * (1 - np.exp(-t / 13)),
        "Ctrl Chamber": np.sin(t / 10) / 10,
        "T Bot": 32.0 + t / 1000, "Ctrl Bot": np.sin(t / 7) / 10,
        "T Door": 33.0, "Ctrl Door": 1.0,
        "T Port": 35.4, "Ctrl Port": .2,
        "RH": np.where((t >= 40) & (t < 52), 0, 90 + np.sin(t / 8) / 10),
        "Ctrl RH": .01 * np.sin(t / 6), "CO2": 50000 + 200 * np.sin(t / 12),
        "Ctrl CO2": .003, "STC31": 5.7, "empty": np.nan
    })
    d.columns = [str(times[0])] + list(d.columns[1:-1]) + [""]
    d.to_csv(path, index=False)
    return d


def test_realistic_structure_and_qa():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "synthetic_controller.csv"
        fake_log(p)
        log = read_controller_log(p)
        assert len(log.frame) == 130
        assert log.frame.SourceRow.min() == 2
        assert log.quality["RH exact zeros"] == 12
        assert log.quality["Invalid timestamp rows"] == 0
        assert log.quality["Empty CSV columns dropped"]
        assert log.quality["Log header timestamp"].startswith("2026-11-14")
        assert len(discover_logs(td)) == 1
        logs, failed = load_folder(td)
        assert len(logs) == 1 and not failed
        window = select_window(log, start_row=60, end_row=100)
        assert len(window) == 41
        assert window.SourceRow.iloc[0] == 60 and window.SourceRow.iloc[-1] == 100
        cut = select_window(log, start_time="2026-11-14 12:12:00", end_time="2026-11-14 12:12:12")
        assert (cut[TIME] >= pd.Timestamp("2026-11-14 12:12:00")).all()
        assert (cut[TIME] <= pd.Timestamp("2026-11-14 12:12:12")).all()


def test_features_stats_and_zero_flag():
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/"x.csv"
        fake_log(p)
        log=read_controller_log(p)
        full=augment(log.frame, mask_rh_zeros=False)
        masked=augment(log.frame, mask_rh_zeros=True)
        assert full["RH"].eq(0).sum() == 12
        assert masked["RH"].eq(0).sum() == 0
        assert masked["RH_Original"].eq(0).sum() == 12
        assert masked.RH.isna().sum() == 12
        assert masked.DewPoint_C.notna().sum() > 100
        assert masked.AbsoluteHumidity_g_m3.notna().sum() > 100
        assert masked["TemperatureZoneRange_C"].notna().all()
        assert full["T Chamber rate/min"].notna().sum()>100
        stat=descriptive_statistics(masked, targets={"RH": (90, 1)})
        assert stat.loc[stat.Channel.eq("RH"), "Time_in_band_pct"].iloc[0] == 100
        ctrl=control_statistics(masked)
        assert len(ctrl)==7
        assert ctrl.loc[ctrl.Controller.eq("Ctrl Door"), "Positive_time_pct"].iloc[0] == 100
        assert ctrl.loc[ctrl.Controller.eq("Ctrl Door"), "Sign_changes"].iloc[0] == 0
        assert np.isfinite(first_sustained_threshold(masked)["Both: hold confirmed (min)"])
        report=build_report(log, mask_rh_zeros=True, rh_tolerance=1)
        assert not report["events"].empty or report["data"].RH.isna().sum() == 12
        assert report["data"].RH.isna().sum() == 12
        assert report["data"].RH_Original.eq(0).sum()==12
        assert report["parameters"]["rh_target"] == 90
        output = export_report(report, Path(td)/"out")
        assert (output/"processed_readings.csv").exists()
        assert (output/"quality_report.json").exists()


def test_excursion_recovery_detection():
    tm=pd.date_range("2026-11-14 12:00", periods=50, freq="1s")
    rh=np.array([90.] * 50)
    rh[10:19] = 0
    frame=pd.DataFrame({TIME:tm, "RH":rh, "SourceRow":np.arange(2,52)})
    events=find_excursions(frame, "RH", 90, 1, min_duration_s=5,
                           recovery_hold_s=10, max_gap_seconds=2)
    assert len(events)==1
    assert events.iloc[0].Recovered
    assert events.iloc[0].Outside_observed_s == 8
    assert events.iloc[0].Recovery_from_start_min == 9/60
    fig=plot_excursions(frame,"RH",90,1,events)
    assert len(fig.data)==2
    frame.loc[23, TIME] += pd.Timedelta(seconds=30)
    # Non-monotonic input should be sorted by caller; positive gap test below.
    original=pd.DataFrame({TIME:tm, "RH": rh})
    original.loc[15:16, TIME] += pd.Timedelta(seconds=15)
    assert isinstance(fig, type(plot_environment(pd.DataFrame({TIME:tm,"RH":rh}))))


def test_graphs_and_lag():
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/"m.csv"
        fake_log(p)
        report=build_report(read_controller_log(p), mask_rh_zeros=True)
        d=report["data"]
        figs=[plot_environment(d,resample="10s"), plot_controller_signals(d,resample="10s"),
              plot_pair(d,"RH",resample="5s"), plot_statistics(report["stats"]),
              plot_statistics(report["stats"],group="RH"),
              plot_control_summary(report["controllers"]),plot_correlation(d)]
        assert all(len(f.data)>0 for f in figs)
        lags=lag_correlation(d,bin_seconds=5,max_lag_seconds=15,min_overlaps=4)
        assert len(lags)==7
        assert len(plot_lag_correlation(lags,measurement="RH",controller="Ctrl RH").data)==1


def test_invalid_csv_and_gaps():
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/"bad.csv"
        pd.DataFrame({"not a time": ["2026-11-14"], "Unrelated": [1]}).to_csv(p,index=False)
        logs, failed=load_folder(td)
        assert not logs and len(failed)==1
        g=Path(td)/"govee_only.csv"
        pd.DataFrame({"Date":["2026-11-14 12:00:00"],"Temperature":[37],"RH":[90]}).to_csv(g,index=False)
        logs, failed=load_folder(td)
        assert not logs and len(failed)==2
        p=Path(td)/"good.csv"
        df=pd.DataFrame({"Timestamp":["2026-11-14 12:00:00", "nonsense",
                    "2026-11-14 12:00:01", "2026-11-14 12:01:01"],
                    "T Chamber":[37,37.2,37.3,38], "RH": [90, 91, 90, 92],
                    "Ctrl RH": [-1,-.1,1, .5]})
        df.to_csv(p,index=False)
        a=read_controller_log(p)
        assert a.quality["Invalid timestamp rows"] == 1
        assert a.quality["Gaps > 5s"] == 1
        d=augment(a.frame,max_gap_seconds=5)
        assert np.isnan(d["RH rate/min"].iloc[-1])
        assert np.isnan(d["T Chamber rate/min"].iloc[-1])
        assert d["LargeGapBefore"].iloc[-1]


if __name__ == "__main__":
    test_realistic_structure_and_qa();test_features_stats_and_zero_flag()
    test_excursion_recovery_detection();test_graphs_and_lag();test_invalid_csv_and_gaps()
    print("ALL SYNTHETIC CONTROLLER LOG TESTS PASSED")