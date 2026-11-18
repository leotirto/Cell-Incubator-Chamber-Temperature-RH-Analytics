"""Synthetic data regression; never commit actual VSE measurement or private labware table."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy import signal
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vse_vibration_core import (
    G0, read_vse, read_labware_reference, select_samples, compute_spectrum,
    spectral_peaks, labware_screen, analyze, spectrogram, time_metrics,
    plot_time_rms, plot_time_shocks, plot_fft, plot_psd, plot_spectrogram,
    plot_mode_ranking, export_report)


def synthetic_export(path:Path, duration_s=30.0,fs=400.0):
    n=int(fs*duration_s)
    t=np.arange(n)/fs
    rng=np.random.default_rng(155)
    x=(.2/G0)*np.sin(2*np.pi*40*t)+.001*rng.standard_normal(n)
    y=.007*np.sin(2*np.pi*80*t)+.001*rng.standard_normal(n)
    z=1+.012*np.sin(2*np.pi*40*t)+.001*rng.standard_normal(n)
    start=pd.Timestamp('2026-11-17 19:06:41.126249')
    timestamps=start+pd.to_timedelta(t,unit='s')
    body='44444_simulated\n\nTime (Date hh:mm:ss.ms)\tX: g\tY: g\tZ: g\t\n'
    body+='\n'.join(f'{ts.strftime("%Y/%m/%d %H:%M:%S.%f")}\t{ax:.6f}\t{ay:.6f}\t{az:.6f}\t' for ts,ax,ay,az in zip(timestamps,x,y,z))
    path.write_text(body,encoding='utf-8')
    return start


def reference_file(path:Path):
    path.write_text('''LABWARE: 8 Well Rectangular | Stabilization Window: 4.0 hrs
Harmonic | Freq (Hz) | Spacing (mm) | F_limit (pN) | Allowable a_threshold (m/s2)
1 | 40.0 | 5.0 | 0.1 | 0.02
2 | 80.0 | 2.5 | 0.05 | 0.01
LABWARE: 6 Well Microplate Circular | Stabilization Window: 4.0 hrs
Harmonic | Freq (Hz) | Spacing (mm) | F_limit (pN) | Allowable a_threshold (m/s2)
1 | 35.0 | 7.0 | 0.11 | 0.04
''',encoding='utf-8')


def test_parse_and_timestamp_filters(tmp_path):
    p=tmp_path/'vse_export.xls';start=synthetic_export(p)
    log=read_vse(p)
    assert len(log.frame)==12000
    assert np.isclose(log.quality['fs_est_Hz'],400,rtol=.001)
    assert log.quality['sampling_gaps_gt_1_5_period']==0
    assert log.frame.Time.iloc[0]==start
    assert .99<log.frame.Z_g.mean()<1.01
    s=select_samples(log,start_s=2,end_s=5)
    assert len(s)==1201
    s2=select_samples(log,start_time=str(start+pd.Timedelta(seconds=4)),end_sample=2001)
    assert len(s2)==401
    with pytest.raises(ValueError):select_samples(log,start_s=31)
    with pytest.raises(ValueError):select_samples(log,start_sample=400,end_sample=100)


def test_psd_peak_amplitude_and_screening(tmp_path):
    p=tmp_path/'signal.xls';synthetic_export(p)
    refpath=tmp_path/'ref.txt';reference_file(refpath)
    ref=read_labware_reference(refpath)
    assert len(ref)==3 and ref.Labware.nunique()==2
    log=read_vse(p)
    result=analyze(log,ref,welch_s=5.12,fft_snapshot_s=5,axis='Vector',rms_bin_s=1,
                   fmin=5,fmax=150,band_halfwidth_Hz=.5)
    s=result['spectrum']
    closest=(s.Frequency_Hz-40).abs().idxmin()
    assert 35< s.Frequency_Hz.iloc[closest]<45
    single=labware_screen(s,ref,labware='8 Well Rectangular')
    assert len(single)==2
    a=single.iloc[0]
    # Nominal amplitude 0.2 m/s2 PEAK => .1414 RMS at 40 Hz on X.
    assert abs(a.PSD_band_RMS_m_s2-.2/np.sqrt(2))<.035
    assert a.Tonal_excess_to_reference_ratio>3
    assert not result['peak_table'].empty
    assert 'shorter than' in ' '.join(result['quality']['warnings'])
    assert len(result['time_metrics'])==30
    assert len(plot_time_rms(result).data)==4
    assert len(plot_time_shocks(result).data)==2
    assert len(plot_psd(result,ref).data)>=1
    assert len(plot_fft(result).data)>=1
    assert len(plot_mode_ranking(result).data)==1
    heat=spectrogram(result['selected'],log.quality['fs_est_Hz'],axis='Z',nperseg=1024)
    assert not heat.empty and len(plot_spectrogram(heat).data)==1
    out=export_report(result,tmp_path/'report')
    assert (out/'quality_and_limitations.json').exists()
    assert (out/'labware_screening.csv').exists()


def test_sampling_gaps_not_silently_spliced(tmp_path):
    p=tmp_path/'signal.csv';synthetic_export(p)
    log=read_vse(p)
    d=log.frame.copy()
    d.loc[6000:,'Time']+=pd.Timedelta(seconds=4)
    assert len(d)==len(log.frame)
    from vse_vibration_core import uninterrupted_segments
    assert len(uninterrupted_segments(d,400))==2
    s,fft,qa=compute_spectrum(d,400,welch_s=5.12)
    assert qa['spectral_coverage_pct']<60


def test_unmeasurable_mode(tmp_path):
    p=tmp_path/'signal.csv';synthetic_export(p)
    log=read_vse(p)
    spectrum,_,_=compute_spectrum(log.frame,400)
    ref=pd.DataFrame([dict(Labware='X',Harmonic=1,Frequency_Hz=400,
                           Spacing_mm=1,Stabilization_h=4,
                           Threshold_m_s2=.002)])
    screen=labware_screen(spectrum,ref)
    assert screen.Status.iloc[0]=='Outside measurable spectrum'

def test_colab_notebook_uses_native_choose_files_not_typed_paths():
    """The source picker is a standalone cell, not an ipywidgets callback."""
    import ast
    import json
    notebook = Path(__file__).resolve().parents[1] / "VSE16g_Vibration_Labware_Analytics.ipynb"
    content = json.loads(notebook.read_text(encoding="utf-8"))
    assert content["nbformat"] == 4
    cells = content["cells"]
    source = [("".join(c["source"])) for c in cells]
    upload_indexes = [i for i, s in enumerate(source) if "uploaded = files.upload()" in s]
    assert len(upload_indexes) == 1
    uploader = source[upload_indexes[0]]
    assert "from google.colab import files" in uploader
    assert "VSE_RAW_PATH" in uploader and "VSE_REF_PATH" in uploader
    assert "'.xls'" in uploader and "'.xlsx'" in uploader and "'.csv'" in uploader
    assert "'.txt'" in uploader
    assert "UPLOAD_NOW = False" not in "\n".join(source)
    dashboard_index = next(i for i, s in enumerate(source) if "source_panel=w.VBox" in s)
    handler_index = next(i for i, s in enumerate(source) if "load_button.on_click(_load)" in s)
    assert upload_indexes[0] < dashboard_index < handler_index
    assert "if raw_file.value and Path(raw_file.value).is_file():" in source[handler_index]
    assert "    _load()" in source[handler_index]
    dashboard = source[dashboard_index]
    assert "raw_file_status" in dashboard and "ref_file_status" in dashboard
    assert "raw_file,ref_file" not in dashboard.split("source_panel=w.VBox", 1)[1].split("time_panel=", 1)[0]
    for cell in cells:
        if cell["cell_type"] == "code":
            code = "".join(cell["source"])
            if not code.lstrip().startswith("%"):
                ast.parse(code)
