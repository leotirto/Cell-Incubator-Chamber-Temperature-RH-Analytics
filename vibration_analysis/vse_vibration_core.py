"""Three-axis VSE-16g acceleration and labware vibration *screening* analytics.

Acceleration is measured at the sensor, NOT automatically at the culture liquid.
Reference thresholds are unvalidated model inputs; the comparison is descriptive,
not a prediction that cell patterns occur. The input source is never modified.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import re
import warnings
from typing import Mapping

import numpy as np
import pandas as pd
from scipy import signal, ndimage
import plotly.graph_objects as go
from plotly.subplots import make_subplots

G0 = 9.80665
AXES = ('X_g', 'Y_g', 'Z_g')
COLORS = {'X': '#2563eb', 'Y': '#d97706', 'Z': '#0d9488', 'Vector': '#8b5cf6'}
SOURCE_URLS = {
    'manufacturer': 'https://convergenceinstruments.com/product/vibration-meter-datalogger-vibration-sentry-e16g/',
    'noise_table': 'https://convergenceinstruments.com/pdf/vibration_sentry_e_datasheet.pdf',
    'cell_culture_patterns': 'https://documents.thermofisher.com/TFS-Assets/BID/Technical-Notes/vibration-patterns-cell-culture-vessels-technical-note.pdf',
    'faraday_review': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC8094912/'
}

@dataclass
class VibrationLog:
    name: str
    frame: pd.DataFrame
    quality: dict


def _find_header(path: Path) -> tuple[int, str]:
    """Find a tab/comma header below optional Vibration Sentry metadata."""
    with path.open('rb') as fh:
        first = fh.read(512)
    if first.startswith(b'\xd0\xcf\x11\xe0') or first.startswith(b'PK\x03\x04'):
        return -1, ''
    with path.open('r', encoding='utf-8-sig', errors='replace') as f:
        for i, line in enumerate(f):
            if i >= 50:
                break
            tokens = line.strip().lower()
            if ('time' in tokens or 'date' in tokens or 'timestamp' in tokens) and (
                re.search(r'(^|[\t,;])\s*x\s*:', line, re.I) or
                re.search(r'(^|[\t,;])\s*x[ _-]?(?:g|accel|axis)', line, re.I)):
                return i, '\t' if '\t' in line else (';' if ';' in line else ',')
    raise ValueError('No X/Y/Z acceleration header found within first 50 lines')


def read_vse(path: str | Path) -> VibrationLog:
    """Read text .xls/.csv or genuine Excel .xls/.xlsx. Source samples indexed 1..N.

    The observed VSE export uses a two-line preamble and tab-separated columns,
    despite the .xls extension. Timestamp resolution is preserved to microseconds.
    """
    path = Path(path)
    head_idx, delimiter = _find_header(path)
    if head_idx < 0:
        try:
            raw = pd.read_excel(path, engine='xlrd' if path.suffix.lower() == '.xls' else 'openpyxl')
        except ImportError as e:
            raise ImportError('True binary .xls requires `pip install xlrd`') from e
    else:
        raw = pd.read_csv(path, sep=delimiter, skiprows=head_idx, low_memory=False,
                          encoding='utf-8-sig')
    raw.columns = [str(x).strip() for x in raw.columns]
    def name_for(axis: str) -> str:
        matches = [c for c in raw.columns if re.match(r'^\s*'+axis+r'\s*(?::|[ _-])', c, re.I)]
        if len(matches) != 1:
            raise ValueError(f'Expected one {axis}-axis acceleration column, found {matches}')
        return matches[0]
    x,y,z = [name_for(c) for c in 'XYZ']
    time_cols = [c for c in raw.columns if 'time' in c.lower() or 'date' in c.lower()]
    if not time_cols:
        raise ValueError('No timestamp column found')
    tc = time_cols[0]
    series = raw[tc]
    if pd.api.types.is_datetime64_any_dtype(series):
        dates = series
    else:
        dates = pd.to_datetime(series.astype(str).str.strip(), errors='coerce',
                               format='%Y/%m/%d %H:%M:%S.%f')
        if dates.notna().mean() < 0.9:
            dates = pd.to_datetime(series, errors='coerce')
    values = raw[[x,y,z]].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float64)
    units = [c.lower() for c in (x,y,z)]
    if all(re.search(r'\bg\b|:\s*g\b|_g\b', c) for c in units):
        unit='g'
    elif all(re.search(r'm/s(?:\^?2|²)', c) for c in units):
        unit='m/s2'
    else:
        raise ValueError(f'Acceleration axis units must agree and specify g or m/s²: {[x,y,z]}')
    if unit == 'm/s2':
        values /= G0
    source_rows = np.arange(1,len(raw)+1,dtype=np.int32)
    valid = dates.notna() & np.isfinite(values).all(axis=1)
    rejected = int((~valid).sum())
    frame = pd.DataFrame(values[valid], columns=AXES)
    frame.insert(0,'Time',dates[valid].to_numpy())
    frame.insert(1,'Sample',source_rows[valid])
    frame = frame.sort_values(['Time','Sample'],kind='stable').reset_index(drop=True)
    if len(frame)<50:
        raise ValueError('At least 50 complete acceleration samples are required')
    ns = frame.Time.to_numpy(dtype='datetime64[ns]').astype('int64')
    dt = np.diff(ns).astype(np.float64) * 1e-9
    positive = dt[dt>0]
    if len(positive)<2:
        raise ValueError('No usable increasing timestamps')
    typical=float(np.median(positive))
    fs=float(round(1/typical,4))
    dups=int((dt<=0).sum())
    gaps=int((dt>1.5*typical).sum())
    q = {'file':path.name,'rows_raw':int(len(raw)), 'rows_valid':int(len(frame)),
         'rejected_invalid_rows':rejected,'start':str(frame.Time.iloc[0]),
         'end':str(frame.Time.iloc[-1]),
         'duration_s':float((frame.Time.iloc[-1]-frame.Time.iloc[0]).total_seconds()),
         'fs_est_Hz':fs,'dt_median_s':typical,'dt_min_s':float(positive.min()),
         'dt_max_s':float(positive.max()),'time_duplicates':dups,
         'sampling_gaps_gt_1_5_period':gaps,
         'sampling_jitter_us_p99':float(np.quantile(abs(positive-typical),.99)*1e6),
         'source_units':unit,'nyquist_Hz':fs/2,
         'acceleration_g_axis_means':[float(frame[c].mean()) for c in AXES],
         'axis_peak_abs_g':[float(frame[c].abs().max()) for c in AXES],
         'samples_near_plusminus_16g':int((np.abs(values[valid])>=15.9).any(axis=1).sum())}
    return VibrationLog(path.name,frame,q)


def read_labware_reference(path: str | Path) -> pd.DataFrame:
    """Parse provided labware model table. No theoretical validation is implied."""
    text=Path(path).read_text(encoding='utf-8-sig')
    rows=[];labware=None;stabilization=np.nan
    for line in text.splitlines():
        if line.strip().upper().startswith('LABWARE:'):
            m=re.search(r'LABWARE:\s*(.*?)\s*\|\s*Stabilization Window:\s*([\d.]+)',line,re.I)
            if not m:
                raise ValueError('Malformed LABWARE heading')
            labware=m.group(1).strip();stabilization=float(m.group(2))
            continue
        if labware and re.match(r'^\s*\d+\s*\|',line):
            fields=[z.strip() for z in line.split('|')]
            if len(fields)!=5:
                raise ValueError(f'Malformed labware mode row: {line}')
            harmonic,f,spacing,force,threshold=map(float,fields)
            rows.append((labware,stabilization,int(harmonic),f,spacing,force,threshold))
    if not rows:
        raise ValueError('No labware harmonic entries found')
    out=pd.DataFrame(rows,columns=['Labware','Stabilization_h','Harmonic','Frequency_Hz',
                                    'Spacing_mm','F_limit_pN','Threshold_m_s2'])
    if (out.Threshold_m_s2<=0).any() or (out.Frequency_Hz<=0).any():
        raise ValueError('Reference frequencies and acceleration thresholds must be positive')
    return out


def select_samples(log: VibrationLog, *, start_time: str | None=None,
                   end_time: str | None=None,start_s:float | None=None,
                   end_s:float | None=None,start_sample:int | None=None,
                   end_sample:int | None=None)->pd.DataFrame:
    """Inclusive absolute timestamp, elapsed seconds and original sample filters.

    Constraints intersect (AND); blank/None means no limit. Times are device-local.
    """
    d=log.frame
    mask=np.ones(len(d),dtype=bool)
    if start_sample is not None:mask &= d.Sample.to_numpy()>=int(start_sample)
    if end_sample is not None:mask &= d.Sample.to_numpy()<=int(end_sample)
    if start_sample is not None and end_sample is not None and start_sample>end_sample:
        raise ValueError('start_sample must not exceed end_sample')
    first=d.Time.iloc[0]
    lo=pd.Timestamp(start_time) if start_time else None
    hi=pd.Timestamp(end_time) if end_time else None
    if start_s is not None:lo2=first+pd.Timedelta(seconds=float(start_s));lo=max(lo,lo2) if lo is not None else lo2
    if end_s is not None:hi2=first+pd.Timedelta(seconds=float(end_s));hi=min(hi,hi2) if hi is not None else hi2
    if lo is not None:mask &= d.Time.to_numpy()>=lo.to_datetime64()
    if hi is not None:mask &= d.Time.to_numpy()<=hi.to_datetime64()
    if not mask.any():raise ValueError('No samples in selected time/sample window')
    selected=d.loc[mask].copy().reset_index(drop=True)
    if len(selected)<100:raise ValueError('Window too short: at least 100 samples required')
    return selected


def uninterrupted_segments(d: pd.DataFrame, fs: float, max_gap_factor:float=1.5) -> list[tuple[int,int]]:
    dates=d.Time.to_numpy(dtype='datetime64[ns]').astype('int64')
    diff=np.diff(dates).astype(np.float64)*1e-9
    boundary=np.flatnonzero((diff>max_gap_factor/fs)|(diff<=0))+1
    edges=np.r_[0,boundary,len(d)]
    return [(int(a),int(b)) for a,b in zip(edges[:-1],edges[1:])]


def _dominant_segment(d: pd.DataFrame,fs:float)->tuple[np.ndarray,dict]:
    segments=uninterrupted_segments(d,fs)
    a,b=max(segments,key=lambda ab:ab[1]-ab[0])
    if b-a < int(fs*4):
        raise ValueError('No uninterrupted segment ≥ 4 s for reliable FFT/PSD. Select another interval.')
    return d.loc[a:b-1,list(AXES)].to_numpy(dtype=np.float64),{
        'segments':len(segments),'longest_segment_samples':b-a,
        'spectral_coverage_pct':100*(b-a)/len(d),'segment_start':str(d.Time.iloc[a]),
        'segment_end':str(d.Time.iloc[b-1]), 'segment_index_start':a}


def _sos_highpass(a:np.ndarray,fs:float,cutoff:float=0.5)->np.ndarray:
    if not (0<cutoff<fs/2):raise ValueError('Invalid highpass cutoff')
    sos=signal.butter(3,cutoff,btype='highpass',fs=fs,output='sos')
    return signal.sosfiltfilt(sos,a,axis=0)


def time_metrics(d:pd.DataFrame,fs:float,bin_s:float=1.0,highpass_Hz:float=0.5)->tuple[pd.DataFrame,dict]:
    """Vibration-only highpass RMS and peaks; excludes gravity/slow orientation drift."""
    if bin_s<=0:raise ValueError('time bin must be positive')
    a=d.loc[:,AXES].to_numpy(dtype=np.float64)*G0
    segments=uninterrupted_segments(d,fs)
    accum=[]
    sample_count=max(10,int(round(fs*bin_s)))
    for a0,b0 in segments:
        if b0-a0<max(30,int(fs*.5)):
            continue
        dyn=_sos_highpass(a[a0:b0],fs,min(highpass_Hz,fs*.2))
        dynamic_norm=np.sqrt(np.sum(dyn**2,axis=1))
        count=b0-a0
        for lo in range(0,count,sample_count):
            hi=min(lo+sample_count,count)
            block=dyn[lo:hi]
            mags=dynamic_norm[lo:hi]
            if len(block)<10:continue
            rms=np.sqrt(np.mean(block**2,axis=0))
            accum.append((d.Time.iloc[a0+lo],float((d.Time.iloc[a0+lo]-d.Time.iloc[0]).total_seconds()),
                          len(block),*rms,float(np.sqrt(np.mean(mags**2))),float(np.max(mags)),
                          float(np.max(np.abs(block)))))
    columns=['Time','Elapsed_s','Samples','RMS_X_m_s2','RMS_Y_m_s2','RMS_Z_m_s2',
             'RMS_Vector_m_s2','Peak_Vector_m_s2','Peak_Axis_m_s2']
    metrics=pd.DataFrame(accum,columns=columns)
    if metrics.empty:raise ValueError('No time bins were long enough to characterize')
    detail={'time_bin_s':bin_s,'highpass_Hz':highpass_Hz,
            'median_vector_rms_m_s2':float(metrics.RMS_Vector_m_s2.median()),
            'p95_vector_rms_m_s2':float(metrics.RMS_Vector_m_s2.quantile(.95)),
            'p99_vector_rms_m_s2':float(metrics.RMS_Vector_m_s2.quantile(.99)),
            'max_bin_rms_m_s2':float(metrics.RMS_Vector_m_s2.max()),
            'max_bin_rms_time':str(metrics.loc[metrics.RMS_Vector_m_s2.idxmax(),'Time']),
            'peak_dynamic_sample_m_s2':float(metrics.Peak_Vector_m_s2.max())}
    return metrics,detail


def compute_spectrum(d:pd.DataFrame,fs:float,welch_s:float=20.48,
                     fft_snapshot_s:float=30.0,axis:str='Vector')->tuple[pd.DataFrame,pd.DataFrame,dict]:
    """One-sided Welch PSD and single-window amplitude FFT, in m/s² units.

    PSD is the primary amplitude statistic. Snapshot FFT uses Hann coherent-gain
    correction for *peak* sinusoidal amplitude and may leak with noninteger cycles.
    Vector PSD means sum of per-axis PSD energies, not vector-summed waveforms.
    """
    if axis not in ('X','Y','Z','Vector'):raise ValueError('Unknown axis')
    raw,qa=_dominant_segment(d,fs)
    nperseg=min(len(raw),max(256,int(round(welch_s*fs))))
    if nperseg<256:raise ValueError('Insufficient samples for Welch PSD')
    freq, p = signal.welch(raw*G0,fs=fs,window='hann',nperseg=nperseg,
                           noverlap=nperseg//2,axis=0,detrend='constant',scaling='density')
    p=np.asarray(p)
    energy=np.sum(p,axis=1)
    spectra=pd.DataFrame({'Frequency_Hz':freq,'PSD_X':p[:,0],'PSD_Y':p[:,1],
                          'PSD_Z':p[:,2],'PSD_Vector':energy,
                          'ASD_Vector_m_s2_sqrtHz':np.sqrt(energy)})
    wanted=max(256,int(fft_snapshot_s*fs))
    wanted=min(len(raw),wanted)
    offset=(len(raw)-wanted)//2
    snap=raw[offset:offset+wanted]*G0
    centered=snap-np.mean(snap,axis=0)
    hann=signal.windows.hann(len(centered),sym=False)
    fft=np.fft.rfft(centered*hann[:,None],axis=0)
    ffft=np.fft.rfftfreq(len(centered),1/fs)
    peak=np.abs(fft)*2/np.sum(hann)
    peak[0,:]=0.0
    if len(centered)%2==0:peak[-1,:]/=2
    snap_fft=pd.DataFrame({'Frequency_Hz':ffft,'Peak_X_m_s2':peak[:,0],
                           'Peak_Y_m_s2':peak[:,1],'Peak_Z_m_s2':peak[:,2],
                           'Peak_Vector_RSS_m_s2':np.sqrt(np.sum(peak**2,axis=1))})
    qa.update({'welch_nperseg':nperseg,'welch_resolution_Hz':fs/nperseg,
               'FFT_snapshot_s':len(centered)/fs,'FFT_offset_s':(qa['segment_index_start']+offset)/fs,
               'spectral_axis':axis})
    return spectra,snap_fft,qa


def spectral_peaks(spectrum:pd.DataFrame,fmin:float=1,fmax:float=300,
                   max_peaks:int=25,min_prominence_dB:float=6.0)->pd.DataFrame:
    """Frequency-local power peaks; approximate prominence above median-smoothed floor."""
    f=spectrum.Frequency_Hz.to_numpy()
    p=spectrum.PSD_Vector.to_numpy()
    if len(f)<5:return pd.DataFrame()
    df=f[1]-f[0]
    width=max(9,int(np.ceil(5/max(df,1e-9)))//2*2+1)
    baseline=ndimage.median_filter(p,size=width,mode='nearest')
    db=10*np.log10(np.maximum(p,1e-20)/np.maximum(baseline,1e-20))
    peaks,_=signal.find_peaks(db,height=min_prominence_dB,distance=max(2,int(.4/df)))
    peaks=peaks[(f[peaks]>=fmin)&(f[peaks]<=min(fmax,f[-1]))]
    peaks=peaks[np.argsort(p[peaks])[-max_peaks:][::-1]]
    return pd.DataFrame({'Frequency_Hz':f[peaks],
                         'PSD_Vector_(m_s2)^2_Hz':p[peaks],
                         'Spectral_excess_dB':db[peaks],
                         'Approx_ASD_m_s2_sqrtHz':np.sqrt(p[peaks])})


def labware_screen(spectrum:pd.DataFrame, reference:pd.DataFrame,
                   *,labware:str|None=None,halfwidth_Hz:float=.5,
                   halfwidth_relative:float=.01)->pd.DataFrame:
    """Screen measured PSD-band RMS against supplied acceleration thresholds.

    The threshold may describe peak, RMS, axis-specific or plate-level amplitude;
    because it is not documented, the ratio is NOT a pass/fail result.
    Measurements at the logger cannot establish well-bottom acceleration.
    """
    freq=spectrum.Frequency_Hz.to_numpy(dtype=float)
    p=spectrum.PSD_Vector.to_numpy(dtype=float)
    if len(freq)<2:raise ValueError('Not enough spectral frequency bins')
    if halfwidth_Hz<=0 or halfwidth_relative<0:raise ValueError('Band width must be positive')
    resolution=freq[1]-freq[0]
    subset=reference if labware is None or labware=='All' else reference.loc[reference.Labware.eq(labware)]
    output=[]
    for r in subset.itertuples(index=False):
        f0=float(r.Frequency_Hz)
        half=max(float(halfwidth_Hz),f0*halfwidth_relative, resolution)
        inside=(freq>=f0-half)&(freq<=f0+half)
        if not inside.any() or f0>freq.max():
            output.append(dict(Labware=r.Labware,Harmonic=r.Harmonic,Freq_Hz=f0,
                               Threshold_m_s2=r.Threshold_m_s2,Status='Outside measurable spectrum'))
            continue
        rms=float(np.sqrt(np.sum(p[inside])*resolution))
        vicinity=(freq>=f0-max(5,5*half))&(freq<=f0+max(5,5*half))&(~inside)
        floor=float(np.median(p[vicinity])) if vicinity.any() else float(np.median(p[inside]))
        excess=float(np.sqrt(np.sum(np.maximum(p[inside]-floor,0))*resolution))
        strongest=np.flatnonzero(inside)[np.argmax(p[inside])]
        local_peak=float(freq[strongest]);strength=float(10*np.log10(max(p[strongest],1e-20)/max(floor,1e-20)))
        output.append(dict(Labware=r.Labware,Harmonic=int(r.Harmonic),Freq_Hz=f0,
             Spacing_mm=r.Spacing_mm,Stabilization_h=r.Stabilization_h,
             Threshold_m_s2=r.Threshold_m_s2,Halfwidth_Hz=half,
             PSD_band_RMS_m_s2=rms,Approx_tonal_excess_RMS_m_s2=excess,
             Band_RMS_to_reference_ratio=rms/r.Threshold_m_s2,
             Tonal_excess_to_reference_ratio=excess/r.Threshold_m_s2,
             Strongest_bin_Hz=local_peak,Offset_Hz=local_peak-f0,
             Strongest_bin_above_local_floor_dB=strength,
             Status='Screening only; threshold type/transfer unverified'))
    return pd.DataFrame(output)


def spectrogram(d:pd.DataFrame,fs:float,axis:str='Z',nperseg:int=4096,
                fmax:float=300, max_frequency_bins:int=500,max_time_bins:int=700)->pd.DataFrame:
    """Windowed FFT: log power vs time, frequency. No interpolation across gaps."""
    raw,qa=_dominant_segment(d,fs)
    nperseg=min(len(raw),int(nperseg))
    if nperseg<64:raise ValueError('Spectrogram nperseg must be at least 64')
    indices={'X':[0],'Y':[1],'Z':[2],'Vector':[0,1,2]}
    chosen=indices.get(axis)
    if chosen is None:raise ValueError('axis must be X,Y,Z or Vector')
    accumulation=None
    for idx in chosen:
        freq,t_rel, power=signal.spectrogram(raw[:,idx]*G0,fs,nperseg=nperseg,
                 noverlap=nperseg//2,window='hann',scaling='density',mode='psd')
        accumulation=power if accumulation is None else accumulation+power
    mask=(freq>=.5)&(freq<=min(fmax,fs/2))
    selected=np.flatnonzero(mask)
    f_stride=max(1,int(np.ceil(len(selected)/max_frequency_bins)))
    t_stride=max(1,int(np.ceil(len(t_rel)/max_time_bins)))
    ind=selected[::f_stride]
    times=pd.Timestamp(qa['segment_start'])+pd.to_timedelta(t_rel[::t_stride],unit='s')
    out=pd.DataFrame(10*np.log10(np.maximum(accumulation[np.ix_(ind,np.arange(0,len(t_rel),t_stride))],1e-18)),
                     index=freq[ind],columns=times)
    out.index.name='Frequency_Hz'
    return out


def summary_by_time_segment(metrics:pd.DataFrame,segment_s:float=300)->pd.DataFrame:
    """Descriptive changes in activity over the recording, e.g. first five minutes."""
    if segment_s<=0:raise ValueError('segment_s must be positive')
    group=(metrics.Elapsed_s//segment_s).astype(int)
    tab=(metrics.groupby(group,sort=True)
         .agg(Start=('Time','first'),End=('Time','last'),Count=('Samples','sum'),
              RMS_Median=('RMS_Vector_m_s2','median'),
              RMS_P95=('RMS_Vector_m_s2',lambda x: x.quantile(.95)),
              Peak_Max=('Peak_Vector_m_s2','max'))).reset_index(drop=True)
    tab.insert(0,'Segment',np.arange(len(tab))+1)
    return tab


def analyze(log:VibrationLog,reference:pd.DataFrame|None=None,*,
            start_time:str|None=None,end_time:str|None=None,start_s:float|None=None,
            end_s:float|None=None,start_sample:int|None=None,end_sample:int|None=None,
            welch_s:float=20.48,fft_snapshot_s:float=30,axis:str='Vector',
            rms_bin_s:float=1,highpass_Hz:float=.5,
            labware:str='All',band_halfwidth_Hz:float=.5,
            fmin:float=1,fmax:float=300)->dict:
    d=select_samples(log,start_time=start_time,end_time=end_time,start_s=start_s,
                     end_s=end_s,start_sample=start_sample,end_sample=end_sample)
    fs=log.quality['fs_est_Hz']
    metrics,stats=time_metrics(d,fs,rms_bin_s,highpass_Hz)
    spectrum,fft,qa=compute_spectrum(d,fs,welch_s,fft_snapshot_s,axis)
    peaks=spectral_peaks(spectrum,fmin,fmax)
    matches=(labware_screen(spectrum,reference,labware=labware,halfwidth_Hz=band_halfwidth_Hz)
             if reference is not None else pd.DataFrame())
    warnings_list=[]
    duration=(d.Time.iloc[-1]-d.Time.iloc[0]).total_seconds()
    if reference is not None and duration<3600*float(reference.Stabilization_h.max()):
        warnings_list.append('Selected exposure is shorter than the 4-hour stabilization window in the model.')
    if qa['spectral_coverage_pct']<95:
        warnings_list.append('Spectral PSD uses only the longest uninterrupted block; inspect missing/gapped data.')
    if fmax>fs/2:
        warnings_list.append('Frequencies above Nyquist are not measurable and were omitted.')
    warnings_list.extend([
      'Model threshold definition (peak versus RMS, location, direction) is unverified: ratios are screening values, NOT pass/fail.',
      'Sensor-to-labware transfer functions, culture medium depth, cell type and attachment state are unknown.',
      'No anti-aliasing filter is guaranteed; out-of-band excitation may alias into the measured spectrum.',
      'MEMS noise floor and transient handling/movement may dominate small spectral amplitudes.'
    ])
    return {'log_name':log.name,'selected':d,'time_metrics':metrics,'time_sections':summary_by_time_segment(metrics),
            'spectrum':spectrum,'fft_snapshot':fft,'spectrogram':None,
            'peak_table':peaks,'labware_matches':matches,'quality':{**log.quality,**qa,**stats,
              'selected_samples':len(d),'selected_from':str(d.Time.iloc[0]),
              'selected_to':str(d.Time.iloc[-1]),'selected_duration_s':duration,
              'reference_labware_options':[] if reference is None else reference.Labware.unique().tolist(),
              'warnings':warnings_list},'settings':dict(welch_s=welch_s,
              fft_snapshot_s=fft_snapshot_s,axis=axis,rms_bin_s=rms_bin_s,
              highpass_Hz=highpass_Hz,labware=labware,
              band_halfwidth_Hz=band_halfwidth_Hz,fmin=fmin,fmax=fmax)}


def _layout(fig,title,*,height=540):
    fig.update_layout(template='plotly_white',title={'text':title,'x':.02},
                      height=height,margin=dict(l=65,r=25,t=80,b=80),
                      font=dict(family='Arial',size=12),hovermode='x unified',
                      legend=dict(orientation='h',y=-.22))
    return fig


def plot_time_rms(result:dict)->go.Figure:
    m=result['time_metrics'];fig=go.Figure()
    for axis in ('X','Y','Z','Vector'):
        fig.add_trace(go.Scattergl(x=m.Time,y=m[f'RMS_{axis}_m_s2'],name=axis,
                                   mode='lines',line=dict(color=COLORS[axis],width=1.4)))
    fig.update_yaxes(title_text='High-pass vibration RMS (m/s²)')
    fig.update_xaxes(title_text='Logger local time')
    return _layout(fig,'Acceleration RMS vs time — gravity/drift removed',height=570)


def plot_time_shocks(result:dict)->go.Figure:
    m=result['time_metrics'];fig=go.Figure()
    fig.add_trace(go.Scattergl(x=m.Time,y=m.Peak_Vector_m_s2,name='Peak/vector',
                                mode='lines',line=dict(color='#d97706',width=1.4)))
    fig.add_trace(go.Scattergl(x=m.Time,y=m.RMS_Vector_m_s2,name='RMS/vector',
                                mode='lines',line=dict(color='#2563eb',width=1.4)))
    fig.update_yaxes(title_text='Acceleration (m/s²)')
    fig.update_xaxes(title_text='Logger local time')
    return _layout(fig,'Peak/shock events and time-varying energy',height=560)


def plot_psd(result:dict,reference:pd.DataFrame|None=None,axis:str='Vector',
             log_y:bool=True)->go.Figure:
    s=result['spectrum'];cfg=result['settings'];fig=go.Figure()
    select=('X','Y','Z') if axis=='Vector' else (axis,)
    for key in select:
        fig.add_trace(go.Scattergl(x=s.Frequency_Hz,y=s[f'PSD_{key}'],name=key,
             mode='lines',line=dict(color=COLORS[key],width=1.5)))
    if axis=='Vector':
        fig.add_trace(go.Scattergl(x=s.Frequency_Hz,y=s.PSD_Vector,name='Vector energy (Σ axes)',
             mode='lines',line=dict(color=COLORS['Vector'],width=1.5)))
    if reference is not None:
        focus=(reference if cfg['labware']=='All' else reference[reference.Labware.eq(cfg['labware'])])
        # Display a limited number of mode markers, without confusing the Y-axis units.
        for f0 in focus.Frequency_Hz.unique()[:15]:
            if cfg['fmin']<=f0<=cfg['fmax']:
                fig.add_vline(x=float(f0),line_dash='dot',line_color='#94a3b8',opacity=.5)
    fig.update_xaxes(title_text='Frequency (Hz)',range=[cfg['fmin'],cfg['fmax']])
    fig.update_yaxes(title_text='Acceleration PSD ((m/s²)²/Hz)',type='log' if log_y else 'linear')
    return _layout(fig,'Welch power spectral density — main frequency diagnostic',height=620)


def plot_fft(result:dict,axis:str='Vector')->go.Figure:
    d=result['fft_snapshot'];cfg=result['settings'];fig=go.Figure()
    fields={'X':'Peak_X_m_s2','Y':'Peak_Y_m_s2','Z':'Peak_Z_m_s2',
            'Vector':'Peak_Vector_RSS_m_s2'}
    keys=('X','Y','Z') if axis=='Vector' else (axis,)
    for key in keys:
        fig.add_trace(go.Scattergl(x=d.Frequency_Hz,y=d[fields[key]],
                        name=key,mode='lines',line=dict(color=COLORS[key],width=1.4)))
    fig.update_xaxes(title_text='Frequency (Hz)',range=[cfg['fmin'],cfg['fmax']])
    fig.update_yaxes(title_text='Hann-windowed snapshot sinusoidal peak amplitude (m/s²)')
    return _layout(fig,'FFT amplitude — centered snapshot only, NOT whole-record PSD',height=530)


def plot_spectrogram(heat:pd.DataFrame)->go.Figure:
    fig=go.Figure(data=[go.Heatmap(x=heat.columns,y=heat.index,z=heat.to_numpy(),
            colorscale='Viridis',colorbar=dict(title='PSD dB re 1 (m/s²)²/Hz'),
            hovertemplate='%{x}<br>%{y:.2f} Hz<br>%{z:.1f} dB<extra></extra>')])
    fig.update_xaxes(title_text='Logger local time')
    fig.update_yaxes(title_text='Frequency (Hz)')
    return _layout(fig,'Time-frequency spectrogram — transient vs persistent tones',height=630)


def plot_mode_ranking(result:dict,top:int=18)->go.Figure:
    d=result['labware_matches'];fig=go.Figure()
    if not d.empty and 'Tonal_excess_to_reference_ratio' in d:
        valid=d.dropna(subset=['Tonal_excess_to_reference_ratio']).nlargest(top,'Tonal_excess_to_reference_ratio')
        labels=valid.Labware.str.replace(' Microplate','',regex=False)+' | '+valid.Freq_Hz.map(lambda x:f'{x:g} Hz')
        fig.add_trace(go.Bar(x=valid.Tonal_excess_to_reference_ratio,y=labels,orientation='h',
            text=valid.Tonal_excess_to_reference_ratio.round(2),textposition='outside',
            marker_color='#0d9488'))
        fig.update_yaxes(autorange='reversed')
    fig.update_xaxes(title_text='Approx. tonal-excess band RMS / supplied threshold (dimensionless)')
    return _layout(fig,'LABWARE SCREENING ONLY — modal ratios; not biological risk',height=680)


def plot_time_sections(result:dict)->go.Figure:
    d=result['time_sections'];fig=go.Figure()
    fig.add_bar(x=d.Segment,y=d.RMS_Median,name='Median RMS',marker_color='#2563eb')
    fig.add_bar(x=d.Segment,y=d.RMS_P95,name='95th percentile RMS',marker_color='#d97706')
    fig.update_layout(barmode='group');fig.update_xaxes(title_text='Sequential 5-minute segment')
    fig.update_yaxes(title_text='High-pass vector RMS (m/s²)')
    return _layout(fig,'Vibration levels by 5-minute section',height=515)




def modal_time_exposure(d:pd.DataFrame,fs:float,matches:pd.DataFrame,*,
                        top_modes:int=8,nperseg:int=8192)->pd.DataFrame:
    """Short-time band RMS screening at top reference modes.

    Uses 50%-overlapping Hann spectra: the `above ratio=1` fraction is a
    descriptive fraction of overlapping time bins, not a statistically
    independent exposure or an accepted cell-attachment risk criterion.
    """
    columns=['Time','Elapsed_s','Labware','Harmonic','Reference_Hz','Threshold_m_s2',
             'Band_RMS_m_s2','Screening_ratio']
    if matches.empty or not {'Tonal_excess_to_reference_ratio','Freq_Hz'}.issubset(matches):
        return pd.DataFrame(columns=columns)
    ref=matches.dropna(subset=['Tonal_excess_to_reference_ratio']).nlargest(
        int(top_modes),'Tonal_excess_to_reference_ratio')
    if ref.empty:return pd.DataFrame(columns=columns)
    raw,qa=_dominant_segment(d,fs)
    nperseg=min(len(raw),int(nperseg))
    if nperseg<256:return pd.DataFrame(columns=columns)
    accumulated=None
    for i in range(3):
        freqs,times,psd=signal.spectrogram(raw[:,i]*G0,fs,nperseg=nperseg,
                     noverlap=nperseg//2,window='hann',scaling='density',mode='psd')
        accumulated=psd if accumulated is None else accumulated+psd
    df=freqs[1]-freqs[0]
    rows=[]
    origin=pd.Timestamp(qa['segment_start'])
    for refrow in ref.itertuples(index=False):
        f0=float(refrow.Freq_Hz)
        hw=float(refrow.Halfwidth_Hz)
        mask=(freqs>=f0-hw)&(freqs<=f0+hw)
        if not mask.any():continue
        band=np.sqrt(np.sum(accumulated[mask,:],axis=0)*df)
        for time_s, r in zip(times,band):
            rows.append((origin+pd.Timedelta(seconds=float(time_s)),
                  (origin-d.Time.iloc[0]).total_seconds()+float(time_s),refrow.Labware,
                  int(refrow.Harmonic),f0,float(refrow.Threshold_m_s2),float(r),
                  float(r/refrow.Threshold_m_s2)))
    return pd.DataFrame(rows,columns=columns)


def modal_time_summary(series:pd.DataFrame)->pd.DataFrame:
    if series.empty:
        return pd.DataFrame(columns=['Labware','Harmonic','Reference_Hz','Window_count',
             'Median_ratio','Max_ratio','Fraction_windows_ratio_above_one_pct'])
    return (series.groupby(['Labware','Harmonic','Reference_Hz'],as_index=False)
            .agg(Window_count=('Screening_ratio','size'),
                 Median_ratio=('Screening_ratio','median'),
                 Max_ratio=('Screening_ratio','max'),
                 Fraction_windows_ratio_above_one_pct=(
                     'Screening_ratio',lambda s:float((s>1).mean()*100))))


def plot_modal_time_exposure(series:pd.DataFrame,max_traces:int=8)->go.Figure:
    fig=go.Figure()
    if not series.empty:
        for key,block in list(series.groupby(['Labware','Harmonic','Reference_Hz'],sort=False))[:max_traces]:
            name=f'{key[0]} | H{key[1]} {key[2]:g}Hz'
            fig.add_trace(go.Scattergl(x=block.Time,y=block.Screening_ratio,
                  name=name,mode='lines',line=dict(width=1.6)))
        fig.add_hline(y=1,line_dash='dash',line_color='#dc2626',
                      annotation_text='Ratio 1 (model comparison only)')
    fig.update_xaxes(title_text='Logger time; overlapping FFT windows')
    fig.update_yaxes(title_text='Short-time band RMS / reference threshold (unverified)')
    return _layout(fig,'Modal exposure over time — screening only, NOT cell-pattern prediction',height=590)


def plot_raw_waveform(result:dict,axis:str='Vector',duration_s:float=2,offset_s:float=0)->go.Figure:
    d=result['selected'];fs=result['quality']['fs_est_Hz']
    begin=d.Time.iloc[0]+pd.Timedelta(seconds=float(offset_s))
    end=begin+pd.Timedelta(seconds=float(duration_s))
    block=d.loc[d.Time.between(begin,end)]
    if len(block)>15000:
        block=block.iloc[::max(1,len(block)//15000)]
    fig=go.Figure()
    for name in ('X','Y','Z') if axis=='Vector' else (axis,):
        fig.add_trace(go.Scattergl(x=block.Time,y=block[f'{name}_g']*G0,
                        name=name,mode='lines',line=dict(color=COLORS[name],width=1)))
    fig.update_xaxes(title_text='Local timestamp (2-second raw waveform, adjustable)')
    fig.update_yaxes(title_text='Raw acceleration incl. gravity (m/s²)')
    return _layout(fig,'Raw acceleration waveform — static gravity remains',height=480)

def export_report(result:dict,destination:str|Path,*,figures:Mapping[str,go.Figure]|None=None)->Path:
    dest=Path(destination);dest.mkdir(parents=True,exist_ok=True)
    for label,key in [('rms_time_bins','time_metrics'),('five_minute_summary','time_sections'),
                      ('welch_psd','spectrum'),('fft_snapshot','fft_snapshot'),
                      ('spectral_peaks','peak_table'),('labware_screening','labware_matches')]:
        result[key].to_csv(dest/f'{label}.csv',index=False)
    if 'modal_exposure' in result and result['modal_exposure'] is not None:
        result['modal_exposure'].to_csv(dest/'modal_exposure_time.csv',index=False)
        modal_time_summary(result['modal_exposure']).to_csv(dest/'modal_exposure_summary.csv',index=False)
    qa=result['quality'].copy();qa['settings']=result['settings'];qa['source_urls']=SOURCE_URLS
    (dest/'quality_and_limitations.json').write_text(json.dumps(qa,indent=2,default=str),encoding='utf-8')
    for label,fig in (figures or {}).items():
        safe=re.sub(r'[^A-Za-z0-9_-]','_',label)
        fig.write_html(dest/f'{safe}.html',include_plotlyjs='cdn',full_html=True)
    return dest


def cli()->None:
    p=argparse.ArgumentParser(description='VSE16g three-axis vibration / labware exposure screening')
    p.add_argument('--input',required=True,help='raw VSE tab-delimited .xls or .csv')
    p.add_argument('--reference',help='user-provided labware frequency/amplitude TXT, kept private')
    p.add_argument('--output',default='vibration_results')
    p.add_argument('--start-time',help='YYYY-MM-DD HH:MM:SS.ffffff')
    p.add_argument('--end-time',help='YYYY-MM-DD HH:MM:SS.ffffff')
    p.add_argument('--start-s',type=float)
    p.add_argument('--end-s',type=float)
    p.add_argument('--start-sample',type=int)
    p.add_argument('--end-sample',type=int)
    p.add_argument('--labware',default='All')
    p.add_argument('--axis',choices=['X','Y','Z','Vector'],default='Vector')
    p.add_argument('--welch-s',type=float,default=20.48)
    p.add_argument('--band-halfwidth-Hz',type=float,default=.5)
    args=p.parse_args()
    log=read_vse(args.input)
    reference=read_labware_reference(args.reference) if args.reference else None
    result=analyze(log,reference,start_time=args.start_time,end_time=args.end_time,
                   start_s=args.start_s,end_s=args.end_s,
                   start_sample=args.start_sample,end_sample=args.end_sample,
                   labware=args.labware,axis=args.axis,welch_s=args.welch_s,
                   band_halfwidth_Hz=args.band_halfwidth_Hz)
    heat=spectrogram(result['selected'],log.quality['fs_est_Hz'],axis=args.axis)
    exposure=modal_time_exposure(result['selected'],log.quality['fs_est_Hz'],result['labware_matches'])
    result['modal_exposure']=exposure
    figures={
        '01_time_rms':plot_time_rms(result),
        '02_transients':plot_time_shocks(result),
        '03_frequency_psd':plot_psd(result,reference,axis=args.axis),
        '04_fft_snapshot':plot_fft(result,axis=args.axis),
        '05_time_frequency_spectrogram':plot_spectrogram(heat),
        '06_time_section_comparison':plot_time_sections(result),
        '07_labware_mode_screen':plot_mode_ranking(result),
        '08_raw_waveform':plot_raw_waveform(result,axis=args.axis),
        '09_modal_screening_over_time':plot_modal_time_exposure(exposure)}
    output=export_report(result,args.output,figures=figures)
    print(json.dumps({k:result['quality'][k] for k in ('selected_samples','selected_duration_s','fs_est_Hz',
          'sampling_gaps_gt_1_5_period','median_vector_rms_m_s2','p95_vector_rms_m_s2')},indent=2))
    print(f'Generated results in {output.resolve()}')

if __name__=='__main__':cli()