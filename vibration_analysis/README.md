# VSE-16g Incubator Hotel Vibration & Labware Pattern Screening

A *separate module* inside the Cell Incubator Chamber Temperature/RH Analytics repository. Reads raw three-axis Vibration Sentry E-16g logger data from tab-separated **text files saved with `.xls`**, `.csv`, or genuine Excel. Supports large datasets (tested with 2,796,160 records sampled at 800 Hz) and private labware model tables of harmonic frequency, predicted spacing, and allowable acceleration.

> **Research/screening tool, NOT validated cell-pattern prediction or regulatory pass/fail.** The accelerometer was mounted on the hotel; the well plate and culture medium response, mechanical transmission and amplitude-threshold definitions have not been verified. Avoid biological claims without direct verification.

## Interactive notebook

[Open VSE vibration analytics in Google Colab](https://colab.research.google.com/github/leotirto/Cell-Incubator-Chamber-Temperature-RH-Analytics/blob/main/vibration_analysis/VSE16g_Vibration_Labware_Analytics.ipynb)

1. Open the notebook in Google Colab and select **Runtime → Run all**. After installing packages, the native **Choose files** dialog opens automatically.
2. Select the VSE recording (`.xls`, `.xlsx` or `.csv`) and optionally **Frequency Affect to Cell Pattern.txt** in the **same chooser**. When the upload finishes, the dashboard displays their names and **loads the raw recording automatically**. No Google Drive mount or typed file path is necessary. To replace inputs, rerun the upload code cell above the dashboard. Keep your experimental inputs private; do not commit them to public GitHub.
3. **TIME WINDOW — always visible:** set either timestamp From/To (device-local), elapsed start/end seconds (zero = unused), or original sample indexes (zero = unused). Filters intersect if more than one method is set. Presets: Full recording, Skip first 5 minutes, First 5 minutes, Last 10 minutes.
4. Adjust Welch window/resolution, analysis axis, min/max Hz, time-bin length, modal band tolerance, and selected labware. Click **Analyze selected window**.
5. Inspect **Overview & QA, Vibration vs time, Welch PSD + FFT, Time-frequency heatmap, Labware modes, Time exposure**. Export the processed CSV and Plotly HTML charts to ZIP using the Export button.
6. If widget callbacks fail, use **RUN_DIRECT_ANALYSIS = True** in the direct-analysis fallback cell. It uses the uploaded file selections and time-window controls, with no manual path entry.

**Uploading a large file:** A VSE recording may be more than 150 MB. Let Colab finish transferring it before analyzing. The files are stored in the temporary Colab session at `/content/vse_private_uploads`, which is cleared if the runtime resets. Re-upload files to restore the analysis. A reference TXT is optional but required for the labware-mode screening charts.

The interactive plots include individual XYZ / RSS acceleration RMS trends, shock peak timelines, 5-minute bar comparisons, a short raw waveform, Welch PSD, a representative FFT amplitude spectrum, the spectrogram heatmap, prominent spectral peaks, labware-mode screening ranks, and short-time modal ratio occupancy.

## Local command line

```bash
pip install -r vibration_analysis/requirements.txt
python vibration_analysis/vse_vibration_core.py \
  --input "/private/Vibrosentry-Record_g-...xls" \
  --reference "/private/Frequency Affect to Cell Pattern.txt" \
  --start-s 300 --output "/private/results"
```

Other options: `--start-time`, `--end-time`, `--start-sample`, `--end-sample`, `--axis {X,Y,Z,Vector}`, `--labware`, `--welch-s`, and `--band-halfwidth-Hz`.

### Output files

- `rms_time_bins.csv` — filtered time-domain X/Y/Z/vector dynamic RMS and peaks
- `five_minute_summary.csv` — activity drift, transient episodes
- `welch_psd.csv` — one-sided PSD in `(m/s²)²/Hz` and vector summed PSD
- `fft_snapshot.csv` — Hann-coherent-gain corrected *peak* amplitude from one centered 30-second snapshot
- `spectral_peaks.csv` — significant PSD elevations above local frequency floor
- `labware_screening.csv` — reference mode, predicted spacing, reference threshold, integrated-band RMS and **non-validated** ratios
- `modal_exposure_time.csv` — time-varying band RMS / model threshold for top frequencies
- `modal_exposure_summary.csv` — descriptive fraction of overlapping FFT windows with screening ratio > 1
- `quality_and_limitations.json` — sample rate/gap/quality checks, preprocessing parameters, scientific cautions
- Interactive Plotly HTML charts (when generated in the notebook or CLI)

## Physics and signal processing

### Measurement and sample-rate integrity

Raw acceleration is **specific force** in logger `g` units; multiply by `9.80665` to obtain m/s². The raw Z mean near ~1 g reflects gravity/orientation. Do **not** use the DC magnitude as mechanical oscillatory excitation. Check the observed median timestamp interval before any frequency conversion:

`f_s = 1 / median(Δt)`; `f_Nyquist = f_s / 2`.

For 800 Hz, `f_Nyquist=400 Hz`, covering all 1.2–295 Hz reference modes, but sensor amplitude/frequency response, aliasing and physical installation must still be verified. The Vibration Sentry manual notes that the sensor does not include an anti-aliasing filter; higher frequency excitations may fold into this band. The sensor clock tolerance can shift measured frequency by several percent if not calibrated against a reference source.

All samples retain their original measurement values. Filtering for **time RMS** applies a 3rd-order Butterworth forward-backward **0.5 Hz high-pass** to remove gravity and slow inclination. The Welch frequency-domain analysis instead removes the local per-segment mean. No smoothing, filling or interpolation alters the spectral input. Timestamp gaps are counted; the PSD uses the **largest uninterrupted segment** and reports its coverage percentage. No interpolation is allowed across missing samples.

### Welch PSD, FFT, and time localization

For acceleration `a(t)` (m/s²), the one-sided Welch power spectral density is `S_aa(f)` in `(m/s²)²/Hz`; the energy in a band around a reference modal frequency is

`a_rms,band = sqrt( Σ_{f in band} S_aa(f) Δf )`.

For the three axes the **energy** is summed: `S_vector = S_X + S_Y + S_Z`, not the three signed signals. A narrow band for mode `f_i` is `f_i ± max(0.5 Hz, 1% × f_i, Δf)` by default (user-adjustable minimum). Broadening the band includes more noise energy; do not mistake it for a single coherent mode.

The engine also estimates `tonal_excess_RMS` by subtracting a locally estimated PSD floor before band integration. This is exploratory and can over/under-estimate the actual tonal component. Spectral peaks are ranked by local excess in decibels. Snapshot FFT uses Hann coherent-gain compensation, so it reports approximate **sinusoidal peak** amplitude, not PSD-band RMS. A frequency–time spectrogram shows whether a mode is persistent or transient. Modal exposure uses overlapping Hann windows, so its count and ratio occupancy are *descriptive*.

### Hydrodynamic and mechanical plausibility

Labware frequency matching at the shelf does **not** prove standing-wave attachment patterns. Relevant physics includes vessel resonances; shell/plate mode shapes; nonrigid labware; the tray-to-plate coupling; liquid depth `h`; fluid viscosity; surface tension `σ`; fluid density `ρ`; the plate natural frequencies; attachment time; and the specific cell type. For ideal free-surface gravity-capillary waves:

`ω² = (g k + (σ/ρ) k³) tanh(k h)`.

Parametrically excited Faraday waves may occur at **half the driving frequency**, depending on forcing and fluid boundary conditions. The supplied model harmonic frequencies are used **as written** and are not automatically halved/doubled. Such transformations would be separate hypotheses requiring validation.

The supplied TXT contains modeled `Spacing_mm`, `F_limit_pN`, `Threshold_m_s2`, and **4-hour stabilization windows**. It does not provide evidence that the thresholds are sensor-level RMS acceleration, peak excitation at the well, or experimentally validated culture limits. The notebook **must not** declare safe/unsafe or conclude that patterns will appear based only on the calculated ratios. Many table threshold values are below the broadband noise scale of this sensor at 800 Hz (manufacturer typical E-16g 4.5 mg RMS on X/Y, 8.9 mg on Z), making calibration and controlled runs especially important.

### Required next experiment for causal conclusions

Measure (i) simultaneously at the hotel, (ii) at the actual well tray/plate bottom, and (iii) if feasible by video of fluid motion and attachment patterns. Repeat with empty vs loaded hotel, fan on/off, robot movement, and controlled shaker sweeps near the identified frequency families. Vary fill depth. Keep cell type, coating and attachment conditions consistent. Run at least as long as the source model stabilization window (4 h) and image plated cells to associate spatial spacing with excitation. Model or measure a transmissibility function `H(f)=a_plate(f)/a_hotel(f)` before comparing measured acceleration to a plate-level threshold.

## Repository safety

Only analysis code, **synthetic tests**, and docs should be public. Keep raw VSE measurements, private frequency/amplitude labware model and generated plots/exports in Colab/private Drive. Input data are never modified by the program.

## References

- [Vibration Sentry E-16g product specifications](https://convergenceinstruments.com/product/vibration-meter-datalogger-vibration-sentry-e16g/)
- [Manufacturer E-series data sheet (800 Hz noise table)](https://convergenceinstruments.com/pdf/vibration_sentry_e_datasheet.pdf)
- [Vibration Sentry E-series manual (sampling / anti-aliasing)](https://www.eruido.org/eruidoorgwww/assets/files/vibration_sentry_e_usersmanual.pdf)
- [Thermo Fisher: Vibration patterns in cell culture vessels](https://documents.thermofisher.com/TFS-Assets/BID/Technical-Notes/vibration-patterns-cell-culture-vessels-technical-note.pdf)
- [Review: The waves that make the pattern (acoustic manipulation)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8094912/)