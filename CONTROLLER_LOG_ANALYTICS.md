# Incubator environmental + PID controller log analytics

This module analyzes a **different CSV format** from the earlier Govee sensor exports. It extends (does not replace) the existing chamber analytics notebook.

**Open the [controller-log notebook in Google Colab](https://colab.research.google.com/github/leotirto/Cell-Incubator-Chamber-Temperature-RH-Analytics/blob/main/Incubator_Controller_Recovery_Analytics.ipynb).**

## Quick start (Colab)

1. Open the notebook and run all cells. The notebook installs its dependencies and fetches the Python engine from this GitHub repository.
2. In the **Data source** panel, use `Google Drive / local folder` and set the folder containing your CSVs, or choose `Upload CSV(s) in Colab`.
3. Click **Load / reload CSVs**, then select your controller log from the dropdown.
4. Filter with inclusive CSV rows, for example `2604` through `3842`, and/or enter device timestamps. Header is row 1; first actual data row is row 2.
5. Enter **analysis references** (37.0 +/- 0.5 degrees C, 90 +/- 1% RH are examples, **not verified firmware setpoints**). Adjust thresholds and event hold length as needed.
6. Click **Analyze selected window**. View environmental line plots, PID command traces, paired response/control plots, min/mean/max bars, controller occupancy bars, excursion/recovery tables, data-quality reports, correlation and lag plots, and the processed readings.
7. **Batch compare all loaded files** computes independent per-file summaries. **Export selected results ZIP** downloads CSV tables and self-contained interactive Plotly HTML charts.

## Logger columns and assumptions

| Logged channel | Interpretation | Default units |
| --- | --- | --- |
| `T Steam` | Steam-associated temperature | Degrees Celsius assumed, confirm device units |
| `T Chamber` | Chamber temperature | Degrees Celsius assumed |
| `T Bot`, `T Door`, `T Port` | Component/location temperatures | Degrees Celsius assumed |
| `Ctrl Steam`, `Ctrl Chamber`, `Ctrl Bot`, `Ctrl Door`, `Ctrl Port` | Signed temperature-controller commands | As logged; [-1, 1] only checked, not normalized |
| `RH`, `Ctrl RH` | Relative humidity and signed controller command | % RH for RH; signed raw command for control |
| `CO2`, `Ctrl CO2` | CO2 measurement and signed controller command | **Raw logged unit**; not automatically ppm or % |
| `STC31` | Secondary sensor signal; label is not sufficient to infer exact meaning | **Raw logged unit** |

A CSV whose *first column header is a timestamp* is supported (the uploaded controller-log example used this unusual header). Trailing entirely empty `Unnamed` columns are removed. All parsed readings get a `SourceRow` value pointing back to the original CSV record position. Malformed timestamps are counted and omitted from time-based analysis; duplicates and missing channels are reported.

## Accuracy, preprocessing and feature engineering

- **Data integrity:** preserves the original CSV, identifies invalid timestamps, duplicate/out-of-order timestamps, missing/non-numeric readings, RH values outside 0–100%, suspicious **RH=0** samples, gaps, and controller values outside [-1, 1].
- **RH=0:** these are flagged and **retained** by default. If you confirm they indicate logger/sensor dropout, choose **Mask RH=0**: RH is excluded from numerical summaries, but original readings are preserved in the exported `RH_Original` field. Values outside 0–100% RH are always flagged and masked for physical derivations.
- **Derived variables:** elapsed time, temperature/RH/CO2 rates of change (per minute), optional rolling means for visualization, chamber-air dew point and absolute humidity (assuming the RH measurement represents air at the logged `T Chamber`), and a range across component temperatures (this is **not** equivalent to chamber-air spatial uniformity).
- **Discontinuity handling:** derivative computations and time-weighted metrics do not bridge timestamp gaps longer than the configured maximum. Time resampling does **not** interpolate readings or feed artificial values into summary statistics.
- **Thresholds:** first temperature/RH crossing, first simultaneous crossing, and earliest observed sustained hold. For each hold, the report distinguishes the **start** and the **confirmation time** after sufficient observed samples.
- **Excursions:** out-of-band episode starts, last outside sample, observed outside duration, worst observation timestamp/value, whether a sufficiently sustained in-band recovery follows, and measured recovery time. Events and holds are measured from logged observations only; values between observations are unknown.
- **Control analytics:** time-weighted fractions of positive / negative / neutral commands, signed mean, absolute mean, positive/negative saturation percentages and sign changes. Positive/negative is the reported sign; do **not** infer that positive equals physical heating or negative equals cooling without the actuator mapping.
- **Correlations:** exploratory Pearson correlation matrix and lagged controller/measurement association. PID feedback creates confounding: these plots do **not** estimate a physically validated response delay or establish causality.

## Run as a Python batch job

From the repository root:

```bash
python -m pip install -r requirements.txt
python incubator_controller_analytics.py --folder "./your_private_controller_CSV_folder" --output "./controller_analysis_output" \
  --start-row 2604 --end-row 3842
```

To mask RH=0 only after confirming a sensor dropout:

```bash
python incubator_controller_analytics.py --folder "./your_private_controller_CSV_folder" --mask-rh-zeros
```

Run `python incubator_controller_analytics.py --help` for time-window options, batch modes and reference values. Plots are saved as standalone interactive HTML files along with `processed_readings.csv`, `statistics.csv`, `pid_statistics.csv`, `threshold_reach.csv`, `excursions.csv`, and `quality_report.json`.

The notebook and batch engine read **all compatible CSV controller logs** in a folder; incompatible files (such as Govee-only CSVs) are reported and skipped. The existing Govee notebook remains the appropriate tool for those files.

## Units, assumptions and safety

- Confirm the CO2 measurement unit and the physical interpretation of `STC31` with firmware/device documentation before converting or labeling units.
- Calibrate the controller sign interpretation and any PID-to-actuator transfer function externally; the file alone does not supply it.
- The default example references are not evidence of firmware setpoints or acceptance limits.
- This is an engineering research/diagnostics tool, **not** a safety controller, alarm monitor, or validated environmental compliance test.
- **Never upload raw company measurements, credentials or exported HTML reports into this public GitHub repository**. They remain local or in your authorized Drive folder.

## Tests

Synthetic regression tests exercise the unusual timestamp-as-header format, row/time slicing, RH dropout handling, derived features, controller metrics, event duration/hold semantics, lag plots, and exports:

```bash
python -m pip install -r requirements.txt pytest
python -m pytest -q tests
```

Test records are synthesized; the supplied experimental CSV has **not** been committed to GitHub.