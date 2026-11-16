# Cell Incubator Chamber — Temperature & RH Analytics

A multi-sensor temperature and relative humidity analysis tool for incubator environmental testing. Includes an **interactive Google Colab notebook** and a **standalone Python analytics engine**.

> **Data security:** This repository is intended for analysis code and documentation, not raw environmental measurements. Sensor CSVs, locally generated reports and credentials are excluded by `.gitignore`. The repository is public; review files before committing.

## Quick start — Google Colab

[Open the notebook in Colab](https://colab.research.google.com/github/leotirto/Cell-Incubator-Chamber-Temperature-RH-Analytics/blob/main/Chamber_Temperature_RH_Analytics.ipynb)

1. Open the link, select **Runtime → Run all**, and authorize Google Drive when prompted.
2. The default input folder is `/content/drive/MyDrive/Envi Data`. Change it in the dashboard as needed.
3. Select **Load / reload CSVs** and choose sensors.
4. By default, all data rows are used. To analyze a slice, uncheck **Use all rows** and set an inclusive CSV range such as **60–1001** (CSV header = row 1).
5. Adjust date/time limits, temperature/RH thresholds, target bands, and plot options. Click **Analyze selected data**.
6. Review line charts, bar comparisons, QA tables, and engineering statistics; export processed CSVs and interactive HTML reports.

## What the pipeline includes

- Robust parsing for multiple CSV encodings/delimiters and flexible Date/Temperature/RH column names.
- Missing/invalid value and duplicate/out-of-order timestamp reports; questionable values are not silently clipped.
- Original-row, date/time and sensor filters with source row traceability.
- First temperature/RH/both threshold crossings, sustained observed crossings, time-in-band and post-reach statistics.
- Derived dew point, absolute humidity, rate-of-change and moving averages.
- Chamber sensor-to-sensor spread on a time-aligned interval; requires synchronized device clocks.
- Interactive Plotly multi-sensor lines and grouped statistical bars.
- Retained legacy `govee-05` timing exception, configurable in the notebook.

### Measurement integrity

Calculations and statistics use the cleaned observed readings. Resampling, interpolation and smoothing are intended for visualization and remain separate from measurement statistics. Sustained threshold detection is based on sampled observations and excludes large gaps; it cannot prove conditions between readings. RH auto-unit conversion is logged and should be checked for very low humidity values.

## Run the engine on a local folder

```bash
python -m pip install -r requirements.txt
python environment_temperature_rh_core.py --folder "./Envi Data" --start-row 60 --end-row 1001
```

Run `python environment_temperature_rh_core.py --help` for other options.

## Tests

```bash
python -m pip install -r requirements.txt pytest
python -m pytest -q tests
```

CI runs synthetic-data tests on Python 3.11 and 3.12 for pushes and pull requests. **No production or company measurement CSVs are included in the repository.**

## Files

| Path | Purpose |
| --- | --- |
| `Chamber_Temperature_RH_Analytics.ipynb` | Google Colab interactive dashboard |
| `environment_temperature_rh_core.py` | Reusable parsing, processing, analytics and chart builders |
| `tests/test_environment_temperature_rh.py` | Synthetic CSV pipeline smoke test |
| `requirements.txt` | Python libraries |
| `.github/workflows/tests.yml` | Automated testing |
| `.gitignore` | Avoid committing raw data, reports or credentials |

## Limitations

- Test results are from synthetic data; real sensor CSVs were not included for validation.
- Timestamps are interpreted as device-local values by default; synchronize sensors for cross-sensor comparisons.
- This is an engineering analysis tool, not a validated regulatory compliance or alarm-monitoring system.

### Colab CSV loading troubleshooting

The controller notebook now enables Colab's custom widget communication bridge and
provides **native code-cell loaders** for both Google Drive and browser uploads.
After running the notebook's setup and dashboard cells, run **one** native loader
cell in section 2A (Drive **or** Upload), then click **Analyze selected window**.
Switching the input dropdown alone does not open the native upload chooser. If a
button still does not respond, run the **Direct analysis fallback** cell in section 2B.

**Do not use Runtime → Run all for a source-specific session:** it would execute
both alternative loader cells. Execute the setup, your chosen loader, and the
analysis cells instead.

## Incubator controller log / recovery analytics (new)

Analyze incubator firmware controller logs containing `T Steam`, `T Chamber`,
`T Bot`, `T Door`, `T Port`, `RH`, `CO2`, `STC31` and signed
`Ctrl ...` PID commands. This is **separate** from the existing Govee
multi-sensor notebook.

- [Open the interactive controller-log notebook in Google Colab](https://colab.research.google.com/github/leotirto/Cell-Incubator-Chamber-Temperature-RH-Analytics/blob/main/Incubator_Controller_Recovery_Analytics.ipynb)
- [Read the controller-log analytics guide](CONTROLLER_LOG_ANALYTICS.md)
- Reusable Python module: `incubator_controller_analytics.py`
- Synthetic regression tests: `tests/test_incubator_controller_analytics.py`

Select CSV files, inclusive row ranges and/or timestamp windows; inspect
measured values and PID commands, threshold hold times, time-in-band, excursions,
recovery, control-sign occupancy/saturation and correlation/lag. Export processed
CSV tables and interactive Plotly HTML.

**No raw experimental CSV is committed.** CO2/STC31 units and physical
controller directions are not assumed; reference targets are editable examples.

**New — threshold drop recovery:** T Chamber ≥ 36.5°C, RH ≥ 85%, CO₂ ≥ 47000
(raw logged units), all editable in Colab. The **Drop recovery** tab adds
per-event return/confirmation times, timeline markers, grouped recovery bars,
and `drop_recovery.csv` in exports. See [interpretation and limitations](CONTROLLER_LOG_ANALYTICS.md#drop-recovery-at-temperature-rh-and-co₂-thresholds).
