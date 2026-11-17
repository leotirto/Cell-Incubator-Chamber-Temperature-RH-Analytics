import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from environment_temperature_rh_core import *


def test_pipeline():
    with TemporaryDirectory() as tmp:
        p=Path(tmp)
        time=pd.date_range('2026-11-01 08:00:00', periods=80, freq='min')
        t=34+3.5*(1-np.exp(-np.arange(80)/10))
        rh=70+22*(1-np.exp(-np.arange(80)/15))
        a=pd.DataFrame({'Date':time.astype(str),'Temperature':np.round(t,3),
                        'RH':np.round(rh,3)})
        a['Temperature'] = a['Temperature'].astype(object)
        a['RH'] = a['RH'].astype(object)
        a.loc[3,'Temperature']='--'
        a.loc[4,'RH']='120%'
        a.loc[5,'Date']='not_a_date'
        a.loc[6,'Temperature']='36,5 °C'
        a.loc[7,'RH']='86,5%'
        a.to_csv(p/'GOVEE-01.csv',index=False)
        b=pd.DataFrame({'Time':time.astype(str),'Temperature(°C)':np.round(t+0.2,3),
                        'Relative Humidity':np.round((rh+1)/100,5)})
        b.to_csv(p/'govee-05.csv',index=False)
        c=pd.DataFrame({'Date':time.astype(str),'Temperature':np.round(t+0.5,3),'RH':np.round(rh+0.5,3)})
        c.to_csv(p/'MainController.csv',index=False)
        data, qa, errs = load_folder(p)
        assert not errs, errs
        assert len(data)==3
        assert qa.loc[qa.Sensor=='GOVEE-01','Invalid timestamps'].iloc[0] == 1
        assert qa.loc[qa.Sensor=='GOVEE-01','RH out of 0–100%'].iloc[0] == 1
        assert qa.loc[qa.Sensor=='govee-05','RH interpretation'].iloc[0] == 'fraction'
        assert abs(data['GOVEE-01'].loc[data['GOVEE-01'].SourceRow==8,TEMP].iloc[0] - 36.5)<1e-8
        # CSV row selection is inclusive and prior to duplicate grouping.
        filtered=select_window(data['MainController'],row_start=60,row_end=71)
        assert len(filtered)==12 and filtered.SourceRow.min()==60 and filtered.SourceRow.max()==71
        ss, overview, stages, spread=run_analysis(data, list(data),
                row_start=2,row_end=70, temp_threshold=36.5,rh_threshold=85,
                except_sensor_names=('govee-05',))
        assert 'govee-05' in ss
        assert len(overview)==3
        assert overview.loc[overview.Sensor=='govee-05','Temp threshold status'].iloc[0]=='EXEMPT'
        assert np.isnan(overview.loc[overview.Sensor=='govee-05','Time to both min'].iloc[0])
        assert len(stages)==3*4*2
        assert len(spread)>0 and spread.TempSpread_C.notna().any()
        for fig in [build_profile(ss), build_bars(stages), build_bars(stages,variable='Relative humidity (%)'),
                    build_timing_bars(overview), build_spread_figure(spread),build_derived_figure(ss)]:
            assert len(fig.data)>0
        # A failed hold must return no false positive.
        d=ss['GOVEE-01'].iloc[:4]
        assert pd.isna(first_sustained_reach(d, temp_threshold=36.5,
                            rh_threshold=85,stage='both', hold_minutes=10,max_gap_min=5))
        # Backwards-sort or duplicate timestamps never increase record count after prepare.
        dupe=pd.concat([data['MainController'],data['MainController'].iloc[[0]]])
        collapsed=prepare_series(dupe)
        assert len(collapsed)==len(data['MainController'])
        assert collapsed.ReadingsAtTime.max()==2
        # Export test (no fabricated data)
        for n,d in ss.items():
            d.to_csv(p/f'{n}_processed.csv',index=False)
        overview.to_csv(p/'summary.csv',index=False)
        assert (p/'summary.csv').exists()
    print('ALL PIPELINE TESTS PASSED')

if __name__=='__main__': test_pipeline()


def test_own_threshold_post_reach_metrics():
    from environment_temperature_rh_core import own_threshold_statistics, plot_own_threshold_bars
    stamps=pd.date_range("2026-11-17 09:00",periods=8,freq="1min")
    data=pd.DataFrame({DATE:stamps,TEMP:[34,35,36.6,37,36.0,38,37,37],
                       RH:[70,80,83,84,87,90,82,91]})
    result=own_threshold_statistics({"govee-01":data},temp_threshold=36.5,
                                     rh_threshold=85.0,method="first")
    a=result.set_index("Parameter")
    assert int(a.loc["Temperature","N after threshold"])==6
    assert int(a.loc["RH","N after threshold"])==4
    assert a.loc["Temperature","Min"]==36.0
    assert a.loc["RH","Min"]==82
    assert a.loc["Temperature","Reached at"]==stamps[2]
    assert a.loc["RH","Reached at"]==stamps[4]
    assert len(plot_own_threshold_bars(result,"Temperature").data)==3
    never=own_threshold_statistics({"govee-01":data},temp_threshold=100,
                                    rh_threshold=100)
    assert (never["N after threshold"]==0).all()
    assert never["Mean"].isna().all()
