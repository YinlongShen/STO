# Training Report: exact

## Configuration

- RHC: False | steps: 100 | segment horizon: 100 | segments: 1 | epochs/segment: 250
- Target: sinusoid amp=0.06 cycles=0.25 phase=0.0 x_span=0.1
- Nodes: 101 | free DOFs: 194

## Metrics

- Final loss: 4.232005e-04
- Final RMSE: 2.057184e-02
- Final endpoint error: 3.495668e-02
- Mean / max path error: 1.925859e-02 / 3.495668e-02
- Best training loss: 4.232017e-04
- Final gradient norm: 1.543464e-03
- Backward time total / mean / median: 81.830883s / 327.324ms / 188.528ms

## Files

- Raw per-epoch timing: `tracking_timing.csv`
- Per-query STO states: `sto_query_report.csv`
- Paper report timeseries: `training_report.csv`
- Machine-readable summary: `training_summary.json`
