# Training Report: exact

## Configuration

- RHC: True | steps: 100 | segment horizon: 20 | segments: 5 | epochs/segment: 250
- Target: sinusoid amp=0.06 cycles=0.25 phase=0.0 x_span=0.1
- Nodes: 101 | free DOFs: 194

## Metrics

- Final loss: 2.582925e-06
- Final RMSE: 1.607148e-03
- Final endpoint error: 4.660517e-04
- Mean / max path error: 1.213611e-03 / 4.148909e-03
- Best training loss: 1.560359e-07
- Final gradient norm: 3.620078e-06
- Backward time total / mean / median: 85.146406s / 68.117ms / 35.394ms

## Files

- Raw per-epoch timing: `tracking_timing.csv`
- Per-query STO states: `sto_query_report.csv`
- Paper report timeseries: `training_report.csv`
- Machine-readable summary: `training_summary.json`
