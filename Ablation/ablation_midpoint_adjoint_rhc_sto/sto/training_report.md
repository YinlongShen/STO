# Training Report: sto

## Configuration

- RHC: True | steps: 100 | segment horizon: 20 | segments: 5 | epochs/segment: 250
- Target: sinusoid amp=0.06 cycles=0.25 phase=0.0 x_span=0.1
- Nodes: 101 | free DOFs: 194

## Metrics

- Final loss: 2.504591e-06
- Final RMSE: 1.582590e-03
- Final endpoint error: 4.716150e-04
- Mean / max path error: 1.193558e-03 / 4.224263e-03
- Best training loss: 1.537097e-07
- Final gradient norm: 3.587756e-06
- Backward time total / mean / median: 23.851248s / 19.081ms / 14.677ms
- STO cache hit rate: 99.00% (24750 / 25000)
- STO exact fallbacks: 250
- STO kappa recomputes: 7887
- Active STO cache estimate: 6.092 MiB

## Files

- Raw per-epoch timing: `tracking_timing.csv`
- Per-query STO states: `sto_query_report.csv`
- Paper report timeseries: `training_report.csv`
- Machine-readable summary: `training_summary.json`
