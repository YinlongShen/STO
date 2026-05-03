# Training Report: sto

## Configuration

- RHC: False | steps: 100 | segment horizon: 100 | segments: 1 | epochs/segment: 250
- Target: sinusoid amp=0.06 cycles=0.25 phase=0.0 x_span=0.1
- Nodes: 101 | free DOFs: 194

## Metrics

- Final loss: 5.076175e-04
- Final RMSE: 2.253037e-02
- Final endpoint error: 3.876447e-02
- Mean / max path error: 2.119541e-02 / 3.876447e-02
- Best training loss: 5.076159e-04
- Final gradient norm: 1.663119e-03
- Backward time total / mean / median: 27.389480s / 109.558ms / 78.166ms
- STO cache hit rate: 96.42% (24104 / 25000)
- STO exact fallbacks: 896
- STO kappa recomputes: 7144
- Active STO cache estimate: 29.300 MiB

## Files

- Raw per-epoch timing: `tracking_timing.csv`
- Per-query STO states: `sto_query_report.csv`
- Paper report timeseries: `training_report.csv`
- Machine-readable summary: `training_summary.json`
