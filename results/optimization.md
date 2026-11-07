# Model optimisation (NYU val, 8 images, 4x, condition hard, tiled CPU inference)

| Variant | Non-zero params | Checkpoint (MB) | CNN latency (ms, CPU) | CNN RMSE, 4× hard (m) |
|---|---|---|---|---|
| FP32 (baseline) | 502,145 | 1.93 | 1409.2 | 0.0704 |
| FP16 weights (storage) | 502,145 | 0.97 | 1519.3 | 0.0704 |
| BF16 autocast inference | 502,145 | 0.97 | 555.6 | 0.0707 |
| 30% L1 pruning (dense storage) | 352,193 | 1.93 | 1361.1 | 0.0854 |
