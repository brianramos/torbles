| configuration | 13 SSB 1T | 13 SSB 2T | M1 1T | M2 1T | M3 1T | M1-M3 1T | M1-M3 2T | answers |
|---|---|---|---|---|---|---|---|---|
| DuckDB (in memory) | 478.0 | 310.8 | 163.7 | 181.9 | 68.4 | 414.1 | 228.3 | reference |
| 0.2.0 before the bridges | 208.2 | 116.7 | 49.7 | 211.7 | 244.0 | 505.4 | 269.5 | all equal DuckDB |
| + vectorized transpose (bridges off) | 212.2 | 117.5 | 48.1 | 219.1 | 257.3 | 524.5 | 268.5 | all equal DuckDB |
| + factor through the key | 211.6 | 119.7 | 48.5 | 212.4 | 115.7 | 376.6 | 199.8 | all equal DuckDB |
| + splitting tree, direct slots | 240.4 | 126.2 | 64.4 | 245.1 | 282.5 | 592.0 | 299.3 | all equal DuckDB |
| + direct slots only | 206.4 | 118.5 | 46.7 | 221.2 | 246.3 | 514.2 | 267.3 | all equal DuckDB |
| + row kernel (cost-chosen) | 214.7 | 115.9 | 48.8 | 75.6 | 62.0 | 186.5 | 93.9 | all equal DuckDB |
| + Möbius (cost-chosen) | 209.5 | 115.0 | 48.8 | 213.3 | 249.3 | 511.4 | 270.9 | all equal DuckDB |
| + factor + row kernel | 209.2 | 122.1 | 49.2 | 72.7 | 61.1 | 183.0 | 95.2 | all equal DuckDB |
| + factor + Möbius | 217.1 | 120.6 | 48.3 | 214.6 | 112.9 | 375.7 | 204.2 | all equal DuckDB |
| + factor + row kernel + Möbius | 213.0 | 120.3 | 48.5 | 70.2 | 64.6 | 183.3 | 93.7 | all equal DuckDB |
| factor + row kernel on every tile | 254.1 | 149.3 | 59.1 | 69.7 | 60.3 | 189.0 | 100.9 | all equal DuckDB |
| factor + Möbius on every tile | 382.3 | 207.4 | 48.9 | 320.9 | 96.9 | 466.7 | 240.1 | all equal DuckDB |
| factor + row kernel, no margin (model as is) | 219.2 | 128.9 | 59.8 | 73.4 | 61.7 | 194.9 | 102.7 | all equal DuckDB |
| defaults: factor + row kernel at margin 0.5 | 213.1 | 114.8 | 48.2 | 71.1 | 59.9 | 179.2 | 96.1 | all equal DuckDB |

| query (1 thread) | DuckDB | 0.2.0 before the bridges | + vectorized transpose (bridges off) | + factor through the key | factor + row kernel, no margin (model as is) | defaults: factor + row kernel at margin 0.5 |
|---|---|---|---|---|---|---|
| Q1.1 | 34.1 | 16.6 | 16.3 | 17.0 | 16.6 | 16.2 |
| Q1.2 | 23.4 | 7.8 | 7.8 | 8.5 | 8.0 | 8.2 |
| Q1.3 | 18.0 | 17.1 | 16.3 | 15.6 | 15.4 | 15.6 |
| Q2.1 | 42.7 | 13.8 | 16.3 | 14.8 | 16.0 | 16.2 |
| Q2.2 | 37.3 | 12.8 | 13.1 | 13.1 | 13.3 | 13.2 |
| Q2.3 | 27.5 | 7.1 | 7.5 | 7.6 | 7.8 | 7.6 |
| Q3.1 | 60.0 | 27.4 | 28.5 | 27.6 | 33.2 | 28.4 |
| Q3.2 | 35.7 | 14.5 | 15.0 | 14.8 | 15.1 | 15.8 |
| Q3.3 | 25.1 | 11.9 | 12.1 | 12.5 | 12.3 | 13.1 |
| Q3.4 | 18.6 | 11.0 | 11.7 | 11.3 | 11.6 | 11.3 |
| Q4.1 | 59.8 | 32.4 | 31.2 | 33.1 | 31.6 | 31.7 |
| Q4.2 | 60.4 | 23.5 | 23.3 | 22.6 | 24.9 | 22.8 |
| Q4.3 | 35.4 | 12.3 | 13.1 | 13.0 | 13.3 | 13.0 |
| M1 | 163.7 | 49.7 | 48.1 | 48.5 | 59.8 | 48.2 |
| M2 | 181.9 | 211.7 | 219.1 | 212.4 | 73.4 | 71.1 |
| M3 | 68.4 | 244.0 | 257.3 | 115.7 | 61.7 | 59.9 |
