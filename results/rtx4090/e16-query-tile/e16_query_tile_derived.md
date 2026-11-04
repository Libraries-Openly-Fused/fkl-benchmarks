| S | excess BQ32 / BQ64 / BQ128 (ms, median of runs) | ratio 128/64 (runs) | median | reading | share of the conversion BQ64 / BQ128 | spread of int8_bq64 / raw_bq64 (%) |
|---:|---|---|---:|---|---|---|
| 512 | 0.9192 / 0.1453 / 0.0837 | 0.571 / 0.722 / 0.548 | 0.5708 | proportional to the number of query tiles (S < 2048: rule applies at S >= 2048) | 0.6104 / 0.5077 | 0.3 / 4.5 |
| 1024 | 1.7840 / 0.2406 / 0.1364 | 0.567 / 0.574 / 0.561 | 0.5671 | proportional to the number of query tiles (S < 2048: rule applies at S >= 2048) | 0.5400 / 0.4599 | 0.3 / 0.5 |
| 2048 | 3.5714 / 0.4961 / 0.2397 | 0.481 / 0.483 / 0.484 | 0.4831 | proportional to the number of query tiles | 0.4389 / 0.4607 | 0.2 / 0.7 |
| 4096 | 7.4721 / 0.9679 / 0.4547 | 0.470 / 0.470 / 0.465 | 0.4698 | proportional to the number of query tiles | 0.3973 / 0.4707 | 0.1 / 0.0 |
| 8192 | 15.1627 / 2.0248 / 0.8860 | 0.437 / 0.438 / 0.438 | 0.4376 | proportional to the number of query tiles | 0.3325 / 0.4800 | 0.1 / 0.0 |
