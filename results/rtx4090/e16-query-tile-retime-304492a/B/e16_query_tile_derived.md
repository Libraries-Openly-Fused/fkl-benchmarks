| S | excess BQ32 / BQ64 / BQ128 (ms, median of runs) | ratio 128/64 (runs) | median | reading | share of the conversion BQ64 / BQ128 | spread of int8_bq64 / raw_bq64 (%) |
|---:|---|---|---:|---|---|---|
| 512 | 0.9199 / 0.1441 / 0.0846 | 0.565 / 0.727 / 0.573 | 0.5728 | proportional to the number of query tiles (S < 2048: rule applies at S >= 2048) | 0.6214 / 0.5068 | 0.2 / 3.9 |
| 1024 | 1.7900 / 0.2448 / 0.1363 | 0.571 / 0.485 / 0.557 | 0.5569 | proportional to the number of query tiles (S < 2048: rule applies at S >= 2048) | 0.5329 / 0.4606 | 0.2 / 3.9 |
| 2048 | 3.5680 / 0.4997 / 0.2432 | 0.456 / 0.501 / 0.484 | 0.4840 | proportional to the number of query tiles | 0.4307 / 0.4626 | 0.0 / 2.4 |
| 4096 | 7.4563 / 0.9626 / 0.4486 | 0.475 / 0.459 / 0.468 | 0.4682 | proportional to the number of query tiles | 0.3925 / 0.4752 | 0.0 / 0.5 |
| 8192 | 15.0909 / 2.0177 / 0.8838 | 0.441 / 0.438 / 0.438 | 0.4380 | proportional to the number of query tiles | 0.3314 / 0.4844 | 0.2 / 0.6 |
