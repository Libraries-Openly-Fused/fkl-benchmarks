| S | excess BQ32 / BQ64 / BQ128 (ms, median of runs) | ratio 128/64 (runs) | median | reading | share of the conversion BQ64 / BQ128 | spread of int8_bq64 / raw_bq64 (%) |
|---:|---|---|---:|---|---|---|
| 512 | 0.9184 / 0.1437 / 0.0836 | 0.582 / 0.726 / 0.570 | 0.5817 | proportional to the number of query tiles (S < 2048: rule applies at S >= 2048) | 0.6212 / 0.5072 | 0.1 / 3.9 |
| 1024 | 1.7889 / 0.2461 / 0.1367 | 0.570 / 0.486 / 0.561 | 0.5613 | proportional to the number of query tiles (S < 2048: rule applies at S >= 2048) | 0.5277 / 0.4577 | 0.2 / 3.5 |
| 2048 | 3.5596 / 0.4921 / 0.2418 | 0.446 / 0.504 / 0.485 | 0.4845 | proportional to the number of query tiles | 0.4355 / 0.4691 | 0.1 / 2.7 |
| 4096 | 7.4494 / 0.9836 / 0.4557 | 0.475 / 0.454 / 0.460 | 0.4597 | proportional to the number of query tiles | 0.3882 / 0.4723 | 0.3 / 0.9 |
| 8192 | 15.1723 / 2.0129 / 0.8742 | 0.436 / 0.438 / 0.429 | 0.4360 | proportional to the number of query tiles | 0.3280 / 0.4848 | 0.3 / 0.1 |
