# Same-day control at N = 2,000: c8fcbe8 and 9e20bd4e alternating (scripts/e3_control_pairs.py)

Compiles in the order they ran, by pair: (c8fcbe8, 9e20bd4e), (9e20bd4e, c8fcbe8), (c8fcbe8, 9e20bd4e). spread = the larger (max - min) / median of the two trees; 'exceeds' = |change| > spread; flag = |change| > 5 %; paired = (new - old) / old within each pair.

| quantity | c8fcbe8 (in order) | 9e20bd4e (in order) | median c8fcbe8 | median 9e20bd4e | spread % | change % | exceeds? | > 5 %? | paired changes % | median paired % | repeats: c8fcbe8 -> 9e20bd4e median | repeats' change % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nvcc wall (s) | 95.528 / 96.014 / 95.467 | 96.908 / 96.315 / 96.599 | 95.528 | 96.599 | 0.61 | +1.12 | yes | no | +1.44 / +0.31 / +1.19 | +1.19 | 95.177 -> 96.686 | +1.59 |
| peak memory (MiB) | 4054 / 4053 / 4054 | 4073 / 4073 / 4073 | 4054 | 4073 | 0.02 | +0.47 | yes | no | +0.47 / +0.49 / +0.47 | +0.47 | 4060 -> 4073 | +0.32 |
| cudafe++ (s) | 0.866 / 0.864 / 0.862 | 0.871 / 0.869 / 0.866 | 0.864 | 0.869 | 0.58 | +0.58 | yes | no | +0.58 / +0.58 / +0.46 | +0.58 | 0.851 -> 0.87 | +2.23 |
| cicc (s) | 65.743 / 66.107 / 65.644 | 66.803 / 66.291 / 66.558 | 65.743 | 66.558 | 0.77 | +1.24 | yes | no | +1.61 / +0.28 / +1.39 | +1.39 | 65.495 -> 66.63 | +1.73 |
| ptxas (s) | 1.666 / 1.659 / 1.668 | 1.669 / 1.663 / 1.664 | 1.666 | 1.664 | 0.54 | -0.12 | no | no | +0.18 / +0.24 / -0.24 | +0.18 | 1.659 -> 1.671 | +0.72 |
| host g++ (s) | 26.753 / 26.885 / 26.796 | 27.067 / 26.993 / 27.014 | 26.796 | 27.014 | 0.49 | +0.81 | yes | no | +1.17 / +0.40 / +0.81 | +0.81 | 26.647 -> 26.941 | +1.10 |

Code quantities over all 6 compiles: registers equal (17), spill_stores equal (0), spill_loads equal (0), stack_bytes equal (0), cmem0_bytes equal (14840), sass_instructions equal (7136), text_bytes equal (114176), cubin_bytes equal (50521760)
