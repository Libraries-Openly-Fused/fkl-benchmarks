### cvGS: resize + cvtColor + multiply + subtract + divide + split (microseconds per call; median of 3 rounds; the largest spread of the 3 rounds of a cell in the last column)

| host flags | B = 2 | B = 12 | B = 42 | B = 142 | slope B 2..42 (ns per crop) | slope B 2..142 | constant (us) | largest spread % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| o3ndebug | 0.042 | 0.197 | 0.695 | 2.161 | 16.3 | 15.1 | 0.011 | 16 |
| o2ndebug | 0.052 | 0.214 | 0.608 | 1.802 | 13.9 | 12.5 | 0.027 | 2 |
| o1ndebug | 0.039 | 0.221 | 0.643 | 1.986 | 15.1 | 13.9 | 0.011 | 28 |
| o0ndebug | 0.306 | 0.961 | 2.600 | 8.113 | 57.4 | 55.8 | 0.195 | 1 |
| o3debugassert | 0.041 | 0.196 | 0.701 | 2.173 | 16.5 | 15.2 | 0.010 | 17 |
| emptybuildtype | 0.295 | 0.942 | 2.650 | 7.993 | 58.9 | 55.0 | 0.185 | 0 |

### FastNPP: resize + swap + MulC + SubC + DivC + CopyBatch (microseconds per call; median of 3 rounds; the largest spread of the 3 rounds of a cell in the last column)

| host flags | B = 2 | B = 12 | B = 42 | B = 142 | slope B 2..42 (ns per crop) | slope B 2..142 | constant (us) | largest spread % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| o3ndebug | 0.071 | 0.637 | 0.947 | 2.859 | 21.9 | 19.9 | 0.031 | 2 |
| o2ndebug | 0.077 | 0.407 | 1.076 | 3.295 | 25.0 | 23.0 | 0.031 | 2 |
| o1ndebug | 0.069 | 0.408 | 1.049 | 3.268 | 24.5 | 22.8 | 0.024 | 2 |
| o0ndebug | 0.411 | 1.724 | 5.163 | 16.472 | 118.8 | 114.7 | 0.181 | 6 |
| o3debugassert | 0.075 | 0.643 | 0.953 | 2.888 | 21.9 | 20.1 | 0.035 | 12 |
| emptybuildtype | 0.411 | 1.734 | 5.140 | 16.572 | 118.2 | 115.4 | 0.180 | 1 |

### cvGS: the read descriptor only (microseconds per call; median of 3 rounds; the largest spread of the 3 rounds of a cell in the last column)

| host flags | B = 2 | B = 12 | B = 42 | B = 142 | slope B 2..42 (ns per crop) | slope B 2..142 | constant (us) | largest spread % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| o3ndebug | 0.040 | 0.195 | 0.636 | 1.965 | 14.9 | 13.8 | 0.012 | 8 |
| o2ndebug | 0.047 | 0.214 | 0.596 | 1.843 | 13.7 | 12.8 | 0.022 | 3 |
| o1ndebug | 0.035 | 0.216 | 0.627 | 1.948 | 14.8 | 13.7 | 0.007 | 1 |
| o0ndebug | 0.208 | 0.855 | 2.461 | 7.984 | 56.3 | 55.5 | 0.097 | 1 |
| o3debugassert | 0.040 | 0.194 | 0.642 | 1.974 | 15.0 | 13.8 | 0.013 | 9 |
| emptybuildtype | 0.209 | 0.841 | 2.547 | 7.871 | 58.5 | 54.7 | 0.099 | 4 |

### FastNPP: the read descriptor only (microseconds per call; median of 3 rounds; the largest spread of the 3 rounds of a cell in the last column)

| host flags | B = 2 | B = 12 | B = 42 | B = 142 | slope B 2..42 (ns per crop) | slope B 2..142 | constant (us) | largest spread % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| o3ndebug | 0.048 | 0.266 | 0.762 | 2.273 | 17.8 | 15.9 | 0.017 | 4 |
| o2ndebug | 0.061 | 0.321 | 0.877 | 2.654 | 20.4 | 18.5 | 0.024 | 3 |
| o1ndebug | 0.046 | 0.290 | 0.799 | 2.438 | 18.8 | 17.1 | 0.012 | 1 |
| o0ndebug | 0.284 | 1.286 | 4.089 | 12.978 | 95.1 | 90.7 | 0.103 | 2 |
| o3debugassert | 0.050 | 0.272 | 0.761 | 2.266 | 17.8 | 15.8 | 0.018 | 3 |
| emptybuildtype | 0.289 | 1.311 | 4.056 | 13.114 | 94.2 | 91.6 | 0.105 | 1 |

