# VCEA comparator audit (R1)

All weights from validation only; test used for scoring only. Comparator list fixed in the script header.

## Reproduction check

```
{
  "main6": {
    "VCEA_mean": 0.1776329504309473,
    "M1_mean": 0.07833414717990031,
    "paper_VCEA_mean": 0.1776,
    "paper_M1_mean": 0.0783
  },
  "e7": {
    "VCEA_mean": 0.2601069085708698,
    "M1_mean": 0.11275047448115852,
    "paper_VCEA_mean": 0.26010677259930604,
    "paper_M1_mean": 0.11275046554026742
  }
}
```

## Mean Skill_MSE by model

| Model             |      e7 |    main6 |
|:------------------|--------:|---------:|
| VCEA              |  0.2601 |   0.1776 |
| VCEA_rho0         |  0.2533 |   0.161  |
| BatesGranger      |  0.2497 |   0.159  |
| NNLS_simplex      |  0.2477 |   0.1459 |
| OracleSingle_test |  0.2257 |   0.1229 |
| VCEA_noNeural     |  0.1891 |   0.0908 |
| DampedLS_val      |  0.1689 |   0.0764 |
| GlobalVal         |  0.1589 |   0.1267 |
| SelectBestVal     |  0.1537 |   0.0847 |
| VCEA_noLS         |  0.1474 |   0.1117 |
| Uniform4          |  0.143  |  -1.8055 |
| M1                |  0.1128 |   0.0783 |
| C3_opt            |  0.0861 |   0.0796 |
| Persistence       |  0      |   0      |
| Linear-Seasonal   | -2.0608 | -36.1495 |

## VCEA minus comparator, tile-level (mean over horizons and seeds)

### main6

| Comparator        |   n_tiles |   mean_delta |   median_delta |   wins |   ci95_low |   ci95_high |   wilcoxon_p_two_sided |
|:------------------|----------:|-------------:|---------------:|-------:|-----------:|------------:|-----------------------:|
| BatesGranger      |         6 |       0.0186 |         0.0103 |      4 |    -0.0001 |      0.0467 |                 0.1562 |
| C3_opt            |         6 |       0.0981 |         0.1305 |      5 |     0.0255 |      0.1565 |                 0.0938 |
| DampedLS_val      |         6 |       0.1012 |         0.0811 |      6 |     0.035  |      0.1757 |                 0.0312 |
| GlobalVal         |         6 |       0.0509 |         0.0334 |      6 |     0.013  |      0.0889 |                 0.0312 |
| Linear-Seasonal   |         6 |      36.3271 |         4.2391 |      6 |     2.0707 |     70.7419 |                 0.0312 |
| M1                |         6 |       0.0993 |         0.102  |      6 |     0.042  |      0.158  |                 0.0312 |
| NNLS_simplex      |         6 |       0.0317 |         0.0366 |      4 |     0.0032 |      0.0596 |                 0.1562 |
| OracleSingle_test |         6 |       0.0548 |         0.0805 |      5 |     0.0034 |      0.0975 |                 0.0938 |
| Persistence       |         6 |       0.1776 |         0.2187 |      6 |     0.1024 |      0.2367 |                 0.0312 |
| SelectBestVal     |         6 |       0.0929 |         0.1088 |      5 |     0.0143 |      0.1674 |                 0.0938 |
| Uniform4          |         6 |       1.9831 |         0.1918 |      6 |     0.0969 |      4.0705 |                 0.0312 |
| VCEA_noLS         |         6 |       0.0659 |         0.0897 |      5 |     0.0119 |      0.1123 |                 0.0938 |
| VCEA_noNeural     |         6 |       0.0868 |         0.0659 |      6 |     0.0338 |      0.1458 |                 0.0312 |
| VCEA_rho0         |         6 |       0.0166 |         0.0044 |      5 |    -0.0193 |      0.0646 |                 0.3125 |

- Best simple combiner on main6: **BatesGranger** (mean 0.1590); VCEA - best = +0.0186; pre-stated margin 0.02 -> NOT MET (exploratory; e7 test labels were already seen).

### e7

| Comparator        |   n_tiles |   mean_delta |   median_delta |   wins |   ci95_low |   ci95_high |   wilcoxon_p_two_sided |
|:------------------|----------:|-------------:|---------------:|-------:|-----------:|------------:|-----------------------:|
| BatesGranger      |        14 |       0.0104 |         0.0084 |     10 |    -0.0024 |      0.0253 |                 0.104  |
| C3_opt            |        14 |       0.174  |         0.1334 |     14 |     0.1141 |      0.2513 |                 0.0001 |
| DampedLS_val      |        14 |       0.0912 |         0.0701 |     13 |     0.0399 |      0.1474 |                 0.004  |
| GlobalVal         |        14 |       0.1012 |         0.0853 |     13 |     0.0504 |      0.1724 |                 0.0002 |
| Linear-Seasonal   |        14 |       2.3209 |         1.4749 |     13 |     1.2658 |      3.5699 |                 0.0004 |
| M1                |        14 |       0.1474 |         0.0985 |     14 |     0.0873 |      0.2189 |                 0.0001 |
| NNLS_simplex      |        14 |       0.0125 |         0.0047 |      8 |    -0.0136 |      0.0444 |                 0.5016 |
| OracleSingle_test |        14 |       0.0344 |         0.0425 |     13 |    -0.016  |      0.0697 |                 0.0134 |
| Persistence       |        14 |       0.2601 |         0.2391 |     14 |     0.1873 |      0.351  |                 0.0001 |
| SelectBestVal     |        14 |       0.1065 |         0.0818 |     13 |     0.0605 |      0.1545 |                 0.0004 |
| Uniform4          |        14 |       0.1171 |         0.0655 |     14 |     0.0657 |      0.1794 |                 0.0001 |
| VCEA_noLS         |        14 |       0.1127 |         0.0844 |     13 |     0.0551 |      0.188  |                 0.0002 |
| VCEA_noNeural     |        14 |       0.071  |         0.0618 |     12 |     0.0344 |      0.1097 |                 0.0052 |
| VCEA_rho0         |        14 |       0.0068 |         0.0061 |     14 |     0.0044 |      0.0097 |                 0.0001 |

- Best simple combiner on e7: **BatesGranger** (mean 0.2497); VCEA - best = +0.0104; pre-stated margin 0.02 -> NOT MET (exploratory; e7 test labels were already seen).
