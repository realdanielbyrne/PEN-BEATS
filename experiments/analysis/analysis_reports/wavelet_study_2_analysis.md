
================================================================================
  1. OVERALL SUMMARY
================================================================================
Total runs:       72
Unique configs:   24
Runs per config:  [3]
Wavelet families: ['Coif2', 'DB3', 'Haar']
basis_dim values: [np.int64(4), np.int64(6), np.int64(15), np.int64(30)]
thetas_dim vals:  [np.int64(3), np.int64(5)]

── Global metric statistics ──
                count    mean    std     min     25%     50%     75%     max
smape         72.0000 13.6141 0.1776 13.2934 13.4947 13.5925 13.7099 14.3209
mase          72.0000  3.1424 0.0737  3.0050  3.0887  3.1336  3.1773  3.4593
owa           72.0000  0.8119 0.0146  0.7848  0.8021  0.8105  0.8190  0.8734
best_val_loss 72.0000 13.5967 0.0615 13.4687 13.5455 13.5982 13.6450 13.7188


================================================================================
  2. MARGINAL EFFECTS (mean ± std across seeds)
================================================================================

── Wavelet Family ──
            smape_summary     mase_summary      owa_summary best_val_loss_summary  n_runs
wavelet                                                                                  
Coif2    13.5693 ± 0.1494  3.1284 ± 0.0623  0.8088 ± 0.0124      13.5818 ± 0.0640      24
DB3      13.6358 ± 0.1344  3.1524 ± 0.0557  0.8138 ± 0.0110      13.5986 ± 0.0614      24
Haar     13.6370 ± 0.2321  3.1464 ± 0.0972  0.8131 ± 0.0193      13.6099 ± 0.0583      24

── Basis Dimension ──
              smape_summary     mase_summary      owa_summary best_val_loss_summary  n_runs
basis_dim                                                                                  
4          13.6712 ± 0.2024  3.1644 ± 0.0886  0.8164 ± 0.0173      13.6029 ± 0.0637      18
6          13.6006 ± 0.2079  3.1341 ± 0.0795  0.8105 ± 0.0163      13.5979 ± 0.0617      18
15         13.6214 ± 0.1339  3.1484 ± 0.0593  0.8129 ± 0.0115      13.5915 ± 0.0685      18
30         13.5631 ± 0.1521  3.1226 ± 0.0629  0.8079 ± 0.0125      13.5947 ± 0.0564      18

── Trend Thetas Dim ──
               smape_summary     mase_summary      owa_summary best_val_loss_summary  n_runs
thetas_dim                                                                                  
3           13.6034 ± 0.1807  3.1399 ± 0.0765  0.8113 ± 0.0151      13.6107 ± 0.0652      36
5           13.6247 ± 0.1764  3.1448 ± 0.0718  0.8125 ± 0.0144      13.5828 ± 0.0550      36


================================================================================
  3. TWO-WAY INTERACTION TABLES (mean sMAPE)
================================================================================

── Wavelet × Basis Dim ──
basis_dim      4       6       15      30
wavelet                                  
Coif2     13.6612 13.4900 13.5569 13.5691
DB3       13.6162 13.6456 13.7014 13.5803
Haar      13.7363 13.6662 13.6058 13.5399


── Wavelet × Thetas Dim ──
thetas_dim       3       5
wavelet                   
Coif2      13.5287 13.6099
DB3        13.5997 13.6720
Haar       13.6818 13.5923


── Basis Dim × Thetas Dim ──
thetas_dim       3       5
basis_dim                 
4          13.7024 13.6400
6          13.5715 13.6296
15         13.5686 13.6741
30         13.5711 13.5551


── Wavelet × Basis Dim (mean OWA) ──
basis_dim     4      6      15     30
wavelet                              
Coif2     0.8158 0.8020 0.8077 0.8096
DB3       0.8110 0.8146 0.8197 0.8100
Haar      0.8225 0.8147 0.8112 0.8040


================================================================================
  4. CONFIG RANKINGS
================================================================================

── Top 10 configs by mean sMAPE ──
                        wavelet  basis_dim  bd_label  thetas_dim  smape_mean  smape_std  mase_mean  owa_mean  owa_std  val_loss_mean  epochs_mean  n_params  runs
config_name                                                                                                                                                      
Coif2_bd6_eq_fcast_td3    Coif2          6  eq_fcast           3     13.4097     0.1013     3.0531    0.7944   0.0084        13.6522      20.6667   5080335     3
Haar_bd30_eq_bcast_td5     Haar         30  eq_bcast           5     13.5089     0.0452     3.0793    0.8008   0.0030        13.5878      24.3333   5144345     3
Haar_bd15_lt_bcast_td3     Haar         15  lt_bcast           3     13.5167     0.0993     3.1058    0.8043   0.0070        13.6147      22.3333   5103375     3
Haar_bd4_lt_fcast_td5      Haar          4  lt_fcast           5     13.5258     0.1338     3.1077    0.8048   0.0115        13.6162      21.3333   5072665     3
Coif2_bd15_lt_bcast_td5   Coif2         15  lt_bcast           5     13.5509     0.1256     3.1188    0.8070   0.0103        13.5589      21.6667   5105945     3
Coif2_bd15_lt_bcast_td3   Coif2         15  lt_bcast           3     13.5628     0.0807     3.1276    0.8085   0.0090        13.5867      20.3333   5103375     3
Coif2_bd30_eq_bcast_td3   Coif2         30  eq_bcast           3     13.5637     0.0235     3.1403    0.8101   0.0023        13.5662      19.0000   5141775     3
Coif2_bd6_eq_fcast_td5    Coif2          6  eq_fcast           5     13.5702     0.1887     3.1346    0.8096   0.0150        13.5270      27.0000   5082905     3
Haar_bd30_eq_bcast_td3     Haar         30  eq_bcast           3     13.5708     0.1647     3.1157    0.8072   0.0107        13.6136      22.3333   5141775     3
Coif2_bd30_eq_bcast_td5   Coif2         30  eq_bcast           5     13.5746     0.2492     3.1295    0.8091   0.0206        13.6180      22.0000   5144345     3

── Bottom 5 configs by mean sMAPE ──
                       wavelet  basis_dim  bd_label  thetas_dim  smape_mean  smape_std  mase_mean  owa_mean  owa_std  val_loss_mean  epochs_mean  n_params  runs
config_name                                                                                                                                                     
Haar_bd6_eq_fcast_td3     Haar          6  eq_fcast           3     13.6929     0.2647     3.1658    0.8173   0.0219        13.6083      23.3333   5080335     3
Haar_bd15_lt_bcast_td5    Haar         15  lt_bcast           5     13.6949     0.1661     3.1723    0.8181   0.0159        13.5676      25.6667   5105945     3
Coif2_bd4_lt_fcast_td5   Coif2          4  lt_fcast           5     13.7438     0.1927     3.1901    0.8219   0.0155        13.5668      30.6667   5072665     3
DB3_bd15_lt_bcast_td5      DB3         15  lt_bcast           5     13.7765     0.0576     3.2232    0.8270   0.0043        13.6295      20.0000   5105945     3
Haar_bd4_lt_fcast_td3     Haar          4  lt_fcast           3     13.9468     0.3279     3.2866    0.8402   0.0291        13.6675      22.6667   5070095     3

── Best single run ──
  Config:    Coif2_bd6_eq_fcast_td3
  Run/Seed:  2/44
  sMAPE:     13.2934
  MASE:      3.0050
  OWA:       0.7848


================================================================================
  5. STATISTICAL TESTS
================================================================================
  Kruskal-Wallis on sMAPE by wavelet     : H=   3.267  p=0.1952  ns
  Kruskal-Wallis on sMAPE by basis_dim   : H=   3.755  p=0.2891  ns
  Kruskal-Wallis on sMAPE by thetas_dim  : H=   0.457  p=0.4992  ns

── Pairwise Mann-Whitney U (basis_dim on sMAPE) ──
  bd= 4 vs bd= 6: U= 192.0  p=0.3506  ns
  bd= 4 vs bd=15: U= 170.0  p=0.8124  ns
  bd= 4 vs bd=30: U= 231.0  p=0.0302  *
  bd= 6 vs bd=15: U= 143.0  p=0.5583  ns
  bd= 6 vs bd=30: U= 169.0  p=0.8371  ns
  bd=15 vs bd=30: U= 203.0  p=0.2001  ns

── One-way ANOVA on sMAPE by basis_dim ──
  F=1.169  p=0.3282


================================================================================
  6. CONVERGENCE DIAGNOSTICS
================================================================================
── Epochs trained by basis_dim ──
             mean    std  min  max
basis_dim                         
4         23.3333 5.0176   13   36
6         23.1111 6.5878   15   39
15        22.3333 4.7651   16   33
30        21.6667 3.6942   15   27

── Epochs trained by wavelet ──
           mean    std  min  max
wavelet                         
Coif2   23.1250 5.4399   15   36
DB3     21.0417 4.2475   13   33
Haar    23.6667 5.2226   15   39

── Training time (seconds) by basis_dim ──
             mean     std
basis_dim                
4         59.1956 11.2605
6         58.6883 16.4363
15        55.4328 11.8812
30        53.3283  9.1756

── Parameter count by basis_dim ──
                  mean      min      max
basis_dim                               
4         5071380.0000  5070095  5072665
6         5081620.0000  5080335  5082905
15        5104660.0000  5103375  5105945
30        5143060.0000  5141775  5144345

── Loss ratio (final_val / best_val) by basis_dim ──
            mean    std    max
basis_dim                     
4         1.0284 0.0216 1.0909
6         1.0446 0.0312 1.1054
15        1.0303 0.0197 1.0798
30        1.0397 0.0217 1.0841

── Val loss curve analysis ──
  Curve lengths: mean=22.6, min=13, max=39
  Improvement (first - last val_loss): mean=1.607, std=2.150


================================================================================
  7. EFFECT SIZES & PRACTICAL SIGNIFICANCE
================================================================================
  sMAPE range across configs: 0.5371
  Best config mean sMAPE:     13.4097 (Coif2_bd6_eq_fcast_td3)
  Worst config mean sMAPE:    13.9468 (Haar_bd4_lt_fcast_td3)

── Eta-squared (proportion of variance explained) ──
  wavelet     : η² = 0.0322  (3.2%)
  basis_dim   : η² = 0.0490  (4.9%)
  thetas_dim  : η² = 0.0036  (0.4%)

── sMAPE by basis_dim semantic label ──
            mean    std  count
bd_label                      
eq_bcast 13.5631 0.1521     18
eq_fcast 13.6006 0.2079     18
lt_bcast 13.6214 0.1339     18
lt_fcast 13.6712 0.2024     18


================================================================================
  ANALYSIS COMPLETE
================================================================================
  Data: /home/realdanielbyrne/GitHub/N-BEATS-Lightning/experiments/results/m4/wavelet_study_2_basis_dim_results.csv
  Rows: 72

