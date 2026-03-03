================================================================================
1. UNIFIED BENCHMARK (experiments/results/m4/unified_benchmark_results.csv)
================================================================================
Total rows: 1479
Periods: ['Daily', 'Monthly', 'Quarterly', 'Weekly', 'Yearly']
Experiments: ['activeG_fcast', 'baseline']
Config names: ['AutoEncoder', 'BottleneckGeneric', 'BottleneckGenericAE', 'Coif2WaveletV3', 'DB4WaveletV3', 'Generic+DB3WaveletV3', 'GenericAE', 'NBEATS-G', 'NBEATS-I', 'NBEATS-I+BottleneckGeneric', 'NBEATS-I+G', 'NBEATS-I+GenericAE', 'NBEATS-I-AE', 'Trend+Coif2WaveletV3', 'Trend+DB3WaveletV3', 'Trend+HaarWaveletV3']

── Yearly (sorted by sMAPE) ──
               config_name  smape_mean  smape_std  owa_mean  runs  epochs_mean
      Trend+Coif2WaveletV3   15.504116   5.154807  0.958153    20        28.45
       Trend+HaarWaveletV3   15.535939   4.011639  0.947630    20        25.65
               NBEATS-I-AE   15.696754   4.478680  0.980088    20        45.80
        Trend+DB3WaveletV3   15.697115   4.968733  0.973620    20        25.05
                  NBEATS-I   16.142098   7.296484  1.026499    20        41.60
NBEATS-I+BottleneckGeneric   16.751023   7.864376  1.060754    20        34.95
                  NBEATS-G   16.906303  12.299307  1.048365    20        37.00
               AutoEncoder   16.997858   9.560402  1.052393    20        46.50
                 GenericAE   17.108151  10.900873  1.043932    20        40.95
        NBEATS-I+GenericAE   17.150781   9.495338  1.073306    20        35.10
      Generic+DB3WaveletV3   17.658263  13.903488  1.108640    20        34.55
                NBEATS-I+G   18.049264  14.870008  1.204783    20        30.40
       BottleneckGenericAE   19.329989  21.107092  1.147328    20        48.10
            Coif2WaveletV3   19.338427  14.943275  1.213417    20        35.25
         BottleneckGeneric   20.144415  14.230570  1.251010    20        43.20
              DB4WaveletV3   21.974649  18.421761  1.325020    20        33.50

── All-period mean sMAPE ──
                            smape_mean  owa_mean  periods
config_name                                              
Trend+Coif2WaveletV3          9.835978  0.903948        5
Trend+HaarWaveletV3           9.889658  0.908337        5
NBEATS-I                     10.075668  0.937728        5
NBEATS-I+BottleneckGeneric   10.289695  0.957002        5
NBEATS-I+GenericAE           10.370227  0.963055        5
AutoEncoder                  10.400663  0.984606        5
GenericAE                    10.493520  0.981858        5
NBEATS-I+G                   10.509104  0.980334        5
NBEATS-I-AE                  10.527058  0.985187        5
BottleneckGenericAE          11.159974  1.035537        5
BottleneckGeneric            11.279128  1.065993        5
NBEATS-G                     11.885705  1.234535        5
Generic+DB3WaveletV3         11.955093  1.245696        5
Coif2WaveletV3               13.063591  1.373626        5
DB4WaveletV3                 13.321800  1.414286        5
Trend+DB3WaveletV3           13.891400  8.551764        5

================================================================================
2. BLOCK BENCHMARK (experiments/results/m4/block_benchmark_results.csv)
================================================================================
Total rows: 420
── Yearly ──
        config_name  smape_mean  smape_std     owa_mean  runs
         NBEATS-I+G   13.530515   0.085532 8.057256e-01     5
   Trend+DB3Wavelet   13.557707   0.134431 8.060708e-01     5
        AutoEncoder   13.560926   0.139843 8.075232e-01     5
BottleneckGenericAE   13.570916   0.134447 8.087486e-01     5
          GenericAE   13.572102   0.094297 8.063332e-01     5
  GenericAEBackcast   13.576056   0.104093 8.099570e-01     5
        NBEATS-I-AE   13.616785   0.073634 8.095104e-01     5
           NBEATS-I   13.669990   0.105968 8.132286e-01     5
           NBEATS-G   13.704468   0.165698 8.198460e-01     5
  BottleneckGeneric   19.889867  13.717216 1.220229e+00     5
       Coif2Wavelet   34.369982  17.897885 2.281837e+00     3
 Generic+DB3Wavelet   63.007737  80.979047 1.344712e+09     5
         DB3Wavelet  148.468544  89.255100 7.813646e+26     3
     Symlet3Wavelet  148.468544  89.255100 7.813646e+26     3
  Trend+HaarWavelet  162.696502  83.413157 1.039037e+22     5
      DB3AltWavelet         NaN        NaN          NaN     0
        HaarWavelet         NaN        NaN          NaN     0

================================================================================
3. WAVELET V3 BENCHMARK (experiments/results/m4/wavelet_v3_benchmark_results.csv)
================================================================================
Total rows: 51
── Yearly ──
         config_name  smape_mean  smape_std  owa_mean  runs
Trend+Coif2WaveletV3   13.451331   0.049538  0.798236     5
  Trend+DB3WaveletV3   13.471825   0.095128  0.800605     5
      Coif2WaveletV3   13.518372   0.033997  0.804401     5
Generic+DB3WaveletV3   13.532373   0.097293  0.804379     5
 Trend+HaarWaveletV3   13.533637   0.086308  0.804961     5
        DB4WaveletV3   13.551952   0.048079  0.806191     5
        DB2WaveletV3   13.560534   0.144006  0.807435     5
        DB3WaveletV3   18.883828  13.062655  1.111751     6
    Symlet3WaveletV3   19.836789  13.975238  1.152146     5
       HaarWaveletV3   19.924285  14.167604  1.181058     5

================================================================================
4. WAVELET SEARCH — ROUND 4 FINAL (experiments/results/m4/wavelet_search_results.csv)
================================================================================
── Round 4 baseline ──
                        smape_mean  smape_std  owa_mean  runs
config_name                                                  
DB3_trmix_s10_d8_o0      13.822196   0.462841  0.827975     5
Coif2_trmix_s10_d8_o0    13.860829   0.185403  0.829739     5
DB4_trmix_s10_d8_o0      13.881237   0.335779  0.833178     5
Haar_trmix_s10_d16_o0    13.886427   0.282841  0.834262     5
Coif3_trmix_s10_d16_o0   13.913178   0.624037  0.833241     5
DB3_trmix_s10_d16_o0     14.024792   0.546785  0.843406     5

── Round 4 active_g=forecast ──
                        smape_mean  smape_std  owa_mean  runs
config_name                                                  
Coif3_trmix_s10_d16_o0   13.856920   0.202906  0.830848     5
DB3_trmix_s10_d16_o0     13.890645   0.321518  0.834541     5
Haar_trmix_s10_d16_o0    14.002484   0.660574  0.837721     5
Coif2_trmix_s10_d8_o0    14.012358   0.455516  0.836872     5
DB3_trmix_s10_d8_o0      14.245871   0.822723  0.858829     5
DB4_trmix_s10_d8_o0      14.299776   0.661739  0.864344     5

================================================================================
5. CROSS-SOURCE COMPARISON — YEARLY BEST CONFIGS
================================================================================
           source          best_config   smape    owa  runs
    unified_bench Trend+Coif2WaveletV3 15.5041 0.9582    20
      block_bench           NBEATS-I+G 13.5305 0.8057     5
       wavelet_v3 Trend+Coif2WaveletV3 13.4513 0.7982     5
wavelet_search_r4  DB3_trmix_s10_d8_o0 13.8222 0.8280     5

================================================================================
6. SEARCH DATA — HYPERPARAMETER MARGINALS (all rounds)
================================================================================

── arch_pattern ──
                         mean        std  count
arch_pattern                                   
trend_mixed         18.169976   6.260144    265
homogeneous         72.310745  28.138076     81
trend_season_mixed  90.585845  48.211551     27

── n_stacks_requested ──
                         mean        std  count
n_stacks_requested                             
10                  24.046137  20.985993    171
20                  33.366226  28.524278     99
30                  55.367824  42.845654    103

── basis_dim ──
                mean        std  count
basis_dim                             
8          31.826192  30.487397    142
16         31.866301  31.521074    133
4          44.494842  36.831128     98

── basis_offset ──
                   mean        std  count
basis_offset                             
0             35.168986  33.009997    373

── wavelet_family ──
                        mean        std  count
wavelet_family                                
DB3WaveletV3       29.869099  30.871834     58
HaarWaveletV3      31.873096  29.243701     45
Coif3WaveletV3     32.546778  36.711034     43
DB10WaveletV3      33.632575  23.162075     34
DB2WaveletV3       34.150099  28.831626     39
Symlet10WaveletV3  36.828211  29.727017     37
DB4WaveletV3       38.462211  38.556724     44
Symlet3WaveletV3   39.448279  29.475801     33
Coif2WaveletV3     42.992186  44.663625     40

================================================================================
7. UNIFIED BENCHMARK — BASELINE vs ACTIVE_G (Yearly)
================================================================================

── baseline ──
                            smape_mean  smape_std  owa_mean  runs  diverged_pct
config_name                                                                    
Trend+HaarWaveletV3          15.295703   2.726056  0.934481    10           0.0
NBEATS-I-AE                  15.696754   4.601406  0.980088    10           0.0
Trend+DB3WaveletV3           15.729453   5.133135  0.975742    10           0.0
Trend+Coif2WaveletV3         15.886176   6.550088  0.995255    10           0.0
NBEATS-I                     16.142098   7.496425  1.026499    10           0.0
NBEATS-I+BottleneckGeneric   16.529267   7.922197  1.032962    10           0.0
NBEATS-I+GenericAE           16.580337   7.766664  1.073860    10           0.0
AutoEncoder                  17.281879  10.808155  1.054225    10           0.0
GenericAE                    18.618303  14.974887  1.125233    10           0.0
NBEATS-G                     19.078560  17.323659  1.202826    10           0.0

── activeG_fcast ──
                      smape_mean  smape_std  owa_mean  runs  diverged_pct
config_name                                                              
NBEATS-G               14.734047   2.959533  0.893905    10           0.0
BottleneckGeneric      15.062889   3.593006  0.918957    10           0.0
Trend+Coif2WaveletV3   15.122056   3.587269  0.921052    10           0.0
BottleneckGenericAE    15.162676   3.530083  0.920913    10           0.0
Generic+DB3WaveletV3   15.448429   4.929360  0.940635    10           0.0
NBEATS-I+G             15.549640   3.768321  0.954716    10           0.0
GenericAE              15.597999   4.641781  0.962632    10           0.0
Trend+DB3WaveletV3     15.664777   5.076254  0.971497    10           0.0
DB4WaveletV3           15.684935   5.057623  0.958924    10           0.0
NBEATS-I-AE            15.696754   4.601406  0.980088    10           0.0

================================================================================
8. CONVERGENCE PATTERNS — STOPPING REASON (Yearly, baseline)
================================================================================
stopping_reason             EARLY_STOPPED  MAX_EPOCHS
config_name                                          
AutoEncoder                             9           1
BottleneckGeneric                       9           1
BottleneckGenericAE                     9           1
Coif2WaveletV3                          9           1
DB4WaveletV3                            9           1
Generic+DB3WaveletV3                    9           1
GenericAE                               9           1
NBEATS-G                                9           1
NBEATS-I                                9           1
NBEATS-I+BottleneckGeneric              9           1
NBEATS-I+G                              9           1
NBEATS-I+GenericAE                      9           1
NBEATS-I-AE                             9           1
Trend+Coif2WaveletV3                    9           1
Trend+DB3WaveletV3                      9           1
Trend+HaarWaveletV3                     9           1

Mean epochs by config:
                            mean        std
config_name                                
Trend+DB3WaveletV3          23.1   8.372574
Trend+HaarWaveletV3         24.5  10.047111
Trend+Coif2WaveletV3        30.2  11.113555
Generic+DB3WaveletV3        32.4  13.484147
NBEATS-I+GenericAE          32.5  12.149531
NBEATS-I+G                  33.0  13.727507
DB4WaveletV3                35.0  15.846486
Coif2WaveletV3              35.3  15.158789
NBEATS-G                    41.5  18.001543
NBEATS-I                    41.6  18.697593
NBEATS-I+BottleneckGeneric  42.3  17.657545
BottleneckGeneric           44.1  18.681244
GenericAE                   44.7  18.342725
NBEATS-I-AE                 45.8  23.078610
AutoEncoder                 50.3  25.002444
BottleneckGenericAE         51.1  23.909319

================================================================================
9. MULTI-PERIOD CONSISTENCY — Top unified bench configs
================================================================================
period                       Daily  Monthly  Quarterly  Weekly  Yearly  overall_mean
config_name                                                                         
Trend+HaarWaveletV3          2.848   13.429     10.188   7.292  15.296         9.811
Trend+Coif2WaveletV3         2.839   13.475     10.146   7.192  15.886         9.907
NBEATS-I                     3.073   13.669     10.428   7.067  16.142        10.076
NBEATS-I+BottleneckGeneric   3.014   13.443     10.315   7.669  16.529        10.194
NBEATS-I+GenericAE           3.115   13.405     10.245   7.978  16.580        10.265
AutoEncoder                  3.173   13.502     10.320   7.543  17.282        10.364
NBEATS-I-AE                  3.173   14.628     10.860   8.277  15.697        10.527
GenericAE                    3.119   13.954     10.350   7.709  18.618        10.750
NBEATS-I+G                   2.837   13.550     10.290   7.684  20.549        10.982
BottleneckGenericAE          3.167   13.937     10.627   8.711  23.497        11.988
BottleneckGeneric            3.010   13.650     10.528   9.715  25.226        12.426
Trend+DB3WaveletV3          22.658   13.285     10.214   7.282  15.729        13.834
