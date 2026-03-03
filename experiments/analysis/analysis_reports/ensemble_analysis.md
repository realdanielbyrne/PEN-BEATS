  Loaded Yearly: (835, 23000)
  Loaded Quarterly: (866, 24000)
  Loaded Monthly: (2794, 48000)
  Loaded Weekly: (2597, 359)

Total ensemble results: 128

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Yearly -- Top 10 Ensemble OWA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #   Config                       Pass           Category             OWA    sMAPE    MASE
  1   Coif2WaveletV3               activeG_fcast  novel_basis       0.7986   13.450   3.076
  2   NBEATS-G                     activeG_fcast  paper_baseline    0.8000   13.461   3.085
  3   NBEATS-G                     baseline       paper_baseline    0.8020   13.468   3.099
  4   Generic+DB3WaveletV3         activeG_fcast  novel_mixed       0.8046   13.515   3.108
  5   Generic+DB3WaveletV3         baseline       novel_mixed       0.8056   13.530   3.113
  6   NBEATS-I                     baseline       paper_baseline    0.8058   13.549   3.110
  7   NBEATS-I                     activeG_fcast  paper_baseline    0.8058   13.549   3.110
  8   DB4WaveletV3                 activeG_fcast  novel_basis       0.8069   13.575   3.112
  9   Trend+Coif2WaveletV3         activeG_fcast  novel_mixed       0.8070   13.580   3.112
  10  AutoEncoder                  baseline       novel_ae          0.8080   13.539   3.129

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Quarterly -- Top 10 Ensemble OWA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #   Config                       Pass           Category             OWA    sMAPE    MASE
  1   Trend+Coif2WaveletV3         baseline       novel_mixed       0.8742   10.013   1.151
  2   Trend+HaarWaveletV3          baseline       novel_mixed       0.8756   10.022   1.153
  3   Trend+Coif2WaveletV3         activeG_fcast  novel_mixed       0.8766   10.009   1.158
  4   Trend+HaarWaveletV3          activeG_fcast  novel_mixed       0.8785   10.035   1.159
  5   Trend+DB3WaveletV3           baseline       novel_mixed       0.8807   10.036   1.165
  6   Trend+DB3WaveletV3           activeG_fcast  novel_mixed       0.8815   10.071   1.163
  7   NBEATS-I+GenericAE           baseline       novel_mixed       0.8836   10.059   1.171
  8   NBEATS-I+G                   baseline       paper_baseline    0.8849   10.083   1.171
  9   NBEATS-I+BottleneckGeneric   baseline       novel_mixed       0.8868   10.096   1.175
  10  NBEATS-I+GenericAE           activeG_fcast  novel_mixed       0.8888   10.119   1.177

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Monthly -- Top 10 Ensemble OWA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #   Config                       Pass           Category             OWA    sMAPE    MASE
  1   Trend+DB3WaveletV3           baseline       novel_mixed       0.8824   12.963   0.921
  2   Trend+DB3WaveletV3           activeG_fcast  novel_mixed       0.8887   13.062   0.927
  3   NBEATS-I+GenericAE           activeG_fcast  novel_mixed       0.8890   13.016   0.931
  4   Trend+Coif2WaveletV3         activeG_fcast  novel_mixed       0.8903   13.107   0.927
  5   Trend+HaarWaveletV3          baseline       novel_mixed       0.8929   13.160   0.929
  6   AutoEncoder                  baseline       novel_ae          0.8940   13.057   0.939
  7   NBEATS-I+GenericAE           baseline       novel_mixed       0.8947   13.171   0.932
  8   NBEATS-I+BottleneckGeneric   baseline       novel_mixed       0.8956   13.217   0.930
  9   Trend+HaarWaveletV3          activeG_fcast  novel_mixed       0.8963   13.206   0.932
  10  NBEATS-G                     baseline       paper_baseline    0.8967   13.090   0.942

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Weekly -- Top 10 Ensemble OWA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #   Config                       Pass           Category             OWA    sMAPE    MASE
  1   NBEATS-I                     activeG_fcast  paper_baseline    0.7628    6.808   2.173
  2   NBEATS-I                     baseline       paper_baseline    0.7628    6.808   2.173
  3   Trend+Coif2WaveletV3         activeG_fcast  novel_mixed       0.7693    6.933   2.171
  4   Trend+HaarWaveletV3          baseline       novel_mixed       0.7730    7.023   2.164
  5   Trend+Coif2WaveletV3         baseline       novel_mixed       0.7802    6.928   2.233
  6   Trend+DB3WaveletV3           baseline       novel_mixed       0.7814    6.974   2.226
  7   Trend+HaarWaveletV3          activeG_fcast  novel_mixed       0.7964    7.203   2.240
  8   NBEATS-I+BottleneckGeneric   baseline       novel_mixed       0.7967    7.184   2.247
  9   NBEATS-I+BottleneckGeneric   activeG_fcast  novel_mixed       0.8090    7.288   2.284
  10  NBEATS-I+G                   activeG_fcast  paper_baseline    0.8162    7.352   2.304

===============================================================================================
  CROSS-PERIOD AVERAGE (best pass per config)
===============================================================================================
  #   Config                       Category           Avg OWA    sMAPE    MASE
  1   Trend+Coif2WaveletV3         novel_mixed         0.8352   10.908   1.840
  2   Trend+HaarWaveletV3          novel_mixed         0.8414   10.997   1.861
  3   Trend+DB3WaveletV3           novel_mixed         0.8416   10.933   1.874
  4   NBEATS-I                     paper_baseline      0.8438   10.995   1.855 <-- PAPER
  5   NBEATS-I+BottleneckGeneric   novel_mixed         0.8489   11.041   1.879
  6   NBEATS-I+GenericAE           novel_mixed         0.8512   11.030   1.890
  7   NBEATS-I+G                   paper_baseline      0.8544   11.092   1.897 <-- PAPER
  8   NBEATS-G                     paper_baseline      0.8576   10.952   1.922 <-- PAPER
  9   Generic+DB3WaveletV3         novel_mixed         0.8597   11.012   1.928
  10  AutoEncoder                  novel_ae            0.8608   11.003   1.937
  11  DB4WaveletV3                 novel_basis         0.8651   11.053   1.932
  12  Coif2WaveletV3               novel_basis         0.8669   10.974   1.957
  13  GenericAE                    novel_ae            0.8707   11.167   1.956
  14  BottleneckGeneric            novel_basis         0.8816   11.234   1.981
  15  BottleneckGenericAE          novel_ae            0.8948   11.452   1.998
  16  NBEATS-I-AE                  novel_mixed         0.9109   11.719   1.990

  BY CATEGORY:
    novel_ae           mean=0.8754  best=0.8608  n=3
    novel_basis        mean=0.8712  best=0.8651  n=3
    novel_mixed        mean=0.8556  best=0.8352  n=7
    paper_baseline     mean=0.8519  best=0.8438  n=3

Done.
