# Analyse de sensibilité de la recherche réexécutée

Morris : r = 10 trajectoires, 4 niveaux, 490 évaluations, 48 hyperparamètres encodés.
- Évaluations effondrées (fitness < 0,30) : 170 sur 490 (34,7 %).
- Paramètres dont l'intervalle à 95 % de mu* contient 0 : 33 sur 48.
- Paramètres marqués « sans influence » par le seuil de l'outil : 20, tous conditionnels.
- Rapport sigma/mu* médian : 2,29 (effets non linéaires ou en interaction).

| Hyperparamètre | mu* | IC 95 % | sigma | mu |
|---|---|---|---|---|
| lam | 0,273 | 0,124 | 0,223 | -0,252 |
| hidden_size | 0,263 | 0,134 | 0,355 | 0,028 |
| Candidate::relukan::Grid_G | 0,227 | 0,237 | 0,436 | -0,073 |
| W | 0,213 | 0,131 | 0,299 | -0,116 |
| Input_Regime | 0,206 | 0,120 | 0,288 | -0,000 |
| lr | 0,205 | 0,082 | 0,233 | 0,105 |
| Base_Candidate | 0,204 | 0,113 | 0,288 | 0,045 |
| mu1 | 0,187 | 0,136 | 0,231 | -0,186 |
| batch_size | 0,144 | 0,083 | 0,188 | -0,065 |
| Cell_Type | 0,131 | 0,057 | 0,170 | -0,023 |
| mu2 | 0,124 | 0,118 | 0,232 | -0,055 |
| Forget::fasterkan::h_bandwidth | 0,115 | 0,109 | 0,173 | 0,112 |
| Candidate::chebyshev::Degree_n | 0,112 | 0,098 | 0,188 | -0,051 |
| Input::chebyshev::Degree_n | 0,112 | 0,160 | 0,260 | 0,033 |
| Base_Input | 0,065 | 0,040 | 0,097 | 0,000 |
| Base_Forget | 0,065 | 0,031 | 0,060 | -0,057 |
| Forget::fasterkan::N_centers | 0,063 | 0,072 | 0,138 | -0,039 |
| Input::fasterkan::N_centers | 0,059 | 0,056 | 0,107 | 0,047 |
| Output::fasterkan::N_centers | 0,040 | 0,081 | 0,126 | -0,040 |
| Candidate::hybrid::K | 0,028 | 0,038 | 0,066 | -0,027 |
| Base_Output | 0,024 | 0,029 | 0,055 | -0,005 |
| Input::fasterkan::h_bandwidth | 0,021 | 0,032 | 0,055 | -0,019 |
| Candidate::hybrid::M | 0,014 | 0,021 | 0,035 | -0,009 |
| Input::hybrid::K | 0,012 | 0,017 | 0,028 | 0,012 |
| Forget::efficientkan::Grid_G | 0,011 | 0,020 | 0,034 | -0,011 |
| Forget::hybrid::M | 0,007 | 0,012 | 0,022 | -0,007 |
| Candidate::fkan::Pade_q | 0,006 | 0,007 | 0,013 | -0,006 |
| Forget::chebyshev::Degree_n | 0,005 | 0,006 | 0,011 | -0,005 |
| Output::efficientkan::Spline_Degree_k | 0,004 | 0,007 | 0,012 | -0,003 |
| Input::efficientkan::Spline_Degree_k | 0,001 | 0,003 | 0,004 | 0,001 |
| Output::hybrid::K | 0,000 | 0,001 | 0,002 | -0,000 |
| Input::fastkan::h_bandwidth | 0,000 | 0,000 | 0,000 | -0,000 |
| Forget::fastkan::N_centers | 0,000 | 0,000 | 0,000 | -0,000 |
| Input::fastkan::N_centers | 0,000 | 0,000 | 0,000 | -0,000 |
| Candidate::wavkan::M_wavelets | 0,000 | 0,000 | 0,000 | -0,000 |
| Output::relukan::Grid_G | 0,000 | 0,000 | 0,000 | -0,000 |
| Input::relukan::Grid_G | 0,000 | 0,000 | 0,000 | -0,000 |
| Forget::hybrid::K | 0,000 | 0,000 | 0,000 | 0,000 |
| Forget::fastkan::h_bandwidth | 0,000 | 0,000 | 0,000 | 0,000 |
| Forget::efficientkan::Spline_Degree_k | 0,000 | 0,000 | 0,000 | 0,000 |
| Forget::relukan::Grid_G | 0,000 | 0,000 | 0,000 | 0,000 |
| Input::hybrid::M | 0,000 | 0,000 | 0,000 | 0,000 |
| Input::efficientkan::Grid_G | 0,000 | 0,000 | 0,000 | 0,000 |
| Candidate::wavkan::Mother_Wavelet | 0,000 | 0,000 | 0,000 | 0,000 |
| Candidate::fourier::K | 0,000 | 0,000 | 0,000 | 0,000 |
| Output::hybrid::M | 0,000 | 0,000 | 0,000 | 0,000 |
| Output::fasterkan::h_bandwidth | 0,000 | 0,000 | 0,000 | 0,000 |
| Output::efficientkan::Grid_G | 0,000 | 0,000 | 0,000 | 0,000 |
