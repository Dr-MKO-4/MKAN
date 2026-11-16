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

## Sobol

N = 16 points de base, 800 évaluations, second ordre non calculé.
- Évaluations effondrées (fitness < 0,30) : 162 sur 800.
- Indices de premier ordre négatifs : 29 sur 48 ; somme des indices totaux : 2,83.
- Indices totaux dont la borne basse à 95 % dépasse 0 : 2 (mu1, Base_Forget).
- Paramètres dont la modification ne change jamais la sortie dans l'échantillon : Cell_Type, Forget::hybrid::M, Candidate::fkan::Pade_q, Output::hybrid::M, Output::hybrid::K.
- Groupes où le paramètre binaire change de valeur entre A et B (sur 16) : {'Cell_Type': 0, 'Input_Regime': 8}.
- Corrélation de Spearman entre mu* de Morris et S_T de Sobol : 0,47 (p = 0,001).

| Hyperparamètre | S_i | IC 95 % | S_Ti | IC 95 % | Part active |
|---|---|---|---|---|---|
| lam | 0,134 | 0,748 | 0,726 | 0,730 | 1,00 |
| hidden_size | -0,023 | 0,490 | 0,375 | 0,454 | 1,00 |
| batch_size | 0,095 | 0,424 | 0,359 | 0,591 | 1,00 |
| lr | -0,159 | 0,417 | 0,288 | 0,294 | 1,00 |
| Base_Candidate | -0,108 | 0,400 | 0,287 | 0,528 | 1,00 |
| W | -0,196 | 0,356 | 0,248 | 0,314 | 1,00 |
| Base_Input | -0,118 | 0,266 | 0,129 | 0,165 | 1,00 |
| mu1 | 0,071 | 0,311 | 0,082 | 0,045 | 1,00 |
| Base_Forget | 0,056 | 0,344 | 0,076 | 0,045 | 1,00 |
| Forget::fasterkan::h_bandwidth | -0,062 | 0,373 | 0,050 | 0,344 | 0,25 |
| Input_Regime | -0,055 | 0,171 | 0,042 | 0,046 | 1,00 |
| mu2 | -0,040 | 0,145 | 0,027 | 0,035 | 1,00 |
| Base_Output | -0,028 | 0,236 | 0,026 | 0,071 | 0,50 |
| Forget::fasterkan::N_centers | -0,036 | 0,295 | 0,025 | 0,193 | 0,25 |
| Forget::fastkan::N_centers | -0,014 | 0,199 | 0,012 | 0,094 | 0,19 |
| Output::relukan::Grid_G | 0,012 | 0,219 | 0,012 | 0,206 | 0,13 |
| Output::efficientkan::Grid_G | -0,014 | 0,221 | 0,007 | 0,109 | 0,12 |
| Forget::fastkan::h_bandwidth | -0,010 | 0,158 | 0,007 | 0,056 | 0,19 |
| Input::hybrid::M | -0,003 | 0,220 | 0,006 | 0,085 | 0,12 |
| Candidate::fourier::K | -0,025 | 0,193 | 0,006 | 0,043 | 0,19 |
| Forget::relukan::Grid_G | -0,020 | 0,153 | 0,006 | 0,054 | 0,25 |
| Input::fastkan::N_centers | -0,010 | 0,152 | 0,006 | 0,085 | 0,19 |
| Input::hybrid::K | -0,008 | 0,149 | 0,004 | 0,055 | 0,12 |
| Input::fasterkan::h_bandwidth | -0,007 | 0,135 | 0,003 | 0,032 | 0,19 |
| Input::fasterkan::N_centers | 0,002 | 0,151 | 0,003 | 0,022 | 0,19 |
| Input::efficientkan::Spline_Degree_k | 0,000 | 0,058 | 0,003 | 0,038 | 0,19 |
| Forget::efficientkan::Grid_G | 0,013 | 0,147 | 0,003 | 0,027 | 0,19 |
| Input::relukan::Grid_G | 0,009 | 0,075 | 0,002 | 0,020 | 0,25 |
| Output::fasterkan::h_bandwidth | 0,009 | 0,146 | 0,002 | 0,028 | 0,12 |
| Output::fasterkan::N_centers | 0,008 | 0,127 | 0,001 | 0,021 | 0,12 |
| Candidate::hybrid::K | -0,005 | 0,117 | 0,001 | 0,010 | 0,12 |
| Forget::chebyshev::Degree_n | 0,002 | 0,081 | 0,001 | 0,028 | 0,07 |
| Candidate::chebyshev::Degree_n | -0,001 | 0,028 | 0,000 | 0,003 | 0,19 |
| Candidate::hybrid::M | -0,002 | 0,035 | 0,000 | 0,005 | 0,12 |
| Candidate::wavkan::M_wavelets | -0,002 | 0,029 | 0,000 | 0,003 | 0,18 |
| Input::fastkan::h_bandwidth | 0,002 | 0,018 | 0,000 | 0,001 | 0,19 |
| Candidate::wavkan::Mother_Wavelet | -0,002 | 0,021 | 0,000 | 0,001 | 0,18 |
| Candidate::relukan::Grid_G | -0,001 | 0,009 | 0,000 | 0,001 | 0,19 |
| Output::efficientkan::Spline_Degree_k | -0,001 | 0,017 | 0,000 | 0,001 | 0,12 |
| Forget::hybrid::K | -0,000 | 0,015 | 0,000 | 0,001 | 0,06 |
| Forget::efficientkan::Spline_Degree_k | 0,002 | 0,017 | 0,000 | 0,000 | 0,19 |
| Input::efficientkan::Grid_G | -0,001 | 0,005 | 0,000 | 0,000 | 0,19 |
| Input::chebyshev::Degree_n | -0,000 | 0,003 | 0,000 | 0,000 | 0,07 |
| Cell_Type | 0,000 | 0,000 | 0,000 | 0,000 | 1,00 |
| Forget::hybrid::M | 0,000 | 0,000 | 0,000 | 0,000 | 0,06 |
| Candidate::fkan::Pade_q | 0,000 | 0,000 | 0,000 | 0,000 | 0,12 |
| Output::hybrid::M | 0,000 | 0,000 | 0,000 | 0,000 | 0,06 |
| Output::hybrid::K | 0,000 | 0,000 | 0,000 | 0,000 | 0,06 |
