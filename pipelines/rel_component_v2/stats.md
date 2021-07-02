E    #       LOSS RELAT...  REL_MICRO_P  REL_MICRO_R  REL_MICRO_F  SCORE
775   10000         142.76        65.00        36.52        46.76    0.11 -> With masking + 3 Lin
278    4500         205.74        52.54        18.56        27.43    0.06 -> Without masking + 3 Lin
276    4500         141.03        63.81        33.67        44.08    0.10 -> With masking + 2 Linear Layers
594    8000         139.12        63.89        34.67        44.95    0.11 -> With masking + 3 Linear Layers

123    5000          65.78        66.46        63.25        64.81    0.16 -> After some adjustments + more annotations

126    5000          67.26        63.43        61.90        62.65    0.16 -> Averaging word vectors + masking
198    7000          63.79        64.69        64.69        64.69    0.16 -> Concatenate word vectors without masking
163    6000          73.91        67.37        64.91        66.12    0.16 -> Concatenate word vectors with masking entities with np.ones()



145    5500          72.43        63.44        62.75        63.09    0.16 -> Change nI / nO (nI -> 256 -> 516 -> nO)