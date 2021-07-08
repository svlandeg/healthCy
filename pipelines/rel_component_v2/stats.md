
# Annotation v1
> Input Tensor: [Word vector, Word vector, Sent vector, dep, pos, dist, dep_dist, is_entity]
E    #       LOSS RELAT...  REL_MICRO_P  REL_MICRO_R  REL_MICRO_F  SCORE
 32    2500         121.24        68.48        69.34        68.91    0.17
✔ F-Score 0.840 , Recall 0.778, Precision 0.913

258   11000         271.25        66.54        72.18        69.24    0.18
✔ F-Score 0.852 , Recall 0.852, Precision 0.852




# Annotation v2 <- WHACK
> Input Tensor: [Word vector, Word vector, Sent vector, dep, pos, dist, dep_dist, is_entity]
E    #       LOSS RELAT...  REL_MICRO_P  REL_MICRO_R  REL_MICRO_F  SCORE
28    1500          35.22        77.63        49.17        60.20    0.14
✔ F-Score 0.621 , Recall 0.562, Precision 0.692











> Input Tensor: [Word vector, Word vector, Sentence Vector] 
E    #       LOSS RELAT...  REL_MICRO_P  REL_MICRO_R  REL_MICRO_F  SCORE
109    4500         186.19        57.76        58.47        58.11    0.15



















----------- OLD --------------

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



36    2500          66.29        66.12        67.23        66.67    0.17 -> Without masking
 5    1000          38.55        69.94        67.68        68.79    0.17

66    1500          61.25        52.94        37.50        43.90    0.10 -> New annotations (160 annotations)
28    1500          57.27        60.87        32.71        42.55    0.10 -> New annotations (350 annotations) -> Conclusion: doesn't work rip

