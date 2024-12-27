# Causality-informed-Generation

.  
├── README.md                    
├── causal_graph : *Contains all causal graphs*  
└── code1 : *Contains source code for generating causal images based on Blender*

GPT-4o-mini
## basic prompt
| **Scene**          | **TPR (%)/Recall (%) ↑** | **FPR (%) ↓** | **SHD (count) ↓** | **Accuracy (%) ↑** | **Precision (%) ↑** | **F1 Score (%) ↑** |
|---------------------|-------------------------:|--------------:|------------------:|-------------------:|--------------------:|-------------------:|
| Spring              |                   92.60 |          3.70 |            0.43   |              95.47 |               92.28 |              91.23 |
| Seesaw              |                   85.00 |          5.48 |            1.75   |              93.00 |               80.19 |              82.52 |
| Magnetic Field      |                   63.33 |         21.43 |            2.2333 |              75.18 |               45.79 |              53.12 |
| Reflection          |                  100.00 |          0.00 |            0.00   |             100.00 |              100.00 |             100.00 |
| Refraction          |                   47.22 |         26.60 |            5.0417 |              68.49 |               29.23 |              35.75 |
| Prism Reflection    |                   66.67 |         20.40 |            3.6521 |              77.18 |               45.72 |              52.86 |
| **3 Variables**     |                          |               |                   |                    |                     |                    |
| **4 Variables**     |                          |               |                   |                    |                     |                    |
| **5 Variables**     |                          |               |                   |                    |                     |                    |

## explicitly functional  prompt
| **Scene**          | **TPR (%)/Recall (%) ↑** | **FPR (%) ↓** | **SHD (count) ↓** | **Accuracy (%) ↑** | **Precision (%) ↑** | **F1 Score (%) ↑** |
|---------------------|-------------------------:|--------------:|------------------:|-------------------:|--------------------:|-------------------:|
| Spring              |                   96.43 |          4.08 |            0.3571 |              96.18 |               95.94 |              96.18 |
| Seesaw              |                   95.00 |          2.14 |            0.65   |              97.40 |               92.50 |              93.73 |
| Magnetic Field      |                   66.67 |         17.46 |            1.8889 |              79.01 |               52.16 |              58.48 |
| Reflection          |                  100.00 |          0.00 |            0.00   |             100.00 |              100.00 |             100.00 |
| Refraction          |                   61.54 |         20.41 |            3.8077 |              76.20 |               42.03 |              48.87 |
| Prism Reflection    |                   62.82 |         21.01 |            3.8461 |              75.96 |               41.35 |              49.14 |
| **3 Variables**     |                          |               |                   |                    |                     |                    |
| **4 Variables**     |                          |               |                   |                    |                     |                    |
| **5 Variables**     |                          |               |                   |                    |                     |                    |
