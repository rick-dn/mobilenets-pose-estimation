## Code for "Adapting MobileNets for mobile based upper body pose estimation"

[![DOI](https://img.shields.io/badge/DOI-10.1109%2FAVSS.2018.8639378-blue)](https://doi.org/10.1109/AVSS.2018.8639378)
This repository contains code related to the research paper "Adapting MobileNets for mobile based upper body pose estimation" by Debnath et al.

**Paper Summary:**

The paper explores adapting MobileNets, a lightweight and efficient CNN architecture, for the task of human pose estimation, specifically for mobile-based systems. [cite: 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229]

**Key Contributions:**

* The authors adapt MobileNets for pose estimation, drawing inspiration from the hourglass network architecture. [cite: 228, 229, 292, 293]
   
* A novel "split stream" architecture is introduced in the final layers of MobileNets. [cite: 297, 298, 308, 309, 310, 311, 312] This design choice aims to reduce overfitting and improve accuracy while also decreasing the model's parameter size. [cite: 297, 298, 308, 309, 310, 311, 312]
   
* The approach leverages transfer learning by utilizing pre-trained MobileNets weights from ImageNet to boost accuracy. [cite: 246, 247, 248, 249, 250, 251, 252, 289, 290]

**Key Findings:**

* The adapted MobileNets model outperforms the original MobileNets for pose estimation. [cite: 345, 346, 347, 348, 349, 350, 351, 352]
   
* The adapted model achieves comparable accuracy to state-of-the-art methods but with a significant reduction in inference time, making it more suitable for mobile applications. [cite: 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344]

**Important Notes:**

* This code is provided for reference purposes only and may not be actively maintained.
* For detailed information on the model architecture, training procedure, and experimental results, please refer to the original paper.

**Citation:**

If you use this code or find it helpful, please cite the original paper:
