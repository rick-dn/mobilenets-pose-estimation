## Code for "Adapting MobileNets for mobile based upper body pose estimation"

[![DOI](https://img.shields.io/badge/DOI-10.1109%2FAVSS.2018.8639378-blue)](https://doi.org/10.1109/AVSS.2018.8639378)

This repository contains code related to the research paper "Adapting MobileNets for mobile based upper body pose estimation" by Debnath et al.

**Paper Summary:**

The paper explores adapting MobileNets, a lightweight and efficient CNN architecture, for the task of human pose estimation, specifically for mobile-based systems. 

**Key Contributions:**

* We adapt MobileNets for pose estimation, drawing inspiration from the hourglass network architecture.   
* A novel "split stream" architecture is introduced in the final layers of MobileNets. This design choice aims to reduce overfitting and improve accuracy while also decreasing the model's parameter size. 
* The approach leverages transfer learning by utilizing pre-trained MobileNets weights from ImageNet to boost accuracy. [cite: 246, 247, 248, 249, 250, 251, 252, 289, 290]

**Key Findings:**

* The adapted MobileNets model outperforms the original MobileNets for pose estimation.   
* The adapted model achieves comparable accuracy to state-of-the-art methods but with a significant reduction in inference time, making it more suitable for mobile applications.

**Important Notes:**

* This code is provided for reference purposes only and may not be actively maintained.
* For detailed information on the model architecture, training procedure, and experimental results, please refer to the original paper.

**Citation:**

If you use this code or find it helpful, please cite our paper.
