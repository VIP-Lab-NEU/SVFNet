# SVFNet: Underwater RGB-Sonar Fusion Classification via Cross-Modal Contrastive Learning and Global Query Spatial Attention

> Official PyTorch implementation of the paper "SVFNet: Underwater RGB-Sonar Fusion Classification via Cross-Modal Contrastive Learning and Global Query Spatial Attention".

## 📖 Abstract
Precise underwater target classification is a prerequisite for autonomous marine operations. Given the limitations of single sensors, multi-sensor fusion is essential. While RGB images provide rich texture information, they suffer from interference caused by underwater light attenuation, leading to compromised classification performance. Conversely, sonar provides robust structural features that effectively compensate for the degradation of RGB data in the presence of interference. However, effective fusion is hindered by inherent heterogeneity, asymmetric feature degradation, and paired data scarcity. To overcome these limitations, an adapted dual-stream framework named SVFNet is developed specifically for RGB-Sonar (RGB-S) fusion classification. Specifically, well-established cross-modal contrastive learning is incorporated to bridge the inherent heterogeneity, while a Global Query Spatial Attention (GQSA) module is adapted and integrated to tackle asymmetric degradation. Finally, to address the extreme scarcity of paired data collected in unconstrained real-world environments, R-S9 is constructed as the first spatiotemporally aligned underwater RGB-S paired benchmark dataset for fusion classification, containing 3,732 image pairs. Experiments on R-S9 demonstrate that SVFNet significantly outperforms leading unimodal baselines, SOTA long-tailed classification methods, and advanced RGB-X fusion architectures. 

**Keywords**: *Multi-sensor fusion, Underwater target recognition, Autonomous Underwater Vehicle (AUV), Contrastive learning, Attention mechanisms*

## 📋 Features
- **Dual-Stream Architecture**: Processes RGB and Sonar images simultaneously to overcome asymmetric feature degradation.
- **Cross-Modal Contrastive Learning**: Bridges the inherent heterogeneity between optical and acoustic modalities.
- **Global Query Spatial Attention (GQSA)**: Adaptively enhances feature representation across modalities.
- **Comprehensive Evaluation Tools**: Built-in support for PR curves, Confusion Matrices, t-SNE clustering, and Grad-CAM heatmaps.

## 📂 Dataset Preparation 

Organize your dataset in the following structure. 

```text
## 📂 Dataset Preparation (R-S9)

Organize your dataset in the following structure. 

```text
dataset_dir/
├── rgb/                     # RGB images (.jpg or .png)
├── sonar/                   # Sonar images (.jpg or .png)
├── train_scenelist.txt      # Training labels
├── val_scenelist.txt        # Validation labels
└── test_scenelist.txt       # Testing labels
```

## 🛠️ Installation

1. Clone the repository:
   ```bash
  git clone https://github.com/VIP-Lab-NEU/SVFNet.git
   cd SVFNet
   ```

      
