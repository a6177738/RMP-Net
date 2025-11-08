# Code for "Reliable Multi-modal Prototypical Contrastive Learning for Difficult Airway Assessment"

This repository contains partial code for the paper:

**"Reliable Multi-modal Prototypical Contrastive Learning for Difficult Airway Assessment"**  
Authors: Xiaofan Li, Bo Peng, et al.

## Description
This repository includes scripts for training and model implementation for **three modalities**: image, keypoints (front and side view), and laryngoscope. The code provided here focuses on individual modality processing, with the following files:

### Included Files
1. **Image Modality**:
   - `train_img.py`: Training script for the image modality.
   - `model_img.py`: Model definition for the image modality.

2. **Keypoints Modality**:
   - `Keypoints/GCNmodel3.py`: Graph Convolutional Network (GCN) model for keypoints (front view), including training functions.
   - `Keypoints/GCNmodel6.py`: Graph Convolutional Network (GCN) model for keypoints (side view), including training functions.

3. **Laryngoscope Modality**:
   - `Laryngoscope/train.py`: Training script for the laryngoscope modality.
   - `Laryngoscope/model.py`: Model definition for the laryngoscope modality.

The **complete multi-modal training and evaluation framework** will be released **once the paper is officially accepted**.

## Prerequisites
- **Python** >= 3.6
- **PyTorch**: Compatible with Python >= 3.6
- Additional dependencies:
  - `numpy`
  - `torch_geometric`
  - `matplotlib`
  - `scipy`
  - `sklearn`

Install dependencies via `pip`:
```bash
pip install numpy torch torchvision torch_geometric matplotlib scipy scikit-learn
