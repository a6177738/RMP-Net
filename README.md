# Code for "Reliable Multi-modal Prototypical Contrastive Learning for Difficult Airway Assessment"

This repository contains partial code for the paper:

**"Reliable Multi-modal Prototypical Contrastive Learning for Difficult Airway Assessment"**  
Authors: Xiaofan Li, Bo Peng, et al.

## Description
This repository includes the training scripts for **individual modalities** (e.g., image, keypoints, and laryngoscope) used in the study. The **complete code**, including the full multi-modal training and evaluation framework, will be released **once the paper is officially accepted**.

We aim to provide the research community with access to modality-specific experiments while maintaining the integrity of the peer-review process.

## Contents
- `image_training/`: Scripts for training on the image modality.
- `keypoints_training/`: Scripts for training on the keypoints modality.
- `laryngoscope_training/`: Scripts for training on the laryngoscope modality.
- Configurations and utility scripts for modality-specific experiments.

## Prerequisites
- **Python** >= 3.6
- **PyTorch**: Compatible with Python >= 3.6
- Additional dependencies:
  - `numpy`
  - `torch_geometric`
  - `matplotlib`
  - `scipy`
  - `sklearn`
  - Built-in `math` module

Install dependencies via `pip`:
```bash
pip install numpy torch torchvision torch_geometric matplotlib scipy scikit-learn
