# TFSNet: EEG-based Emotion Recognition using Temporal and Frequency-Spatial Feature

[![Paper](https://img.shields.io/badge/Paper-IEEE-blue)](https://ieeexplore.ieee.org/abstract/document/11388788)
[![Poster](https://img.shields.io/badge/Poster-PDF-red)]([poster](https://github.com/yrlee040212/TFSNet/blob/main/poster.pdf)
[![Code](https://img.shields.io/badge/Framework-PyTorch-green)](#)

This repository provides the official implementation of **TFSNet**, a multi-domain EEG emotion recognition model that integrates:

- Temporal representation (S4D encoder)
- Frequency-spatial representation (CNN + graph filtering)
- Multi-modal feature fusion

---

## Introduction

We propose **TFSNet**, a framework for EEG-based emotion recognition that combines temporal dynamics and frequency-spatial representations within a unified architecture.

The temporal branch models EEG sequences using **S4D**, while the frequency-spatial branch captures spectral and inter-channel relationships using CNN and graph-based filtering. The two representations are fused to improve classification performance.

---

## Repository Structure

```text
TFSNet/
├── dataset.py          # dataset loaders
├── model.py            # model definitions
├── s4_modules.py       # S4D implementation
├── fs_modules.py       # adjacency + graph operations
├── run_train_eval.py   # training & evaluation
└── vis_adj.py          # adjacency visualization
```

---

## Getting Started

### Requirements

- Python 3.8+
- PyTorch

Install dependencies:

```bash
pip install -r requirements.txt

```

### Requirements

The code expects EEG data to be organized in a subject-wise 10-fold structure:

```text
data_root/
└── valence/
    └── subject_0/
        └── fold_0/
            ├── train_segments.npy
            ├── train_de_grids.npy
            ├── train_labels.npy
            ├── val_segments.npy
            ├── val_de_grids.npy
            └── val_labels.npy
```

### Supported targets

- valence  
- arousal  
- dominance  

---

## Training

### Run training

```bash
python run_train_eval.py \
  --domain multi \
  --emotion valence \
  --train \
  --data_root ./data \
  --output_root ./output
```

---

### Available domains

- temporal : S4D-based temporal model  
- frequency : frequency-spatial model  
- multi : TFSNet (fusion)  

---

### Learnable adjacency

Use the following flag to enable a learnable adjacency matrix:

```bash
--train
```

---

## Evaluation

The pipeline performs:

- Subject-dependent 10-fold cross-validation  
- Binary classification  

### Metrics

- Accuracy  
- Precision  
- Recall  
- Specificity  
- F1-score  

Results are saved as `.csv` files.

---

## Output

```text
output_root/
└── domain/
    └── emotion/
        └── train_or_not/
            ├── model/
            ├── log/
            └── adj/
```

---

## Visualization

```bash
python vis_adj.py
```

---

## Reference

- S4: https://github.com/state-spaces/s4

---

## Citation

```bibtex
@inproceedings{lee2025tfsnet,
  title={TFSNet: EEG-based Emotion Recognition using Temporal and Frequency-Spatial Feature},
  author={Lee, Yeryeong and Jang, Hyeryung},
  booktitle={2025 16th International Conference on Information and Communication Technology Convergence (ICTC)},
  pages={111--116},
  year={2025},
  organization={IEEE}
}
```

---

## License

This project is for research and academic purposes.
