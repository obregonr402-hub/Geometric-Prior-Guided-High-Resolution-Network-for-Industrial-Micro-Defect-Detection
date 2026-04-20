# Geometric Prior-Guided High-Resolution Network for Industrial Micro-Defect Detection
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19657217.svg)](https://doi.org/10.5281/zenodo.19657217)

> **⚠️ Notice**  
> This code is directly related to our manuscript submitted to *The Visual Computer*.  
> **Readers are encouraged to cite our paper** if they find this work useful for their research.

---

## 📖 Abstract

In PCB surface defect detection, the weak representation of tiny defects, severe background interference, and the coexistence of multi-scale targets remain major challenges that hinder detection sensitivity and localization accuracy. To address these challenges, we propose a high-resolution geometric-prior-enhanced network, termed **HCD-Net**. Specifically, a high-resolution guided feature extraction module is introduced to preserve fine-grained defect features while improving the representation of small targets with minimal computational overhead. A cross-axis spatial attention mechanism, termed **CASA**, is further incorporated to model defect regions and their spatial dependencies, improving anomaly discrimination in complex backgrounds. In addition, a dual-filtering feature pyramid structure is designed to strengthen multi-scale feature interaction through channel-wise and spatial filtering, leading to more accurate defect localization. Furthermore, the **NWD-InnerSIoU** loss is employed to improve bounding box regression for small and irregular defects while alleviating instability during training. Experimental results on the PCB defect dataset demonstrate that HCD-Net achieves a **mAP@50 of 90.2%**, outperforming several mainstream detection models and surpassing RT-DETR by approximately **3.0%**, while maintaining low parameter complexity and computational cost.

---

## 🎯 Key Contributions

| Module | Description |
|--------|-------------|
| **HGRA Backbone** | High-resolution guided feature extraction with dilated convolutions and max-pooling peak preservation |
| **CASA** | Cross-Axis Spatial Attention incorporating geometric priors with O(N) complexity |
| **DSFPN** | Dual-filtering Feature Pyramid Network for noise-suppressed multi-scale fusion |
| **NWD-InnerSIoU** | Joint loss function balancing tiny defect robustness and geometry-aware localization |

---

## 📊 Experimental Results

### PCB Dataset (6 defect classes)

| Metric | Value |
|--------|-------|
| mAP@50 | **90.2%** |
| Recall | **82.0%** |
| Precision | 95.8% |
| Parameters | **14.17M** |
| GFLOPs | **45.3** |
| Inference Time | 12.3 ms |

### Improvement over RT-DETR

| Metric | Gain |
|--------|------|
| mAP@50 | **+2.6%** |
| Recall | **+4.8%** |
| Parameters | **-28.7%** |
| GFLOPs | **-20.5%** |

---

## 🖥️ Environment & Dependencies

- **OS**: Ubuntu 20.04.5 LTS
- **Python**: 3.8.10
- **PyTorch**: 2.0.1
- **CUDA**: 11.7
- **Ultralytics**: 8.3.63

### Installation

```bash
pip install -r requirements.txt
