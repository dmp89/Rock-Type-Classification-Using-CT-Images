# μCT CNN Benchmark for Rock Type Classification

This repository contains the code used in the study  
**“Deep Learning-Based Benchmarking of CNN Architectures for Rock Type Classification Using μCT Images”**.

The purpose of this codebase is to provide a **reproducible and standardized workflow** for benchmarking multiple pretrained convolutional neural network (CNN) architectures applied to rock type classification (sandstone, limestone, and shale) using micro-computed tomography (μCT) images.

## Overview

The implemented pipeline evaluates 17 pretrained CNN architectures under controlled conditions, with a strong emphasis on:
- consistent preprocessing and data augmentation,
- **sample-level data splitting to prevent data leakage**,
- fair architectural comparison using identical training and validation protocols,
- class-wise and global performance evaluation.

The workflow is designed for **benchmarking and methodological comparison**, rather than for deployment in field-scale or operational settings.

## Key Features

- Support for multiple pretrained CNN architectures (e.g., DenseNet, Inception, NASNet, ResNet).
- Sample-level train/validation/test split to ensure realistic evaluation.
- GPU-accelerated training.
- Standard performance metrics, including accuracy, precision, recall, MPCA, confusion matrices, and learning curves.
- Modular structure to facilitate extension to additional datasets or architectures.

## Data

Due to size constraints, μCT image datasets are **not included** in this repository.  
Users are expected to organize their datasets following the directory structure described in the configuration files.

Non-geological images used for out-of-domain input rejection (the “Others” class) were sourced from the COCO dataset.

## Reproducibility

All experiments reported in the associated manuscript were conducted using this codebase, with fixed random seeds and consistent preprocessing steps. Hardware and software dependencies are specified to support reproducibility.

## Disclaimer

This code is intended for **research and benchmarking purposes only**. Reported performance metrics represent upper-bound results obtained under controlled experimental conditions and should not be interpreted as direct indicators of performance in heterogeneous or field-acquired μCT workflows.

## Contact

For questions related to the code or the associated publication, please contact the corresponding author.
