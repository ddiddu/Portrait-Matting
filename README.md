# Portrait Matting: Real-Time Background Removal Using Deep Learning

## Overview

This project presents a fast and effective real-time portrait matting model, developed during an internship at Samsung SDS for the Knox Meeting video conferencing service. The model is designed for real-time use with low CPU usage and is optimized for environments without GPU support, such as virtual desktop infrastructure (VDI).

## Features

- **Real-Time Performance:** Achieves 11.4 FPS on a 2.6GHz 6-core Intel Core i7 CPU, about 1.5x faster than MODNet.
- **High Accuracy:** Outperforms MODNet, especially in challenging poses and clothing, and in cases where foreground and background colors are similar.
- **Robustness:** Enhanced stability across various lighting and camera conditions using Consistency Constraint Loss.
- **Ease of Training:** Trains using only RGB images as input, without requiring trimaps or ground truth alpha mattes.

## Model Architecture

- **Base Model:** MODNet, a state-of-the-art portrait matting model.
- **Custom Improvements:**
  - Removed the detail prediction branch for speed.
  - Added upsampling and convolution layers to the semantic estimation branch.
  - Introduced Dense Connections to preserve high-resolution details and reduce overfitting.
  - Applied Consistency Constraint Loss for robustness to texture and lighting changes.

## Training

- **Transfer Learning:** Fine-tuned from pretrained MODNet weights.
- **Knowledge Distillation:** Used MODNet outputs as ground truth for training the custom model.
- **Data:** Trained on the Supervisely Person Segmentation dataset (2,667 images).
- **Data Augmentation:** Applied random noise, blur, color, brightness, contrast, and sharpness changes to improve generalization.
- **Validation:** Evaluated on the PPM-100 benchmark using MSE and MAD metrics.

## Results

| Model      | Consistency Loss | MSE (↓) | MAD (↓) |
|------------|-----------------|---------|---------|
| MODNet     |                 | 0.01324 | 0.01944 |
| MODNet-S   |                 | 0.01700 | 0.02445 |
| MODNet-D   |                 | 0.01098 | 0.01679 |
| MODNet-C   | ✓               | 0.00829 | 0.01362 |

- **Model Size:** Comparable to MODNet, with minimal increase in parameters.
- **Inference Speed:** 11.4 FPS (MODNet-C) vs. 7.8 FPS (MODNet).

## Usage

1. **Install Requirements**
	```
	pip install -r requirements.txt
	```
2. **Run Demo**
	```
	python launch.py
	```
	This launches a Gradio interface for real-time background removal.

3. **ONNX Export and Inference**
	- Export models to ONNX using scripts in the `onnx/` directory.
	- Run inference using the provided ONNX models.

## References

- [MODNet: Is a Green Screen Really Necessary for Real-Time Portrait Matting?](https://arxiv.org/abs/2011.11961)
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
- [Supervisely Person Dataset](https://supervise.ly)
- [PortraitNet: Real-time portrait segmentation network for mobile device](https://www.sciencedirect.com/science/article/pii/S0097849319300992)

## Acknowledgements

Developed as part of an internship at Samsung SDS, Knox Meeting Development Group.
