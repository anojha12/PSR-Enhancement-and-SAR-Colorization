# PSR Image Enhancement Techniques

This repository contains implementations of various image enhancement techniques for Permanently Shadowed Regions (PSR) of the Moon. The project explores methods to improve visibility in these regions while preserving crucial terrain texture information for mission planning purposes.

## Overview

PSRs are regions on the Moon's poles that never receive direct sunlight due to the Moon's minimal axial tilt. While these regions are not completely dark, they are poorly illuminated by scattered light from surrounding features. This project implements and evaluates several enhancement techniques to improve visibility in these challenging conditions.

## Implemented Methods

1. **Dual Illumination Estimation**
   - Optimizes illumination maps for both under and over-exposed regions
   - Uses reflective padding to prevent streaking artifacts
   - Configurable lambda (λ) and gamma (γ) parameters

2. **Mask-Based Controllable Light Enhancement Diffusion (Mask-CLED)**
   - Employs conditional diffusion models for targeted enhancement
   - Uses binary masking with dilation for region of interest selection
   - Controllable brightness levels for specific regions

3. **LoLi-IEA (CNN-based Enhancement)**
   - Two-stage CNN architecture
   - Automatic light classification for local/global enhancement
   - Improved performance without manual mask generation

4. **SAR Colorization**
   - Pix2pix GAN implementation for SAR image colorization
   - Trained on Sentinel-1 (SAR) and Sentinel-2 (optical) imagery
   - Helps in terrain texture mapping and classification

## Performance Metrics

The enhancement techniques are evaluated using:
- Lightness Order Error (LOE)
- SPAQ (Perceptual Image Quality Assessment)
- Cosine Similarity Score

## Dataset

The project uses:
- PSR imagery from Chandrayaan-2 Orbiter High Resolution Camera
- SEN1-2 dataset for SAR colorization (282,384 co-registered image patches)

## Authors
- Bharath Raam Radhakrishnan
- Aneesh Ojha
