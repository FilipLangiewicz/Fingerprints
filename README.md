# Fingerprint Thinning and Minutiae Detection

This project implements fingerprint image processing techniques to extract minutiae (ridge endings and bifurcations) from grayscale fingerprint images. It includes thinning algorithms, post-processing, and minutiae detection and filtering.

## Features

- **Thinning Algorithms**
  - `KMM`: An iterative thinning method for preserving ridge structure.
  - `Morphology`: A simpler morphological thinning method.

- **Post-Processing**
  - Corrects the original binary image and thinning result using morphological operations.

- **Minutiae Detection**
  - Identifies ridge endings and bifurcations.
  - Filters out closely spaced or redundant minutiae with similar directions.

- **Visualization**
  - Provides visualization of binary image, thinning result, detected minutiae, and corrected minutiae.

## File Structure

- `main.py`: Main script to run the processing pipeline.
- `utils/`: Helper functions for thinning, filtering, and visualization.
- `data/`: Folder to store fingerprint images.
