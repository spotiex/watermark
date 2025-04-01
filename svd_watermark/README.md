# SVD-Based Image Watermarking

This project implements a Singular Value Decomposition (SVD) based digital watermarking system for images. It provides functionality to embed a watermark into a host image and then extract it, with both non-blind and blind extraction methods.

## Features

- Embed watermarks into color or grayscale host images
- Extract watermarks using both non-blind (with original) and blind methods
- Calculate image quality metrics (PSNR, SSIM) after watermarking
- Calculate watermark extraction accuracy metrics (NC, BER)
- Visualize results with before/after comparisons

## Installation

### Requirements

- Python 3.6+ (Project Python version 3.12)
- NumPy
- OpenCV (cv2)
- Matplotlib
- scikit-image

### Setup

1. Clone this repository:
   ```
   git clone https://github.com/spotiex/svd-watermark.git
   cd svd-watermark
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## File Structure

- `utils.py`: Basic utility functions for image processing and SVD operations
- `embed.py`: Functions for embedding watermarks into images
- `extract.py`: Functions for extracting watermarks from images
- `metrics.py`: Functions for calculating image quality and watermark extraction metrics
- `main.py`: Main program that integrates all components and provides a command-line interface

## Usage

```
python main.py --host <host_image> --watermark <watermark_image> --alpha <strength> --output_dir <output_directory>
```

### Parameters

- `--host`: Path to the host image (can be color or grayscale)
- `--watermark`: Path to the watermark image (will be converted to grayscale)
- `--alpha`: Watermark embedding strength factor (default: 0.1)
- `--output_dir`: Directory where the results will be saved

### Example

```
python main.py --host images/lena.png --watermark images/logo.png --alpha 0.2 --output_dir ./results
```

## Output Files

The program generates the following output files in the specified directory:

- `watermark_binary.png`: The preprocessed binary watermark
- `watermarked.png`: The host image with the embedded watermark
- `extracted_watermark.png`: The watermark extracted using the non-blind method
- `blind_extracted_watermark.png`: The watermark extracted using the blind method
- `results_visualization.png`: A visual comparison of the original and watermarked images, along with the original and extracted watermarks

## Methodology

### Embedding Process

1. The host image is divided into non-overlapping 8×8 blocks
2. SVD is applied to each block
3. The watermark is converted to a binary image
4. Watermark bits are embedded by modifying the singular values of the host image blocks
5. The modified blocks are reconstructed to form the watermarked image

### Extraction Process

#### Non-blind Extraction
- Requires both the watermarked image and the original host image
- Compares the singular values of corresponding blocks to extract the watermark bits

#### Blind Extraction
- Uses only the watermarked image
- Estimates the watermark bits based on the statistical properties of the singular values

## Performance Metrics

- **PSNR (Peak Signal-to-Noise Ratio)**: Measures the quality of the watermarked image compared to the original
- **SSIM (Structural Similarity Index)**: Measures the perceived quality of the watermarked image
- **NC (Normalized Correlation)**: Measures the similarity between the original and extracted watermarks
- **BER (Bit Error Rate)**: Measures the proportion of bit errors in the extracted watermark