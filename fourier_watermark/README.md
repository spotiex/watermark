# FFT-Based Digital Watermarking System

This project implements a digital watermarking system based on the Fast Fourier Transform (FFT). The system allows embedding a binary watermark image into a host image and extracting the watermark from the watermarked image.

## Features

- Support for color and grayscale host images
- Two watermark embedding methods:
  - Standard method: embeds watermark in multiple positions in the middle frequency region
  - DCT-like method: simulates DCT transform by embedding at specific middle frequency coefficient positions
- Non-blind extraction (requires the original image)
- Multiple evaluation metrics: PSNR, SSIM, NC, BER
- Result visualization: original/watermarked images, extracted watermark, spectrum analysis

## Working Principle

The system uses the Fourier transform to convert images from the spatial domain to the frequency domain, where the watermark information is embedded by modifying the magnitude of specific frequency components. The detailed steps are:

1. **Image Blocking**: Divide the host image into non-overlapping 8×8 pixel blocks
2. **Fourier Transform**: Perform 2D discrete Fourier transform on each block
3. **Watermark Embedding**:
   - Binarize the watermark to 0 or 1
   - For each watermark bit, modify the magnitude of a specific position in the frequency domain of the corresponding image block
   - Increase magnitude if the watermark bit is 1; decrease magnitude if the bit is 0
4. **Inverse Transform**: Perform inverse Fourier transform on the modified frequency coefficients to obtain the watermarked image
5. **Watermark Extraction**:
   - Compare the magnitude difference in the frequency domain between the watermarked image and the original image

## File Structure

- `utils.py`: Provides basic functions such as image reading/writing, Fourier transform, image blocking, etc.
- `embed.py`: Implements watermark embedding algorithms
- `extract.py`: Implements watermark extraction algorithms
- `metrics.py`: Implements evaluation metric calculations
- `main.py`: Main program that processes command line arguments and calls various modules

## System Requirements

- Python 3.6+
- Dependencies:
  - NumPy
  - OpenCV (cv2)
  - Matplotlib
  - SciPy
  - scikit-image

## Usage

```bash
python main.py --host <host_image_path> --watermark <watermark_image_path> --output_dir <output_directory> [--alpha <watermark_strength>] [--method <embedding_method>]
```

Parameter description:
- `--host`: Host image file path (required)
- `--watermark`: Watermark image file path (required)
- `--output_dir`: Output directory path (required)
- `--alpha`: Watermark strength factor, affects watermark invisibility and robustness (default: 0.1)
- `--method`: Watermark embedding method, options are 'standard' or 'dct_like' (default: 'standard')

## Example

```bash
python main.py --host images/lena.png --watermark images/logo.png --output_dir results --alpha 0.2 --method standard
```

## Differences Between Standard FFT and DCT-like Methods

1. **Standard FFT Method**:
   - Embeds watermark at multiple positions in the middle frequency region of the Fourier spectrum
   - More dispersed embedding positions may provide better robustness

2. **DCT-like Method**:
   - Simulates the middle frequency coefficient positions commonly used in DCT transform
   - More concentrated embedding positions, which may provide better visual quality in some cases

## Output Results

After executing the program, the following files will be generated in the specified output directory:

- `watermark_binary.png`: Preprocessed binary watermark
- `watermarked.png`: Watermarked host image
- `extracted_watermark.png`: Extracted watermark
- `results_visualization.png`: Result visualization image
- `fft_spectrum.png`: FFT spectrum example image
- `difference_image.png`: Difference between original and watermarked image (magnified 10×)
- `metrics.txt`: Detailed evaluation metric results