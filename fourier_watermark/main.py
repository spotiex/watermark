import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import cv2

from utils import read_image, save_image, preprocess_watermark
from embed import embed_watermark, embed_watermark_dct_like
from extract import extract_watermark, extract_watermark_dct_like
from metrics import calculate_psnr, calculate_ssim, calculate_nc, calculate_ber

# 设置中文字体，防止显示乱码
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']  
plt.rcParams['axes.unicode_minus'] = False

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='基于傅里叶变换的数字水印嵌入和提取')
    
    # 参数列表
    parser.add_argument('--host', required=True, help='宿主图像文件路径')
    parser.add_argument('--watermark', required=True, help='水印图像文件路径')
    parser.add_argument('--alpha', type=float, default=0.1, help='水印强度因子')
    parser.add_argument('--output_dir', required=True, help='输出目录路径')
    parser.add_argument('--method', choices=['standard', 'dct_like'], default='standard', 
                        help='水印嵌入方法：标准FFT区域或DCT风格的FFT实现')
    
    return parser.parse_args()

def visualize_results(original, watermarked, original_watermark, extracted_watermark, output_path):
    """可视化结果"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 显示原始图像(彩色或灰度)
    if len(original.shape) == 3:
        axes[0, 0].imshow(original)
    else:
        axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('原始图像', fontsize=12)
    axes[0, 0].axis('off')
    
    # 显示带水印图像(彩色或灰度)
    if len(watermarked.shape) == 3:
        axes[0, 1].imshow(watermarked)
    else:
        axes[0, 1].imshow(watermarked, cmap='gray')
    axes[0, 1].set_title('带水印图像', fontsize=12)
    axes[0, 1].axis('off')
    
    # 显示原始水印
    axes[1, 0].imshow(original_watermark, cmap='gray')
    axes[1, 0].set_title('原始水印', fontsize=12)
    axes[1, 0].axis('off')
    
    # 显示提取的水印
    axes[1, 1].imshow(extracted_watermark, cmap='gray')
    axes[1, 1].set_title('提取水印', fontsize=12)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def visualize_fft_spectrum(img_block, fft_coeffs, output_path):
    """可视化原始图像块和其FFT频谱"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # 显示原始图像块
    axes[0].imshow(img_block, cmap='gray')
    axes[0].set_title('原始图像块', fontsize=12)
    axes[0].axis('off')
    
    # 显示FFT频谱（取对数以增强可视性）
    magnitude_spectrum = np.log(np.abs(fft_coeffs) + 1)
    axes[1].imshow(magnitude_spectrum, cmap='viridis')
    axes[1].set_title('FFT频谱（对数尺度）', fontsize=12)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    """主函数"""
    args = parse_arguments()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 读取宿主图像(保持原格式)和水印图像(转为灰度)
    host_img = read_image(args.host, as_gray=False)
    watermark = read_image(args.watermark, as_gray=True)
    
    # 设置固定参数
    block_size = (8, 8)  # 默认块大小为8x8
    
    # 根据宿主图像尺寸确定水印大小
    if len(host_img.shape) == 3:
        h, w, _ = host_img.shape
    else:
        h, w = host_img.shape
    watermark_size = min(h, w) // 16  # 更合理的水印大小
    
    # 预处理水印成二值图像
    watermark_binary = preprocess_watermark(watermark, (watermark_size, watermark_size))
    watermark_path = os.path.join(args.output_dir, 'watermark_binary.png')
    save_image(watermark_binary * 255, watermark_path)
    
    # 根据选择的方法嵌入水印
    if args.method == 'standard':
        watermarked_img = embed_watermark(host_img, watermark_binary, block_size, args.alpha)
    else:  # 'dct_like'
        watermarked_img = embed_watermark_dct_like(host_img, watermark_binary, block_size, args.alpha)
        
    watermarked_path = os.path.join(args.output_dir, 'watermarked.png')
    save_image(watermarked_img, watermarked_path)
    
    # 提取水印 (非盲提取)
    if args.method == 'standard':
        extracted_watermark = extract_watermark(
            watermarked_img, host_img, watermark_binary.shape, block_size
        )
    else:  # 'dct_like'
        extracted_watermark = extract_watermark_dct_like(
            watermarked_img, host_img, watermark_binary.shape, block_size
        )
    
    extracted_path = os.path.join(args.output_dir, 'extracted_watermark.png')
    save_image(extracted_watermark * 255, extracted_path)
    
    # 为评估指标转换为灰度图像(如果需要)
    if len(host_img.shape) == 3:
        host_gray = cv2.cvtColor(host_img, cv2.COLOR_RGB2GRAY)
        watermarked_gray = cv2.cvtColor(watermarked_img, cv2.COLOR_RGB2GRAY)
    else:
        host_gray = host_img
        watermarked_gray = watermarked_img
    
    # 计算评估指标
    psnr_value = calculate_psnr(host_gray, watermarked_gray)
    ssim_value = calculate_ssim(host_gray, watermarked_gray)
    nc_value = calculate_nc(watermark_binary, extracted_watermark)
    ber_value = calculate_ber(watermark_binary, extracted_watermark)
    
    # 将评估结果保存到文件
    metrics_path = os.path.join(args.output_dir, 'metrics.txt')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write("基于傅里叶变换的数字水印评估结果:\n")
        f.write(f"使用方法: {args.method}\n")
        f.write(f"水印强度因子 (alpha): {args.alpha}\n\n")
        f.write(f"PSNR (宿主图像质量): {psnr_value:.2f} dB\n")
        f.write(f"SSIM (结构相似性): {ssim_value:.4f}\n")
        f.write(f"NC (归一化相关性): {nc_value:.4f}\n")
        f.write(f"BER (位错误率): {ber_value:.4f}\n")
    
    # 打印评估结果
    print("\n基于傅里叶变换的数字水印评估结果:")
    print(f"使用方法: {args.method}")
    print(f"水印强度因子 (alpha): {args.alpha}")
    print(f"PSNR (宿主图像质量): {psnr_value:.2f} dB")
    print(f"SSIM (结构相似性): {ssim_value:.4f}")
    print(f"NC (归一化相关性): {nc_value:.4f}")
    print(f"BER (位错误率): {ber_value:.4f}")
    
    # 可视化结果
    visualize_path = os.path.join(args.output_dir, 'results_visualization.png')
    visualize_results(host_img, watermarked_img, watermark_binary * 255, 
                     extracted_watermark * 255, visualize_path)
    
    # 可视化FFT频谱（使用第一个块作为示例）
    from utils import get_blocks, fft_transform
    blocks, _, _ = get_blocks(host_img, block_size)
    first_block = blocks[0]
    fft_coeffs = fft_transform(first_block)
    
    spectrum_path = os.path.join(args.output_dir, 'fft_spectrum.png')
    visualize_fft_spectrum(first_block, fft_coeffs, spectrum_path)
    
    # 显示图像差异
    if len(host_img.shape) == 3:
        diff_img = cv2.cvtColor(host_img, cv2.COLOR_RGB2GRAY) - cv2.cvtColor(watermarked_img, cv2.COLOR_RGB2GRAY)
    else:
        diff_img = host_img - watermarked_img
    
    diff_img = np.abs(diff_img)
    # 为了更好地可视化差异，将差异放大
    diff_img = np.clip(diff_img * 10, 0, 255).astype(np.uint8)
    
    diff_path = os.path.join(args.output_dir, 'difference_image.png')
    save_image(diff_img, diff_path)
    
    print(f"\n所有结果已保存到目录: {args.output_dir}")

if __name__ == "__main__":
    main()