import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from utils import read_image, save_image, preprocess_watermark
from embed import embed_watermark
from extract import extract_watermark, blind_extract_watermark
from metrics import calculate_psnr, calculate_ssim, calculate_nc, calculate_ber
import cv2
# 设置中文字体，防止显示乱码
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']  
plt.rcParams['axes.unicode_minus'] = False

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='SVD水印嵌入和提取')
    
    # 简化后的参数列表
    parser.add_argument('--host', required=True, help='宿主图像文件路径')
    parser.add_argument('--watermark', required=True, help='水印图像文件路径')
    parser.add_argument('--alpha', type=float, default=0.1, help='水印强度因子')
    parser.add_argument('--output_dir', required=True, help='输出目录路径')
    
    return parser.parse_args()

def visualize_results(original, watermarked, original_watermark, extracted_watermark, output_path):
    """可视化结果"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # 显示原始图像(彩色或灰度)
    if len(original.shape) == 3:
        axes[0, 0].imshow(original)
    else:
        axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original Image', fontsize=12)
    axes[0, 0].axis('off')
    
    # 显示带水印图像(彩色或灰度)
    if len(watermarked.shape) == 3:
        axes[0, 1].imshow(watermarked)
    else:
        axes[0, 1].imshow(watermarked, cmap='gray')
    axes[0, 1].set_title('Watermarked Image', fontsize=12)
    axes[0, 1].axis('off')
    
    # 显示原始水印
    axes[1, 0].imshow(original_watermark, cmap='gray')
    axes[1, 0].set_title('Original Watermark', fontsize=12)
    axes[1, 0].axis('off')
    
    # 显示提取的水印
    axes[1, 1].imshow(extracted_watermark, cmap='gray')
    axes[1, 1].set_title('Extracted Watermark', fontsize=12)
    axes[1, 1].axis('off')
    
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
    watermark_size = min(h, w) // 8 // 2
    
    # 预处理水印成二值图像
    watermark_binary = preprocess_watermark(watermark, (watermark_size, watermark_size))
    watermark_path = os.path.join(args.output_dir, 'watermark_binary.png')
    save_image(watermark_binary * 255, watermark_path)
    
    # 嵌入水印
    watermarked_img = embed_watermark(host_img, watermark_binary, block_size, args.alpha)
    watermarked_path = os.path.join(args.output_dir, 'watermarked.png')
    save_image(watermarked_img, watermarked_path)
    
    # 提取水印 (非盲提取)
    extracted_watermark = extract_watermark(
        watermarked_img, host_img, watermark_binary.shape, block_size
    )
    extracted_path = os.path.join(args.output_dir, 'extracted_watermark.png')
    save_image(extracted_watermark * 255, extracted_path)
    
    # 盲提取水印
    blind_extracted_watermark = blind_extract_watermark(
        watermarked_img, watermark_binary.shape, block_size
    )
    blind_extracted_path = os.path.join(args.output_dir, 'blind_extracted_watermark.png')
    save_image(blind_extracted_watermark * 255, blind_extracted_path)
    
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
    blind_nc_value = calculate_nc(watermark_binary, blind_extracted_watermark)
    blind_ber_value = calculate_ber(watermark_binary, blind_extracted_watermark)
    
    # 打印评估结果
    print("\nWatermark Evaluation Results:")
    print(f"PSNR (Host Image Quality): {psnr_value:.2f} dB")
    print(f"SSIM (Structural Similarity): {ssim_value:.4f}")
    print(f"Non-blind Extraction - NC (Normalized Correlation): {nc_value:.4f}")
    print(f"Non-blind Extraction - BER (Bit Error Rate): {ber_value:.4f}")
    print(f"Blind Extraction - NC (Normalized Correlation): {blind_nc_value:.4f}")
    print(f"Blind Extraction - BER (Bit Error Rate): {blind_ber_value:.4f}")
    
    # 可视化结果
    visualize_path = os.path.join(args.output_dir, 'results_visualization.png')
    visualize_results(host_img, watermarked_img, watermark_binary * 255, extracted_watermark * 255, visualize_path)
    
    print(f"\nAll results saved to directory: {args.output_dir}")

if __name__ == "__main__":
    main()