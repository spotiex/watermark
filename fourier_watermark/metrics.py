import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def calculate_psnr(original, watermarked):
    """
    计算峰值信噪比 (PSNR)
    
    参数:
        original: 原始图像
        watermarked: 水印图像
        
    返回:
        PSNR值 (dB)
    """
    # 确保输入类型一致
    original = original.astype(np.float64)
    watermarked = watermarked.astype(np.float64)
    
    # 确保图像是灰度图像
    if len(original.shape) == 3:
        original = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    if len(watermarked.shape) == 3:
        watermarked = cv2.cvtColor(watermarked, cv2.COLOR_RGB2GRAY)
    
    # 使用scikit-image的psnr函数
    return psnr(original, watermarked, data_range=255)

def calculate_ssim(original, watermarked):
    """
    计算结构相似性指数 (SSIM)
    
    参数:
        original: 原始图像
        watermarked: 水印图像
        
    返回:
        SSIM值 (0-1)
    """
    # 确保输入类型一致
    original = original.astype(np.float64)
    watermarked = watermarked.astype(np.float64)
    
    # 确保图像是灰度图像
    if len(original.shape) == 3:
        original = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    if len(watermarked.shape) == 3:
        watermarked = cv2.cvtColor(watermarked, cv2.COLOR_RGB2GRAY)
    
    # 使用scikit-image的ssim函数
    return ssim(original, watermarked, data_range=255)

def calculate_nc(original_watermark, extracted_watermark):
    """
    计算归一化相关性 (NC)
    
    参数:
        original_watermark: 原始水印
        extracted_watermark: 提取的水印
        
    返回:
        NC值 (0-1)
    """
    # 确保输入类型一致
    original_watermark = original_watermark.astype(np.float64)
    extracted_watermark = extracted_watermark.astype(np.float64)
    
    # 计算归一化相关性
    numerator = np.sum(original_watermark * extracted_watermark)
    denominator = np.sqrt(np.sum(original_watermark**2) * np.sum(extracted_watermark**2))
    
    # 避免除零错误
    if denominator == 0:
        return 0
    
    return numerator / denominator

def calculate_ber(original_watermark, extracted_watermark):
    """
    计算位错误率 (BER)
    
    参数:
        original_watermark: 原始水印 (二值)
        extracted_watermark: 提取的水印 (二值)
        
    返回:
        BER值 (0-1)
    """
    # 确保输入是二值图像
    original_watermark = (original_watermark > 0).astype(np.uint8)
    extracted_watermark = (extracted_watermark > 0).astype(np.uint8)
    
    # 计算错误位数
    total_bits = original_watermark.size
    error_bits = np.sum(original_watermark != extracted_watermark)
    
    return error_bits / total_bits