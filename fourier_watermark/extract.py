import numpy as np
from utils import fft_transform, get_blocks, get_middle_frequency_indices

def extract_watermark(watermarked_img, original_img, watermark_shape, block_size=(8, 8), threshold=0):
    """
    从带水印图像中提取水印（非盲提取，需要原始图像）
    
    参数:
        watermarked_img: 带水印的图像
        original_img: 原始图像（用于比较）
        watermark_shape: 水印图像的形状，如(64, 64)
        block_size: 图像块大小
        threshold: 提取水印时的阈值，默认为0
        
    返回:
        提取的水印图像
    """
    # 获取图像块
    watermarked_blocks, w_positions, _ = get_blocks(watermarked_img, block_size)
    original_blocks, o_positions, _ = get_blocks(original_img, block_size)
    
    # 计算需要多少块来存储水印
    watermark_size = watermark_shape[0] * watermark_shape[1]
    if watermark_size > len(watermarked_blocks):
        raise ValueError("指定的水印尺寸过大，无法从图像中提取")
    
    # 获取中频区域索引
    mid_freq_indices = get_middle_frequency_indices(block_size)
    
    # 存储提取的水印位
    extracted_bits = np.zeros(watermark_size, dtype=np.uint8)
    
    # 对每个图像块提取水印位
    for i in range(watermark_size):
        # 对水印图像块和原始图像块执行FFT变换
        watermarked_fft = fft_transform(watermarked_blocks[i])
        original_fft = fft_transform(original_blocks[i])
        
        # 获取用于嵌入水印的中频位置
        embed_pos = mid_freq_indices[i % len(mid_freq_indices)]
        
        # 计算幅值差异
        watermarked_magnitude = np.abs(watermarked_fft[embed_pos])
        original_magnitude = np.abs(original_fft[embed_pos])
        diff = watermarked_magnitude - original_magnitude
        
        # 根据差异判断水印位的值
        if diff > threshold:
            extracted_bits[i] = 1
        else:
            extracted_bits[i] = 0
    
    # 重塑为水印图像形状
    extracted_watermark = extracted_bits.reshape(watermark_shape)
    
    return extracted_watermark

def extract_watermark_dct_like(watermarked_img, original_img, watermark_shape, block_size=(8, 8), threshold=0):
    """
    从带水印图像中提取水印（DCT风格的FFT实现，非盲提取）
    
    参数:
        watermarked_img: 带水印的图像
        original_img: 原始图像（用于比较）
        watermark_shape: 水印图像的形状，如(64, 64)
        block_size: 图像块大小
        threshold: 提取水印时的阈值，默认为0
        
    返回:
        提取的水印图像
    """
    # 获取图像块
    watermarked_blocks, w_positions, _ = get_blocks(watermarked_img, block_size)
    original_blocks, o_positions, _ = get_blocks(original_img, block_size)
    
    # 计算需要多少块来存储水印
    watermark_size = watermark_shape[0] * watermark_shape[1]
    if watermark_size > len(watermarked_blocks):
        raise ValueError("指定的水印尺寸过大，无法从图像中提取")
    
    # 存储提取的水印位
    extracted_bits = np.zeros(watermark_size, dtype=np.uint8)
    
    # 对每个图像块提取水印位
    for i in range(watermark_size):
        # 对水印图像块和原始图像块执行FFT变换
        watermarked_fft = fft_transform(watermarked_blocks[i])
        original_fft = fft_transform(original_blocks[i])
        
        # 选择嵌入的系数位置（模拟DCT中间频率系数）
        bh, bw = block_size
        embed_row, embed_col = bh//2 + 1, bw//2 + 1  # 中频位置
        
        # 计算幅值差异
        watermarked_magnitude = np.abs(watermarked_fft[embed_row, embed_col])
        original_magnitude = np.abs(original_fft[embed_row, embed_col])
        diff = watermarked_magnitude - original_magnitude
        
        # 根据差异判断水印位的值
        if diff > threshold:
            extracted_bits[i] = 1
        else:
            extracted_bits[i] = 0
    
    # 重塑为水印图像形状
    extracted_watermark = extracted_bits.reshape(watermark_shape)
    
    return extracted_watermark