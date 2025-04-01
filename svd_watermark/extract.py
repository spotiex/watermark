import numpy as np
from utils import svd_decomposition, get_blocks

def extract_watermark(watermarked_img, original_img, watermark_shape, block_size=(8, 8), threshold=0):
    """
    从带水印图像中提取水印
    
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
        # 对水印图像块和原始图像块执行SVD分解
        _, S_watermarked, _ = svd_decomposition(watermarked_blocks[i])
        _, S_original, _ = svd_decomposition(original_blocks[i])
        
        # 计算奇异值差异
        diff = S_watermarked[0] - S_original[0]
        
        # 根据差异判断水印位的值
        if diff > threshold:
            extracted_bits[i] = 1
        else:
            extracted_bits[i] = 0
    
    # 重塑为水印图像形状
    extracted_watermark = extracted_bits.reshape(watermark_shape)
    
    return extracted_watermark

def blind_extract_watermark(watermarked_img, watermark_shape, block_size=(8, 8), threshold=0):
    """
    在没有原始图像的情况下，从带水印图像中盲提取水印
    
    参数:
        watermarked_img: 带水印的图像
        watermark_shape: 水印图像的形状，如(64, 64)
        block_size: 图像块大小
        threshold: 提取水印时的阈值
        
    返回:
        提取的水印图像
    """
    # 获取图像块
    watermarked_blocks, positions, _ = get_blocks(watermarked_img, block_size)
    
    # 计算需要多少块来存储水印
    watermark_size = watermark_shape[0] * watermark_shape[1]
    if watermark_size > len(watermarked_blocks):
        raise ValueError("指定的水印尺寸过大，无法从图像中提取")
    
    # 存储提取的水印位
    extracted_bits = np.zeros(watermark_size, dtype=np.uint8)
    
    # 计算所有块的第一奇异值的均值，用于盲提取的参考
    s_values = []
    for i in range(watermark_size):
        _, S, _ = svd_decomposition(watermarked_blocks[i])
        s_values.append(S[0])
    
    s_mean = np.mean(s_values)
    
    # 对每个图像块提取水印位
    for i in range(watermark_size):
        _, S, _ = svd_decomposition(watermarked_blocks[i])
        
        # 根据奇异值是否高于平均值来判断水印位
        if S[0] > s_mean + threshold:
            extracted_bits[i] = 1
        else:
            extracted_bits[i] = 0
    
    # 重塑为水印图像形状
    extracted_watermark = extracted_bits.reshape(watermark_shape)
    
    return extracted_watermark