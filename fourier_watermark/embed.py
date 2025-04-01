import numpy as np
from utils import fft_transform, ifft_transform, get_blocks, reconstruct_from_blocks, get_middle_frequency_indices

def embed_watermark(img, watermark, block_size=(8, 8), alpha=0.1):
    """
    将水印嵌入到图像中（基于傅里叶变换）
    
    参数:
        img: 原始图像
        watermark: 水印图像（二值，0或1）
        block_size: 图像块大小
        alpha: 水印强度因子
        
    返回:
        嵌入水印后的图像
    """
    # 获取图像块
    blocks, positions, original_shape = get_blocks(img, block_size)
    watermarked_blocks = []
    
    # 确保水印尺寸足够
    watermark_h, watermark_w = watermark.shape
    watermark_size = watermark_h * watermark_w
    if watermark_size > len(blocks):
        raise ValueError("水印尺寸过大，无法嵌入到图像中")
    
    # 扁平化水印为一维数组
    watermark_bits = watermark.flatten()
    
    # 获取中频区域索引
    mid_freq_indices = get_middle_frequency_indices(block_size)
    
    # 对每个图像块嵌入水印位
    for i, block in enumerate(blocks):
        if i < watermark_size:
            # 获取当前水印位值 (0 或 1)
            bit = watermark_bits[i]
            
            # 对图像块执行FFT变换
            fft_coeffs = fft_transform(block)
            
            # 获取用于嵌入水印的中频位置
            embed_pos = mid_freq_indices[i % len(mid_freq_indices)]
            
            # 修改频域系数幅值
            magnitude = np.abs(fft_coeffs[embed_pos])
            phase = np.angle(fft_coeffs[embed_pos])
            
            if bit == 1:
                # 增加幅值
                new_magnitude = magnitude * (1 + alpha)
            else:
                # 减小幅值
                new_magnitude = magnitude * (1 - alpha)
            
            # 根据修改后的幅值重建复数形式的频域系数
            fft_coeffs[embed_pos] = new_magnitude * np.exp(1j * phase)
            
            # 执行傅里叶逆变换，重建图像块
            modified_block = ifft_transform(fft_coeffs)
            
            # 对结果进行裁剪，确保像素值在有效范围内
            modified_block = np.clip(modified_block, 0, 255)
            
            watermarked_blocks.append(modified_block)
        else:
            # 对于未用于嵌入水印的块，保持不变
            watermarked_blocks.append(block)
    
    # 从修改后的块重建图像
    watermarked_img = reconstruct_from_blocks(watermarked_blocks, positions, original_shape)
    
    return watermarked_img.astype(np.uint8)

def embed_watermark_dct_like(img, watermark, block_size=(8, 8), alpha=0.1):
    """
    将水印嵌入到图像中（DCT风格的FFT实现）
    
    参数:
        img: 原始图像
        watermark: 水印图像（二值，0或1）
        block_size: 图像块大小
        alpha: 水印强度因子
        
    返回:
        嵌入水印后的图像
    """
    # 获取图像块
    blocks, positions, original_shape = get_blocks(img, block_size)
    watermarked_blocks = []
    
    # 确保水印尺寸足够
    watermark_h, watermark_w = watermark.shape
    watermark_size = watermark_h * watermark_w
    if watermark_size > len(blocks):
        raise ValueError("水印尺寸过大，无法嵌入到图像中")
    
    # 扁平化水印为一维数组
    watermark_bits = watermark.flatten()
    
    # 对每个图像块嵌入水印位
    for i, block in enumerate(blocks):
        if i < watermark_size:
            # 获取当前水印位值 (0 或 1)
            bit = watermark_bits[i]
            
            # 对图像块执行FFT变换
            fft_coeffs = fft_transform(block)
            
            # 选择嵌入的系数位置（模拟DCT中间频率系数）
            bh, bw = block_size
            embed_row, embed_col = bh//2 + 1, bw//2 + 1  # 中频位置
            
            # 修改频域系数
            magnitude = np.abs(fft_coeffs[embed_row, embed_col])
            phase = np.angle(fft_coeffs[embed_row, embed_col])
            
            if bit == 1:
                # 增加幅值
                new_magnitude = magnitude * (1 + alpha)
            else:
                # 减小幅值
                new_magnitude = magnitude * (1 - alpha)
            
            # 根据修改后的幅值重建复数形式的频域系数
            fft_coeffs[embed_row, embed_col] = new_magnitude * np.exp(1j * phase)
            
            # 由于傅里叶变换的对称性，同时修改对称位置的系数
            fft_coeffs[bh-embed_row, bw-embed_col] = np.conj(fft_coeffs[embed_row, embed_col])
            
            # 执行傅里叶逆变换，重建图像块
            modified_block = ifft_transform(fft_coeffs)
            
            # 对结果进行裁剪，确保像素值在有效范围内
            modified_block = np.clip(modified_block, 0, 255)
            
            watermarked_blocks.append(modified_block)
        else:
            # 对于未用于嵌入水印的块，保持不变
            watermarked_blocks.append(block)
    
    # 从修改后的块重建图像
    watermarked_img = reconstruct_from_blocks(watermarked_blocks, positions, original_shape)
    
    return watermarked_img.astype(np.uint8)