import numpy as np
from utils import svd_decomposition, get_blocks, reconstruct_from_blocks

def embed_watermark(img, watermark, block_size=(8, 8), alpha=0.1):
    """
    将水印嵌入到图像中
    
    参数:
        img: 原始图像（灰度）
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
            
            # 对图像块执行SVD分解
            U, S, Vh = svd_decomposition(block)
            
            # 修改奇异值
            if bit == 1:
                # 增加第一个奇异值
                S[0] = S[0] * (1 + alpha)
            else:
                # 减小第一个奇异值
                S[0] = S[0] * (1 - alpha)
            
            # 重建修改后的图像块
            modified_block = np.dot(U, np.dot(np.diag(S), Vh))
            
            # 对结果进行裁剪，确保像素值在有效范围内
            modified_block = np.clip(modified_block, 0, 255)
            
            watermarked_blocks.append(modified_block)
        else:
            # 对于未用于嵌入水印的块，保持不变
            watermarked_blocks.append(block)
    
    # 从修改后的块重建图像
    watermarked_img = reconstruct_from_blocks(watermarked_blocks, positions, original_shape)
    
    return watermarked_img.astype(np.uint8)