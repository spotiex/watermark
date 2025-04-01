import numpy as np
import cv2
from scipy import linalg

def read_image(file_path, as_gray=False):
    """
    读取图像文件
    
    参数:
        file_path: 图像文件路径
        as_gray: 是否以灰度模式读取
        
    返回:
        图像数据，如果as_gray为True则返回灰度图像
    """
    if as_gray:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(file_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if img is None:
        raise FileNotFoundError(f"无法读取图像文件: {file_path}")
    
    return img

def save_image(img, file_path):
    """
    保存图像到文件
    
    参数:
        img: 图像数据
        file_path: 保存路径
    
    返回:
        保存成功返回True，否则返回False
    """
    try:
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_to_save = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_to_save = img
        
        # 确保像素值在0-255范围内
        if img_to_save.dtype != np.uint8:
            img_to_save = np.clip(img_to_save, 0, 255).astype(np.uint8)
            
        return cv2.imwrite(file_path, img_to_save)
    except Exception as e:
        print(f"保存图像失败: {e}")
        return False

def svd_decomposition(img_block):
    """
    对图像块进行SVD分解
    
    参数:
        img_block: 图像块数据
        
    返回:
        U, S, V: SVD分解结果
    """
    U, S, Vh = linalg.svd(img_block.astype(np.float64))
    return U, S, Vh

def get_blocks(img, block_size):
    """
    将图像分割成不重叠的块
    
    参数:
        img: 图像数据
        block_size: 块大小，如(8, 8)
        
    返回:
        blocks: 图像块列表
        shape: 原始图像形状
    """
    # 检查是否为彩色图像
    if len(img.shape) == 3:
        h, w, _ = img.shape
        is_color = True
    else:
        h, w = img.shape
        is_color = False
        
    bh, bw = block_size
    
    # 计算需要的块数量
    num_blocks_h = h // bh
    num_blocks_w = w // bw
    
    # 如果图像尺寸不能被块大小整除，则裁剪图像
    if h % bh != 0 or w % bw != 0:
        if is_color:
            img = img[:num_blocks_h * bh, :num_blocks_w * bw, :]
            h, w, _ = img.shape
        else:
            img = img[:num_blocks_h * bh, :num_blocks_w * bw]
            h, w = img.shape
    
    # 转换彩色图像为灰度进行SVD处理
    if is_color:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img
    
    blocks = []
    positions = []
    
    for i in range(0, h, bh):
        for j in range(0, w, bw):
            block = img_gray[i:i+bh, j:j+bw]
            blocks.append(block)
            positions.append((i, j))
    
    return blocks, positions, (h, w, img)

def reconstruct_from_blocks(blocks, positions, original_shape):
    """
    从图像块重建图像
    
    参数:
        blocks: 图像块列表
        positions: 每个块的位置
        original_shape: 原始图像形状，包含原始彩色图像
        
    返回:
        重建后的图像
    """
    h, w, original_img = original_shape
    
    # 检查原始图像是否为彩色
    is_color = len(original_img.shape) == 3
    
    if is_color:
        # 创建灰度图像副本
        img_gray = np.zeros((h, w), dtype=blocks[0].dtype)
        
        # 重建灰度图像
        for block, (i, j) in zip(blocks, positions):
            bh, bw = block.shape
            img_gray[i:i+bh, j:j+bw] = block
        
        # 复制原始彩色图像
        result_img = original_img.copy()
        
        # 计算原始灰度图与重建灰度图的差异
        original_gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
        diff = img_gray.astype(np.float32) - original_gray.astype(np.float32)
        
        # 将差异添加到各个通道
        for c in range(3):
            result_img[:,:,c] = np.clip(result_img[:,:,c] + diff, 0, 255).astype(np.uint8)
        
        return result_img
    else:
        # 对于灰度图像，直接重建
        img = np.zeros((h, w), dtype=blocks[0].dtype)
        
        for block, (i, j) in zip(blocks, positions):
            bh, bw = block.shape
            img[i:i+bh, j:j+bw] = block
        
        return img

def preprocess_watermark(watermark, target_shape):
    """
    预处理水印图像，调整大小并转为二值图像
    
    参数:
        watermark: 水印图像
        target_shape: 目标形状，如(64, 64)
        
    返回:
        处理后的水印
    """
    # 调整水印大小
    watermark_resized = cv2.resize(watermark, target_shape)
    
    # 转换为二值图像 (0 或 1)
    _, watermark_binary = cv2.threshold(watermark_resized, 127, 1, cv2.THRESH_BINARY)
    
    return watermark_binary