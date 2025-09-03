#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
创建测试图片数据集，包含不同宽度的图片用于测试FilterByImageWidth功能
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import string

def generate_random_text(length=5):
    """生成随机文本"""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

def create_test_image(width, height=32, text=None):
    """
    创建指定宽度的测试图片
    
    Args:
        width: 图片宽度
        height: 图片高度，默认32
        text: 图片中的文本内容
    
    Returns:
        PIL Image对象
    """
    if text is None:
        text = generate_random_text()
    
    # 创建白色背景图片
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    # 尝试使用系统字体，如果没有则使用默认字体
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf', 16)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    # 计算文本位置，使其居中
    if font:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    else:
        # 估算文本大小
        text_width = len(text) * 8
        text_height = 12
    
    x = max(0, (width - text_width) // 2)
    y = max(0, (height - text_height) // 2)
    
    # 绘制文本
    draw.text((x, y), text, fill='black', font=font)
    
    return img

def create_test_dataset():
    """创建完整的测试数据集"""
    test_data_dir = './test_data'
    os.makedirs(test_data_dir, exist_ok=True)
    
    test_cases = []
    
    # 创建小图片（应被过滤掉）- 宽度20-49
    print("创建小图片（宽度20-49，应被过滤）...")
    for i in range(5):
        width = random.randint(20, 49)
        text = generate_random_text()
        img = create_test_image(width, text=text)
        filename = f'small_{width}w_{i+1}.jpg'
        filepath = os.path.join(test_data_dir, filename)
        img.save(filepath)
        test_cases.append((filename, text))
        print(f"  创建 {filename} (宽度: {width}px)")
    
    # 创建目标图片（应保留）- 宽度50-200
    print("\\n创建目标图片（宽度50-200，应保留）...")
    for i in range(10):
        width = random.randint(50, 200)
        text = generate_random_text()
        img = create_test_image(width, text=text)
        filename = f'target_{width}w_{i+1}.jpg'
        filepath = os.path.join(test_data_dir, filename)
        img.save(filepath)
        test_cases.append((filename, text))
        print(f"  创建 {filename} (宽度: {width}px)")
    
    # 创建大图片（应被过滤掉）- 宽度201-500
    print("\\n创建大图片（宽度201-500，应被过滤）...")
    for i in range(5):
        width = random.randint(201, 500)
        text = generate_random_text()
        img = create_test_image(width, text=text)
        filename = f'large_{width}w_{i+1}.jpg'
        filepath = os.path.join(test_data_dir, filename)
        img.save(filepath)
        test_cases.append((filename, text))
        print(f"  创建 {filename} (宽度: {width}px)")
    
    # 创建标注文件
    print("\\n创建标注文件...")
    label_file = os.path.join(test_data_dir, 'test_list.txt')
    with open(label_file, 'w', encoding='utf-8') as f:
        for filename, text in test_cases:
            f.write(f'{filename}\\t{text}\\n')
    
    print(f"\\n测试数据集创建完成！")
    print(f"- 总图片数: {len(test_cases)}")
    print(f"- 小图片: 5张 (宽度20-49, 应被过滤)")
    print(f"- 目标图片: 10张 (宽度50-200, 应保留)")  
    print(f"- 大图片: 5张 (宽度201-500, 应被过滤)")
    print(f"- 标注文件: {label_file}")
    
    return test_cases

if __name__ == '__main__':
    create_test_dataset()