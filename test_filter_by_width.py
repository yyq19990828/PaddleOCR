#!/usr/bin/env python3
"""
测试FilterByImageWidth操作符的实现
"""

import numpy as np
import sys
import os

# 添加PaddleOCR路径到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ppocr.data.imaug.operators import FilterByImageWidth


def test_filter_by_image_width():
    """测试FilterByImageWidth操作符的各种配置"""
    
    print("=== 测试FilterByImageWidth操作符 ===")
    
    # 创建测试图像（不同宽度）
    test_images = {
        "narrow": np.zeros((100, 50, 3), dtype=np.uint8),     # 宽度50
        "medium": np.zeros((100, 200, 3), dtype=np.uint8),    # 宽度200  
        "wide": np.zeros((100, 500, 3), dtype=np.uint8),      # 宽度500
    }
    
    test_cases = [
        {
            "name": "无过滤 (None)",
            "config": None,
            "expected": {"narrow": True, "medium": True, "wide": True}
        },
        {
            "name": "闭区间过滤 [100, 300]",
            "config": [100, 300],
            "expected": {"narrow": False, "medium": True, "wide": False}
        },
        {
            "name": "开区间过滤 [150, ]",
            "config": [150, None],
            "expected": {"narrow": False, "medium": True, "wide": True}
        },
        {
            "name": "最大宽度限制 [, 250]",
            "config": [None, 250],
            "expected": {"narrow": True, "medium": True, "wide": False}
        }
    ]
    
    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")
        
        try:
            # 创建过滤器实例
            filter_op = FilterByImageWidth(width_range=test_case['config'])
            print(f"配置: {test_case['config']}")
            
            # 测试每个图像
            for img_name, img_data in test_images.items():
                data = {"image": img_data}
                result = filter_op(data)
                
                actual_pass = result is not None
                expected_pass = test_case['expected'][img_name]
                
                status = "✓" if actual_pass == expected_pass else "✗"
                print(f"  {img_name} (宽度{img_data.shape[1]}): {status} 期望={expected_pass}, 实际={actual_pass}")
                
                if actual_pass != expected_pass:
                    print(f"    错误: 期望{expected_pass}，实际{actual_pass}")
                    
        except Exception as e:
            print(f"  错误: {e}")
    
    # 测试错误情况
    print(f"\n--- 测试参数验证 ---")
    error_cases = [
        {"config": "invalid", "desc": "无效类型"},
        {"config": [100], "desc": "长度不足"},  
        {"config": [100, 50], "desc": "最小值大于最大值"},
        {"config": [None, None], "desc": "两个值都为None"},
    ]
    
    for case in error_cases:
        try:
            FilterByImageWidth(width_range=case['config'])
            print(f"  {case['desc']}: ✗ 应该抛出异常")
        except Exception as e:
            print(f"  {case['desc']}: ✓ 正确抛出异常: {type(e).__name__}")
    
    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    test_filter_by_image_width()