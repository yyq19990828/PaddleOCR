#!/usr/bin/env python3
"""
直接测试FilterByImageWidth的核心逻辑
"""

import numpy as np


class FilterByImageWidth(object):
    """根据图像宽度过滤数据样本"""

    def __init__(self, width_range=None, **kwargs):
        self.width_range = width_range
        
        # 验证参数格式
        if self.width_range is not None:
            if not isinstance(self.width_range, (list, tuple)):
                raise ValueError("width_range must be a list or tuple, got {}".format(type(self.width_range)))
            
            if len(self.width_range) == 2:
                min_width, max_width = self.width_range
                if min_width is not None and max_width is not None:
                    if min_width > max_width:
                        raise ValueError("min_width ({}) should be less than or equal to max_width ({})".format(min_width, max_width))
                elif min_width is None and max_width is None:
                    raise ValueError("width_range cannot be [None, None]")
            else:
                raise ValueError("width_range should have exactly 2 elements, got {}".format(len(self.width_range)))

    def __call__(self, data):
        """执行宽度过滤"""
        # 如果未设置过滤条件，直接返回
        if self.width_range is None:
            return data
            
        img = data["image"]
        assert isinstance(img, np.ndarray), "invalid input 'img' in FilterByImageWidth, expected numpy array, got {}".format(type(img))
        
        # 获取图片宽度
        img_width = img.shape[1]
        
        # 检查宽度是否在范围内
        if not self._is_width_in_range(img_width):
            return None  # 不符合条件，返回None触发数据跳过
            
        return data

    def _is_width_in_range(self, width):
        """检查宽度是否在指定范围内"""
        min_width, max_width = self.width_range
        
        # 检查最小宽度条件
        if min_width is not None and width < min_width:
            return False
            
        # 检查最大宽度条件
        if max_width is not None and width > max_width:
            return False
            
        return True


def test_filter():
    """测试FilterByImageWidth功能"""
    
    print("=== FilterByImageWidth 功能测试 ===")
    
    # 创建测试图像
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
            "name": "开区间过滤 [150, None]",
            "config": [150, None],
            "expected": {"narrow": False, "medium": True, "wide": True}
        },
    ]
    
    all_passed = True
    
    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")
        
        try:
            filter_op = FilterByImageWidth(width_range=test_case['config'])
            
            for img_name, img_data in test_images.items():
                data = {"image": img_data}
                result = filter_op(data)
                
                actual_pass = result is not None
                expected_pass = test_case['expected'][img_name]
                
                status = "✓" if actual_pass == expected_pass else "✗"
                print(f"  {img_name} (宽度{img_data.shape[1]}): {status} 期望={expected_pass}, 实际={actual_pass}")
                
                if actual_pass != expected_pass:
                    all_passed = False
                    
        except Exception as e:
            print(f"  错误: {e}")
            all_passed = False
    
    # 测试错误情况
    print(f"\n--- 参数验证测试 ---")
    error_cases = [
        {"config": [100, 50], "desc": "最小值大于最大值"},
        {"config": [None, None], "desc": "两个值都为None"},
    ]
    
    for case in error_cases:
        try:
            FilterByImageWidth(width_range=case['config'])
            print(f"  {case['desc']}: ✗ 应该抛出异常")
            all_passed = False
        except Exception as e:
            print(f"  {case['desc']}: ✓ 正确抛出异常")
    
    print(f"\n=== 测试{'通过' if all_passed else '失败'} ===")
    return all_passed


if __name__ == "__main__":
    test_filter()