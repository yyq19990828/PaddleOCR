#!/usr/bin/env python3
"""
测试FilterByImageWidth的统计功能
"""

import numpy as np

# 模拟logger
class MockLogger:
    def info(self, msg):
        print(f"[INFO] {msg}")

class FilterByImageWidth(object):
    """根据图像宽度过滤数据样本，并收集过滤统计信息"""

    def __init__(self, width_range=None, **kwargs):
        self.width_range = width_range
        self.logger = MockLogger()  # 使用模拟logger
        
        # 统计计数器
        self.total_count = 0
        self.filtered_count = 0
        
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
        
        # 增加总计数
        self.total_count += 1
        
        # 获取图片宽度
        img_width = img.shape[1]
        
        # 检查宽度是否在范围内
        if not self._is_width_in_range(img_width):
            self.filtered_count += 1  # 增加过滤计数
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
    
    def get_statistics(self):
        """获取过滤统计信息"""
        kept_count = self.total_count - self.filtered_count
        filter_rate = (self.filtered_count / self.total_count * 100.0) if self.total_count > 0 else 0.0
        keep_rate = (kept_count / self.total_count * 100.0) if self.total_count > 0 else 0.0
        
        return {
            'total_count': self.total_count,
            'filtered_count': self.filtered_count,
            'kept_count': kept_count,
            'filter_rate': filter_rate,
            'keep_rate': keep_rate,
            'width_range': self.width_range
        }
    
    def print_statistics(self):
        """打印过滤统计信息"""
        if self.width_range is None:
            self.logger.info("FilterByImageWidth: No filtering applied (width_range=None)")
            return
            
        stats = self.get_statistics()
        
        self.logger.info("FilterByImageWidth Statistics:")
        self.logger.info("- Total images: {}".format(stats['total_count']))
        self.logger.info("- Filtered out: {} ({:.1f}%)".format(stats['filtered_count'], stats['filter_rate']))
        self.logger.info("- Kept images: {} ({:.1f}%)".format(stats['kept_count'], stats['keep_rate']))
        self.logger.info("- Width range: {}".format(stats['width_range']))


def test_statistics():
    """测试统计功能"""
    print("=== FilterByImageWidth 统计功能测试 ===")
    
    # 创建测试数据
    test_images = [
        np.zeros((100, 30, 3), dtype=np.uint8),   # 宽度30
        np.zeros((100, 50, 3), dtype=np.uint8),   # 宽度50  
        np.zeros((100, 80, 3), dtype=np.uint8),   # 宽度80
        np.zeros((100, 100, 3), dtype=np.uint8),  # 宽度100
        np.zeros((100, 150, 3), dtype=np.uint8),  # 宽度150
        np.zeros((100, 200, 3), dtype=np.uint8),  # 宽度200
    ]
    
    print(f"\n--- 测试配置: [60, 120] ---")
    
    # 创建过滤器
    filter_op = FilterByImageWidth(width_range=[60, 120])
    
    # 处理所有图像
    results = []
    for i, img in enumerate(test_images):
        data = {"image": img}
        result = filter_op(data)
        kept = result is not None
        results.append(kept)
        print(f"图像{i+1} (宽度{img.shape[1]}): {'保留' if kept else '过滤'}")
    
    # 验证统计信息
    stats = filter_op.get_statistics()
    expected_kept = sum(results)
    expected_filtered = len(results) - expected_kept
    
    print(f"\n--- 统计验证 ---")
    print(f"总图像数: {stats['total_count']} (期望: {len(test_images)})")
    print(f"过滤图像数: {stats['filtered_count']} (期望: {expected_filtered})")
    print(f"保留图像数: {stats['kept_count']} (期望: {expected_kept})")
    print(f"过滤率: {stats['filter_rate']:.1f}%")
    print(f"保留率: {stats['keep_rate']:.1f}%")
    
    # 打印统计信息
    print(f"\n--- 格式化输出 ---")
    filter_op.print_statistics()
    
    # 验证结果
    all_correct = (
        stats['total_count'] == len(test_images) and
        stats['filtered_count'] == expected_filtered and
        stats['kept_count'] == expected_kept
    )
    
    print(f"\n=== 统计功能测试{'通过' if all_correct else '失败'} ===")
    
    return all_correct


if __name__ == "__main__":
    test_statistics()