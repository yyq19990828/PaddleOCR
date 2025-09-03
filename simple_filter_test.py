#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简单的FilterByImageWidth功能直接测试
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ppocr.data.imaug.operators import FilterByImageWidth


def test_filter_directly():
    """直接测试FilterByImageWidth操作符"""
    print("="*60)
    print("FilterByImageWidth 直接功能测试")
    print("="*60)
    
    # 创建过滤器实例
    filter_op = FilterByImageWidth(width_range=[50, 200])
    print(f"创建过滤器: width_range={filter_op.width_range}")
    
    # 测试图片列表 - 读取我们创建的测试图片
    test_images = [
        'small_20w_2.jpg',   # 宽度20，应被过滤
        'small_42w_1.jpg',   # 宽度42，应被过滤
        'target_84w_1.jpg',  # 宽度84，应保留
        'target_160w_2.jpg', # 宽度160，应保留
        'target_197w_7.jpg', # 宽度197，应保留
        'large_242w_1.jpg',  # 宽度242，应被过滤
        'large_427w_4.jpg',  # 宽度427，应被过滤
    ]
    
    results = []
    
    print("\\n开始测试每张图片:")
    for img_name in test_images:
        img_path = os.path.join('./test_data', img_name)
        if not os.path.exists(img_path):
            print(f"  跳过 {img_name}: 文件不存在")
            continue
        
        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            print(f"  跳过 {img_name}: 无法读取图片")
            continue
        
        height, width, channels = img.shape
        print(f"  测试 {img_name}: 尺寸 {width}x{height}")
        
        # 构造数据格式 (模拟ppocr的数据格式)
        data = {
            'image': img,
            'img_path': img_path,
        }
        
        # 应用过滤器
        result = filter_op(data)
        
        if result is None:
            print(f"    ✗ 被过滤 (宽度 {width} 不在范围 {filter_op.width_range})")
            results.append((img_name, width, False, "过滤"))
        else:
            print(f"    ✓ 保留 (宽度 {width} 在范围 {filter_op.width_range})")
            results.append((img_name, width, True, "保留"))
    
    # 输出统计信息
    print("\\n过滤器统计信息:")
    filter_op.print_statistics()
    
    stats = filter_op.get_statistics()
    print("\\n详细统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 验证结果
    print("\\n结果验证:")
    expected_keep = 0
    expected_filter = 0
    actual_keep = 0
    actual_filter = 0
    
    for img_name, width, kept, action in results:
        # 预期结果
        should_keep = 50 <= width <= 200
        if should_keep:
            expected_keep += 1
        else:
            expected_filter += 1
        
        # 实际结果
        if kept:
            actual_keep += 1
        else:
            actual_filter += 1
        
        # 验证
        correct = (kept == should_keep)
        status = "✓" if correct else "✗"
        print(f"  {status} {img_name}: 宽度{width}, 预期{['过滤','保留'][should_keep]}, 实际{action}")
    
    print(f"\\n总体验证:")
    print(f"  预期保留: {expected_keep}, 实际保留: {actual_keep}")
    print(f"  预期过滤: {expected_filter}, 实际过滤: {actual_filter}")
    print(f"  统计总数: {stats['total_count']}, 测试图片数: {len(results)}")
    
    all_correct = (expected_keep == actual_keep and expected_filter == actual_filter and stats['total_count'] == len(results))
    print(f"  结果: {'✓ 测试通过' if all_correct else '✗ 测试失败'}")
    
    return all_correct, stats

def test_different_ranges():
    """测试不同宽度范围的过滤效果"""
    print("\\n" + "="*60)
    print("不同宽度范围测试")
    print("="*60)
    
    test_ranges = [
        [20, 50],    # 只保留小图
        [50, 200],   # 保留目标图
        [200, 500],  # 只保留大图
        None         # 不过滤
    ]
    
    test_image = 'target_84w_1.jpg'  # 宽度84的测试图
    img_path = os.path.join('./test_data', test_image)
    
    if not os.path.exists(img_path):
        print(f"测试图片 {test_image} 不存在")
        return {}
    
    img = cv2.imread(img_path)
    height, width, channels = img.shape
    print(f"使用测试图片: {test_image} (宽度: {width})")
    
    results = {}
    
    for width_range in test_ranges:
        print(f"\\n测试范围: {width_range}")
        
        if width_range is None:
            print("  无过滤 - 图片应该保留")
            results['no_filter'] = True
            continue
        
        # 创建过滤器
        filter_op = FilterByImageWidth(width_range=width_range)
        
        # 测试图片
        data = {'image': img, 'img_path': img_path}
        result = filter_op(data)
        
        kept = result is not None
        should_keep = width_range[0] <= width <= width_range[1]
        
        correct = (kept == should_keep)
        status = "✓" if correct else "✗"
        action = "保留" if kept else "过滤"
        expected = "保留" if should_keep else "过滤"
        
        print(f"  {status} 结果: {action} (预期: {expected})")
        results[str(width_range)] = correct
    
    return results

if __name__ == '__main__':
    try:
        print("开始FilterByImageWidth直接测试...")
        
        # 1. 直接功能测试
        success1, stats = test_filter_directly()
        
        # 2. 不同范围测试
        success2_results = test_different_ranges()
        success2 = all(success2_results.values())
        
        # 3. 总结
        print("\\n" + "="*60)
        print("测试总结")
        print("="*60)
        
        print(f"基础功能测试: {'✓ 通过' if success1 else '✗ 失败'}")
        print(f"不同范围测试: {'✓ 通过' if success2 else '✗ 失败'}")
        
        if success1:
            print(f"\\n过滤统计信息验证:")
            print(f"  - 总图片数: {stats['total_count']}")
            print(f"  - 过滤掉: {stats['filtered_count']} ({stats['filter_rate']:.1f}%)")
            print(f"  - 保留: {stats['kept_count']} ({stats['keep_rate']:.1f}%)")
        
        overall_success = success1 and success2
        print(f"\\n整体测试结果: {'✓ 所有测试通过' if overall_success else '✗ 部分测试失败'}")
        
        if overall_success:
            print("\\nFilterByImageWidth操作符工作正常！")
        else:
            print("\\nFilterByImageWidth操作符存在问题，需要进一步调试。")
        
    except Exception as e:
        print(f"\\n测试执行出错: {e}")
        import traceback
        traceback.print_exc()