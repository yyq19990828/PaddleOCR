#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FilterByImageWidth性能影响测试
"""

import os
import sys
import time
import cv2
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ppocr.data.imaug.operators import FilterByImageWidth


def performance_test():
    """测试过滤器的性能影响"""
    print("="*60)
    print("FilterByImageWidth 性能影响测试")
    print("="*60)
    
    # 收集测试图片
    test_images = []
    test_data_dir = './test_data'
    
    for filename in os.listdir(test_data_dir):
        if filename.endswith('.jpg'):
            img_path = os.path.join(test_data_dir, filename)
            img = cv2.imread(img_path)
            if img is not None:
                test_images.append({
                    'image': img,
                    'img_path': img_path,
                    'filename': filename
                })
    
    print(f"收集到 {len(test_images)} 张测试图片")
    
    # 重复测试图片以增加数据量
    repeated_images = test_images * 50  # 重复50次
    print(f"扩展到 {len(repeated_images)} 张图片进行性能测试")
    
    # 测试1: 无过滤处理时间
    print("\\n测试1: 基准时间（无过滤处理）...")
    start_time = time.time()
    
    processed_count = 0
    for data in repeated_images:
        # 模拟一些基本处理（相当于其他transform操作的开销）
        img = data['image']
        height, width = img.shape[:2]
        # 简单的处理操作
        _ = img.copy()
        processed_count += 1
    
    baseline_time = time.time() - start_time
    print(f"基准时间: {baseline_time:.3f}秒 ({processed_count} 张图片)")
    
    # 测试2: 有过滤处理时间
    print("\\n测试2: 过滤处理时间...")
    filter_op = FilterByImageWidth(width_range=[50, 200])
    
    start_time = time.time()
    
    filtered_count = 0
    kept_count = 0
    
    for data in repeated_images:
        # 应用过滤器
        result = filter_op(data.copy())  # 使用copy避免修改原数据
        
        if result is None:
            filtered_count += 1
        else:
            kept_count += 1
            # 模拟相同的基本处理
            img = result['image']
            _ = img.copy()
    
    filter_time = time.time() - start_time
    print(f"过滤时间: {filter_time:.3f}秒")
    print(f"处理结果: 保留 {kept_count}, 过滤 {filtered_count}")
    
    # 计算性能影响
    if baseline_time > 0:
        performance_impact = ((filter_time - baseline_time) / baseline_time) * 100
    else:
        performance_impact = 0
    
    print(f"\\n性能分析:")
    print(f"  基准时间: {baseline_time:.3f}秒")
    print(f"  过滤时间: {filter_time:.3f}秒")
    print(f"  额外开销: {filter_time - baseline_time:.3f}秒")
    print(f"  性能影响: {performance_impact:+.1f}%")
    
    # 打印过滤器统计
    print("\\n过滤器统计:")
    filter_op.print_statistics()
    
    # 性能要求检查
    performance_ok = abs(performance_impact) < 5.0
    print(f"\\n性能要求检查:")
    print(f"  要求: <5% 性能影响")
    print(f"  实际: {performance_impact:+.1f}%")
    print(f"  结果: {'✓ 通过' if performance_ok else '✗ 未通过'}")
    
    return {
        'baseline_time': baseline_time,
        'filter_time': filter_time,
        'performance_impact': performance_impact,
        'performance_ok': performance_ok,
        'total_images': len(repeated_images),
        'kept_count': kept_count,
        'filtered_count': filtered_count
    }

def test_with_different_image_sizes():
    """测试不同图片尺寸下的性能影响"""
    print("\\n" + "="*60)
    print("不同图片尺寸性能测试")
    print("="*60)
    
    # 创建不同尺寸的测试图片
    test_sizes = [
        (50, 32),    # 小图
        (100, 32),   # 中图
        (200, 32),   # 大图
        (500, 32),   # 超大图
    ]
    
    results = {}
    
    for width, height in test_sizes:
        print(f"\\n测试尺寸: {width}x{height}")
        
        # 创建测试图片
        test_img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        test_data = {'image': test_img, 'img_path': f'test_{width}x{height}.jpg'}
        
        # 重复多次以获得稳定的时间测量
        num_repeats = 1000
        test_images = [test_data.copy() for _ in range(num_repeats)]
        
        # 测试无过滤
        start_time = time.time()
        for data in test_images:
            _ = data['image'].copy()
        baseline_time = time.time() - start_time
        
        # 测试有过滤
        filter_op = FilterByImageWidth(width_range=[75, 300])  # 设置一个范围
        start_time = time.time()
        
        for data in test_images:
            result = filter_op(data.copy())
            if result is not None:
                _ = result['image'].copy()
        
        filter_time = time.time() - start_time
        
        # 计算影响
        performance_impact = ((filter_time - baseline_time) / baseline_time) * 100 if baseline_time > 0 else 0
        
        print(f"  基准时间: {baseline_time:.3f}秒")
        print(f"  过滤时间: {filter_time:.3f}秒")
        print(f"  性能影响: {performance_impact:+.1f}%")
        
        results[f"{width}x{height}"] = {
            'baseline_time': baseline_time,
            'filter_time': filter_time,
            'performance_impact': performance_impact,
            'performance_ok': abs(performance_impact) < 5.0
        }
        
        # 重置过滤器统计（创建新实例）
        filter_op = FilterByImageWidth(width_range=[75, 300])
    
    return results

if __name__ == '__main__':
    try:
        print("开始FilterByImageWidth性能测试...")
        
        # 1. 基础性能测试
        perf_results = performance_test()
        
        # 2. 不同尺寸测试
        size_results = test_with_different_image_sizes()
        
        # 3. 总结
        print("\\n" + "="*60)
        print("性能测试总结")
        print("="*60)
        
        print(f"\\n基础性能测试:")
        print(f"  处理图片: {perf_results['total_images']} 张")
        print(f"  保留图片: {perf_results['kept_count']} 张")
        print(f"  过滤图片: {perf_results['filtered_count']} 张")
        print(f"  性能影响: {perf_results['performance_impact']:+.1f}%")
        print(f"  性能要求: {'✓ 满足' if perf_results['performance_ok'] else '✗ 不满足'}")
        
        print(f"\\n不同尺寸测试:")
        all_size_ok = True
        for size, result in size_results.items():
            status = "✓" if result['performance_ok'] else "✗"
            print(f"  {status} {size}: {result['performance_impact']:+.1f}% 影响")
            if not result['performance_ok']:
                all_size_ok = False
        
        overall_ok = perf_results['performance_ok'] and all_size_ok
        print(f"\\n总体性能评估: {'✓ 所有测试通过' if overall_ok else '✗ 部分测试未通过'}")
        
        if overall_ok:
            print("\\nFilterByImageWidth的性能影响在可接受范围内 (<5%)！")
        else:
            print("\\n注意: FilterByImageWidth的性能影响可能需要进一步优化。")
        
        # 保存结果
        import json
        results = {
            'basic_performance': perf_results,
            'size_performance': size_results,
            'overall_ok': overall_ok,
            'timestamp': time.time()
        }
        
        with open('performance_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\\n性能测试结果已保存到: performance_results.json")
        
    except Exception as e:
        print(f"\\n性能测试执行出错: {e}")
        import traceback
        traceback.print_exc()