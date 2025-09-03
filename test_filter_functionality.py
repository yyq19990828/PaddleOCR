#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试FilterByImageWidth功能的独立脚本
"""

import os
import sys
import json
import time

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ppocr.data import build_dataloader
from ppocr.utils import logging
import yaml

def load_config(file_path):
    """加载配置文件"""
    with open(file_path, 'rb') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    return config

def test_filter_by_image_width():
    """测试FilterByImageWidth过滤功能"""
    print("="*60)
    print("FilterByImageWidth 集成测试")
    print("="*60)
    
    # 加载配置
    config_path = 'configs/test/test_filter_by_width.yml'
    config = load_config(config_path)
    
    print(f"\\n加载配置文件: {config_path}")
    
    # 构建数据加载器
    print("\\n构建测试数据加载器...")
    logger = logging.get_logger()
    eval_loader = build_dataloader(config, 'Eval', 'cpu', logger)
    
    print(f"数据加载器构建完成，预期批次数: {len(eval_loader)}")
    
    # 遍历数据以触发过滤
    print("\\n开始数据遍历和过滤...")
    total_samples = 0
    start_time = time.time()
    
    try:
        for batch_idx, batch in enumerate(eval_loader):
            batch_size = len(batch[0])  # batch[0] 是图片数据
            total_samples += batch_size
            print(f"批次 {batch_idx + 1}: 包含 {batch_size} 个样本")
            
            if batch_idx >= 2:  # 只处理前3个批次用于测试
                break
                
    except Exception as e:
        print(f"数据遍历出错: {e}")
        return
    
    end_time = time.time()
    
    print(f"\\n遍历完成!")
    print(f"- 处理批次数: {batch_idx + 1}")
    print(f"- 总样本数: {total_samples}")
    print(f"- 处理时间: {end_time - start_time:.2f}秒")
    
    # 检查过滤统计 - 查找过滤器实例
    print("\\n查找FilterByImageWidth统计信息...")
    
    # 尝试从数据加载器中获取统计信息
    dataset = eval_loader.dataset
    transforms = getattr(dataset, 'ops', [])
    
    filter_op = None
    for op in transforms:
        if hasattr(op, '__class__') and 'FilterByImageWidth' in op.__class__.__name__:
            filter_op = op
            break
    
    if filter_op:
        print("\\n找到FilterByImageWidth操作符!")
        # 输出统计信息
        filter_op.print_statistics()
        
        # 获取详细统计
        stats = filter_op.get_statistics()
        print("\\n详细统计信息:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        return stats
    else:
        print("警告: 未找到FilterByImageWidth操作符")
        return None

def test_different_width_ranges():
    """测试不同width_range配置的效果"""
    print("\\n" + "="*60)
    print("测试不同width_range配置")
    print("="*60)
    
    # 测试配置
    test_configs = [
        [20, 50],    # 只保留小图
        [50, 200],   # 默认配置，保留目标图
        [200, 500],  # 只保留大图
        None         # 不过滤
    ]
    
    results = {}
    
    for width_range in test_configs:
        print(f"\\n测试配置: width_range = {width_range}")
        
        # 动态修改配置
        config_path = 'configs/test/test_filter_by_width.yml'
        config = load_config(config_path)
        
        if width_range is None:
            # 移除过滤器
            transforms = config['Eval']['dataset']['transforms']
            config['Eval']['dataset']['transforms'] = [
                t for t in transforms if 'FilterByImageWidth' not in t
            ]
        else:
            # 修改过滤范围
            for transform in config['Eval']['dataset']['transforms']:
                if 'FilterByImageWidth' in transform:
                    transform['FilterByImageWidth']['width_range'] = width_range
        
        # 构建数据加载器
        try:
            logger = logging.get_logger()
            eval_loader = build_dataloader(config, 'Eval', 'cpu', logger)
            
            # 快速遍历以获取统计
            total_samples = 0
            for batch_idx, batch in enumerate(eval_loader):
                total_samples += len(batch[0])
                if batch_idx >= 1:  # 只处理前2个批次
                    break
            
            # 获取统计信息
            if width_range is not None:
                dataset = eval_loader.dataset
                transforms = getattr(dataset, 'ops', [])
                
                for op in transforms:
                    if hasattr(op, '__class__') and 'FilterByImageWidth' in op.__class__.__name__:
                        stats = op.get_statistics()
                        results[str(width_range)] = stats
                        print(f"  结果: 保留 {stats['kept_count']}/{stats['total_count']} 张图片 ({stats['keep_rate']:.1f}%)")
                        break
            else:
                results['no_filter'] = {'kept_count': total_samples, 'total_count': total_samples}
                print(f"  结果: 保留 {total_samples} 张图片 (无过滤)")
                
        except Exception as e:
            print(f"  错误: {e}")
            results[str(width_range)] = None
    
    return results

def performance_test():
    """性能影响测试"""
    print("\\n" + "="*60)
    print("性能影响测试")
    print("="*60)
    
    # 测试不过滤的情况
    config_path = 'configs/test/test_filter_by_width.yml'
    config_no_filter = load_config(config_path)
    
    # 移除过滤器
    transforms = config_no_filter['Eval']['dataset']['transforms']
    config_no_filter['Eval']['dataset']['transforms'] = [
        t for t in transforms if 'FilterByImageWidth' not in t
    ]
    
    print("\\n测试无过滤的性能...")
    start_time = time.time()
    
    try:
        logger = logging.get_logger()
        eval_loader_no_filter = build_dataloader(config_no_filter, 'Eval', 'cpu', logger)
        total_no_filter = 0
        for batch_idx, batch in enumerate(eval_loader_no_filter):
            total_no_filter += len(batch[0])
            if batch_idx >= 2:
                break
    except Exception as e:
        print(f"无过滤测试出错: {e}")
        return None
    
    time_no_filter = time.time() - start_time
    
    # 测试有过滤的情况
    print("\\n测试有过滤的性能...")
    start_time = time.time()
    
    try:
        config_with_filter = load_config(config_path)
        logger = logging.get_logger()
        eval_loader_with_filter = build_dataloader(config_with_filter, 'Eval', 'cpu', logger)
        total_with_filter = 0
        for batch_idx, batch in enumerate(eval_loader_with_filter):
            total_with_filter += len(batch[0])
            if batch_idx >= 2:
                break
    except Exception as e:
        print(f"有过滤测试出错: {e}")
        return None
        
    time_with_filter = time.time() - start_time
    
    # 计算性能影响
    performance_impact = ((time_with_filter - time_no_filter) / time_no_filter) * 100 if time_no_filter > 0 else 0
    
    print(f"\\n性能测试结果:")
    print(f"  无过滤用时: {time_no_filter:.3f}秒 ({total_no_filter} 样本)")
    print(f"  有过滤用时: {time_with_filter:.3f}秒 ({total_with_filter} 样本)")
    print(f"  性能影响: {performance_impact:+.1f}%")
    
    performance_ok = abs(performance_impact) < 5.0
    print(f"  性能要求: {'✓ 通过' if performance_ok else '✗ 未通过'} (要求<5%)")
    
    return {
        'time_no_filter': time_no_filter,
        'time_with_filter': time_with_filter,
        'performance_impact': performance_impact,
        'performance_ok': performance_ok,
        'samples_no_filter': total_no_filter,
        'samples_with_filter': total_with_filter
    }

if __name__ == '__main__':
    # 设置日志级别
    logger = logging.get_logger()
    
    try:
        # 1. 基础功能测试
        print("开始集成测试...")
        basic_stats = test_filter_by_image_width()
        
        # 2. 不同配置测试
        range_results = test_different_width_ranges()
        
        # 3. 性能测试
        performance_results = performance_test()
        
        # 4. 生成测试报告
        print("\\n" + "="*60)
        print("集成测试完整报告")
        print("="*60)
        
        print("\\n1. 基础过滤功能:")
        if basic_stats:
            print(f"   ✓ FilterByImageWidth操作符工作正常")
            print(f"   ✓ 过滤范围: {basic_stats['width_range']}")
            print(f"   ✓ 总图片: {basic_stats['total_count']}")
            print(f"   ✓ 过滤掉: {basic_stats['filtered_count']} ({basic_stats['filter_rate']:.1f}%)")
            print(f"   ✓ 保留: {basic_stats['kept_count']} ({basic_stats['keep_rate']:.1f}%)")
        else:
            print("   ✗ 基础功能测试失败")
        
        print("\\n2. 不同配置测试:")
        for config, result in range_results.items():
            if result:
                print(f"   ✓ {config}: 保留 {result['kept_count']}/{result['total_count']} 张")
            else:
                print(f"   ✗ {config}: 测试失败")
        
        print("\\n3. 性能影响测试:")
        if performance_results:
            print(f"   ✓ 性能影响: {performance_results['performance_impact']:+.1f}%")
            if performance_results['performance_ok']:
                print("   ✓ 性能要求满足 (<5%)")
            else:
                print("   ✗ 性能要求不满足 (≥5%)")
        else:
            print("   ✗ 性能测试失败")
        
        # 验收标准检查
        print("\\n4. 验收标准检查:")
        checks = [
            ("测试配置文件创建完成", os.path.exists('configs/test/test_filter_by_width.yml')),
            ("测试数据集准备完成", os.path.exists('test_data/test_list.txt')),
            ("FilterByImageWidth成功加载", basic_stats is not None),
            ("统计信息正确输出", basic_stats is not None and 'total_count' in basic_stats),
            ("过滤效果验证正确", basic_stats is not None and basic_stats['filtered_count'] > 0),
            ("性能影响测试通过", performance_results is not None and performance_results['performance_ok']),
            ("多种配置场景测试完成", len(range_results) > 0)
        ]
        
        all_passed = True
        for check_name, passed in checks:
            status = "✓ 通过" if passed else "✗ 失败"
            print(f"   {status} {check_name}")
            if not passed:
                all_passed = False
        
        print(f"\\n总体结果: {'✓ 所有验收标准通过' if all_passed else '✗ 部分验收标准未通过'}")
        
        # 保存测试结果
        test_result = {
            'basic_stats': basic_stats,
            'range_results': range_results,
            'performance_results': performance_results,
            'validation_checks': dict(checks),
            'all_passed': all_passed,
            'timestamp': time.time()
        }
        
        with open('test_filter_results.json', 'w', encoding='utf-8') as f:
            json.dump(test_result, f, indent=2, ensure_ascii=False)
        
        print(f"\\n测试结果已保存到: test_filter_results.json")
        
    except Exception as e:
        print(f"\\n测试执行出错: {e}")
        import traceback
        traceback.print_exc()