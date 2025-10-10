#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车牌识别准确率统计脚本 - 增强版
支持多种统计模式和详细分析
"""

import os
import re
import json
import argparse
from collections import defaultdict, Counter
from typing import Tuple, Dict, List, Optional

try:
    from rapidfuzz.distance import Levenshtein
except ImportError:
    print("警告: rapidfuzz 未安装，将使用备用的编辑距离计算方法")
    import difflib
    
    class Levenshtein:
        @staticmethod
        def normalized_distance(s1, s2):
            """
            备用的归一化编辑距离计算
            
            编辑距离（Edit Distance）说明：
            - 编辑距离衡量两个字符串之间的相似度
            - 表示将一个字符串转换为另一个字符串需要的最少编辑操作次数
            - 操作包括：插入、删除、替换字符
            - 归一化编辑距离：编辑距离除以较长字符串的长度，范围[0,1]
            - 0表示完全相同，1表示完全不同
            - 该指标比完全匹配更宽松，能反映部分正确的情况
            """
            if not s1 and not s2:
                return 0.0
            if not s1 or not s2:
                return 1.0
            
            # 使用 difflib 计算相似度，然后转换为距离
            similarity = difflib.SequenceMatcher(None, s1, s2).ratio()
            return 1.0 - similarity


class PlateAccuracyAnalyzer:
    def __init__(self, confidence_threshold: float = 0.0):
        self.confidence_threshold = confidence_threshold
        self.results = []
        self.stats = {}
    
    def extract_plate_from_filename(self, filename: str) -> str:
        """从文件名中提取车牌号码"""
        basename = os.path.splitext(filename)[0]
        
        # 尝试多种方式提取车牌号
        import re
        
        # 匹配标准车牌格式：省份简称 + 字母 + 4-6位数字字母组合 + 可选的"挂"、"学"、"港"、"澳"等后缀
        plate_pattern = r'^([京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领][A-Z][A-Z0-9]{4,6}[挂学港澳]?)'
        
        # 方式1: 直接从文件名开头提取车牌号
        match = re.match(plate_pattern, basename)
        if match:
            return match.group(1)
        
        # 方式2: 处理分割的车牌格式
        parts = basename.split('_')
        if len(parts) >= 2:
            # 尝试重组车牌号
            potential_plates = []
            
            # 情况1: "冀A_YC79挂" -> "冀AYC79挂"
            if len(parts) >= 2:
                first_part = parts[0]  # "冀A"
                second_part = parts[1]  # "YC79挂"
                
                if (len(first_part) == 2 and
                    first_part[0] in '京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领' and
                    first_part[1].isalpha()):
                    potential_plates.append(first_part + second_part)
            
            # 情况2: "浙E_AB123_港" -> "浙EAB123港"
            if len(parts) >= 3:
                first_part = parts[0]   # "浙E"
                second_part = parts[1]  # "AB123"
                third_part = parts[2]   # "港"
                
                if (len(first_part) == 2 and
                    first_part[0] in '京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领' and
                    first_part[1].isalpha()):
                    # 检查第三部分是否是后缀
                    if third_part in ['挂', '学', '港', '澳']:
                        potential_plates.append(first_part + second_part + third_part)
                    else:
                        # 如果第三部分不是后缀，可能是其他信息，只合并前两部分
                        potential_plates.append(first_part + second_part)
            
            # 情况3: "川K_ABC12_挂" -> "川KABC12挂"
            if len(parts) >= 3:
                first_part = parts[0]   # "川K"
                second_part = parts[1]  # "ABC12"
                third_part = parts[2]   # "挂"
                
                if (len(first_part) == 2 and
                    first_part[0] in '京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领' and
                    first_part[1].isalpha() and
                    third_part in ['挂', '学', '港', '澳']):
                    potential_plates.append(first_part + second_part + third_part)
            
            # 验证重组的车牌号，选择最长的有效车牌
            valid_plates = []
            for plate in potential_plates:
                if re.match(plate_pattern, plate):
                    valid_plates.append(plate)
            
            if valid_plates:
                # 返回最长的有效车牌（通常包含更完整的信息）
                return max(valid_plates, key=len)
        
        # 方式3: 如果以上都失败，返回第一个下划线前的部分（兼容旧格式）
        return parts[0] if parts else basename
    
    def parse_result_file(self, result_file: str) -> bool:
        """解析识别结果文件"""
        self.results = []
        
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        parts = line.split('\t')
                        if len(parts) != 3:
                            print(f"警告: 第{line_num}行格式错误，跳过: {line}")
                            continue
                        
                        image_path = parts[0]
                        predicted_text = parts[1]
                        confidence = float(parts[2])
                        
                        filename = os.path.basename(image_path)
                        true_plate = self.extract_plate_from_filename(filename)
                        
                        self.results.append({
                            'filename': filename,
                            'true_plate': true_plate,
                            'predicted_plate': predicted_text,
                            'confidence': confidence,
                            'image_path': image_path
                        })
                        
                    except (ValueError, IndexError) as e:
                        print(f"警告: 第{line_num}行解析错误，跳过: {line}, 错误: {e}")
                        continue
            
            return True
            
        except FileNotFoundError:
            print(f"错误: 文件不存在 {result_file}")
            return False
        except Exception as e:
            print(f"错误: 读取文件失败 {e}")
            return False
    
    def calculate_basic_stats(self) -> Dict:
        """
        计算基本统计信息
        
        统计指标说明：
        1. 完全正确率（Accuracy）：完全匹配的样本占比，要求每个字符都正确
        2. 归一化编辑距离（Normalized Edit Distance）：平均编辑距离，范围[0,1]
        3. 编辑距离相似度：1 - 归一化编辑距离，范围[0,1]，越接近1越相似
        
        编辑距离的优势：
        - 能捕捉部分正确的识别结果（如"川A123B4"识别为"川A123B5"）
        - 比完全匹配更客观地评估模型性能
        - 对于车牌识别场景，1-2个字符错误仍有实用价值
        """
        total_count = 0
        correct_count = 0
        total_edit_distance = 0.0
        filtered_results = []
        
        stats = {
            'total': len(self.results),
            'correct': 0,
            'incorrect': 0,
            'filtered_by_confidence': 0,
            'accuracy': 0.0,  # 完全正确率
            'norm_edit_distance': 0.0,  # 平均归一化编辑距离 [0,1]，越小越好
            'norm_edit_distance_similarity': 0.0,  # 编辑距离相似度 [0,1]，越大越好
            'confidence_threshold': self.confidence_threshold,
            'filtered_total': 0,
            'incorrect_samples': [],
            'correct_samples': []
        }
        
        for result in self.results:
            if result['confidence'] < self.confidence_threshold:
                stats['filtered_by_confidence'] += 1
                continue
            
            total_count += 1
            filtered_results.append(result)
            
            true_plate = result['true_plate'].upper()
            predicted_plate = result['predicted_plate'].upper()
            
            # 计算归一化编辑距离
            # 例子：真实="川A123B4", 预测="川A123B5" -> 编辑距离=1/7≈0.143
            # 表示只有一个字符不同，相似度很高
            edit_distance = Levenshtein.normalized_distance(true_plate, predicted_plate)
            total_edit_distance += edit_distance
            
            # 完全匹配检查（传统准确率）
            if true_plate == predicted_plate:
                correct_count += 1
                stats['correct'] += 1
                stats['correct_samples'].append(result)
            else:
                stats['incorrect'] += 1
                stats['incorrect_samples'].append(result)
        
        if total_count > 0:
            stats['accuracy'] = correct_count / total_count
            stats['norm_edit_distance'] = total_edit_distance / total_count
            # 编辑距离相似度：1表示完全相同，0表示完全不同
            # 相比完全正确率，这个指标更能体现"接近正确"的程度
            stats['norm_edit_distance_similarity'] = 1 - (total_edit_distance / total_count)
        
        stats['filtered_total'] = total_count
        return stats
    
    def analyze_by_province(self) -> Dict:
        """按省份统计"""
        province_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'accuracy': 0.0, 'edit_distance': 0.0, 'edit_similarity': 0.0})
        
        for result in self.results:
            if result['confidence'] < self.confidence_threshold:
                continue
                
            # 提取省份简称
            true_plate = result['true_plate']
            if len(true_plate) > 0 and true_plate[0] in '京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领':
                province = true_plate[0]
            else:
                province = '其他'
            
            province_stats[province]['total'] += 1
            
            true_plate_upper = true_plate.upper()
            predicted_plate_upper = result['predicted_plate'].upper()
            
            # 计算编辑距离
            # 对于车牌识别，编辑距离能更好地反映实际可用性
            # 例如：川A123B4 vs 川A123B5 (编辑距离小，仍有参考价值)
            # 对比：川A123B4 vs X5Y9Z2 (编辑距离大，完全无用)
            edit_distance = Levenshtein.normalized_distance(true_plate_upper, predicted_plate_upper)
            province_stats[province]['edit_distance'] += edit_distance
            
            if true_plate_upper == predicted_plate_upper:
                province_stats[province]['correct'] += 1
        
        # 计算各省准确率和编辑距离相似度
        for province in province_stats:
            if province_stats[province]['total'] > 0:
                province_stats[province]['accuracy'] = province_stats[province]['correct'] / province_stats[province]['total']
                province_stats[province]['edit_similarity'] = 1 - (province_stats[province]['edit_distance'] / province_stats[province]['total'])
        
        return dict(province_stats)
    
    def analyze_by_confidence_range(self) -> Dict:
        """按置信度区间统计"""
        confidence_ranges = [
            (0.0, 0.5, '0.0-0.5'),
            (0.5, 0.7, '0.5-0.7'),
            (0.7, 0.8, '0.7-0.8'),
            (0.8, 0.9, '0.8-0.9'),
            (0.9, 0.95, '0.9-0.95'),
            (0.95, 1.0, '0.95-1.0')
        ]
        
        range_stats = {}
        
        for min_conf, max_conf, range_name in confidence_ranges:
            range_data = {'total': 0, 'correct': 0, 'accuracy': 0.0, 'edit_distance': 0.0, 'edit_similarity': 0.0}
            
            for result in self.results:
                if min_conf <= result['confidence'] < max_conf:
                    range_data['total'] += 1
                    
                    true_plate_upper = result['true_plate'].upper()
                    predicted_plate_upper = result['predicted_plate'].upper()
                    
                    # 计算编辑距离
                    edit_distance = Levenshtein.normalized_distance(true_plate_upper, predicted_plate_upper)
                    range_data['edit_distance'] += edit_distance
                    
                    if true_plate_upper == predicted_plate_upper:
                        range_data['correct'] += 1
            
            if range_data['total'] > 0:
                range_data['accuracy'] = range_data['correct'] / range_data['total']
                range_data['edit_similarity'] = 1 - (range_data['edit_distance'] / range_data['total'])
            
            range_stats[range_name] = range_data
        
        return range_stats
    
    def analyze_error_patterns(self) -> Dict:
        """分析错误模式"""
        if not hasattr(self, 'stats') or not self.stats.get('incorrect_samples'):
            return {}
        
        error_patterns = {
            'missing_province': [],
            'wrong_province': [],
            'character_substitution': [],
            'length_mismatch': [],
            'partial_recognition': [],
            'other': []
        }
        
        char_errors = Counter()  # 字符错误统计
        
        for sample in self.stats['incorrect_samples']:
            true_plate = sample['true_plate']
            pred_plate = sample['predicted_plate']
            
            # 长度不匹配
            if len(true_plate) != len(pred_plate):
                error_patterns['length_mismatch'].append(sample)
                continue
            
            # 检查省份简称问题
            if len(true_plate) > 0 and len(pred_plate) > 0:
                # 缺少省份简称
                if true_plate[0] in '京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领':
                    if pred_plate[0] not in '京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领':
                        error_patterns['missing_province'].append(sample)
                        continue
                    elif pred_plate[0] != true_plate[0]:
                        error_patterns['wrong_province'].append(sample)
                        continue
            
            # 字符替换错误
            if len(true_plate) == len(pred_plate):
                error_patterns['character_substitution'].append(sample)
                # 统计字符错误
                for i, (true_char, pred_char) in enumerate(zip(true_plate, pred_plate)):
                    if true_char != pred_char:
                        char_errors[f'{true_char}->{pred_char}'] += 1
            else:
                error_patterns['other'].append(sample)
        
        # 添加字符错误统计
        error_patterns['character_error_stats'] = dict(char_errors.most_common(20))
        
        return error_patterns
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """生成详细报告"""
        # 计算基本统计
        self.stats = self.calculate_basic_stats()
        
        # 各项分析
        province_stats = self.analyze_by_province()
        confidence_stats = self.analyze_by_confidence_range()
        error_patterns = self.analyze_error_patterns()
        
        # 生成报告
        report = []
        report.append("=" * 80)
        report.append("车牌识别准确率详细分析报告")
        report.append("=" * 80)
        
        # 基本统计
        report.append(f"总样本数量: {self.stats['total']}")
        report.append(f"置信度阈值: {self.stats['confidence_threshold']:.2f}")
        report.append(f"低于阈值被过滤: {self.stats['filtered_by_confidence']}")
        report.append(f"参与计算样本数: {self.stats['filtered_total']}")
        report.append(f"完全正确: {self.stats['correct']}")
        report.append(f"识别错误: {self.stats['incorrect']}")
        report.append(f"完全正确率: {self.stats['accuracy']:.4f} ({self.stats['accuracy']*100:.2f}%)")
        report.append(f"归一化编辑距离: {self.stats['norm_edit_distance']:.4f} (越小越好，0=完全相同)")
        report.append(f"编辑距离相似度: {self.stats['norm_edit_distance_similarity']:.4f} ({self.stats['norm_edit_distance_similarity']*100:.2f}%) (越大越好，1=完全相同)")
        report.append("")
        report.append("指标说明:")
        report.append("- 完全正确率：要求每个字符都完全匹配")
        report.append("- 编辑距离相似度：考虑部分正确的情况，更全面评估识别质量")
        report.append("- 编辑距离能反映识别结果的实用性（如川A123B4 vs 川A123B5仍有价值）")
        
        # 按省份统计
        if province_stats:
            report.append("\n" + "=" * 60)
            report.append("各省份识别准确率统计")
            report.append("=" * 60)
            report.append(f"{'省份':<6} {'总数':<8} {'正确':<8} {'准确率':<10} {'编辑相似度':<12}")
            report.append("-" * 55)
            
            # 按准确率排序
            sorted_provinces = sorted(province_stats.items(), key=lambda x: x[1]['accuracy'], reverse=True)
            for province, stats in sorted_provinces:
                if stats['total'] >= 5:  # 只显示样本数>=5的省份
                    report.append(f"{province:<6} {stats['total']:<8} {stats['correct']:<8} "
                                f"{stats['accuracy']*100:<8.2f}% {stats['edit_similarity']*100:<10.2f}%")
        
        # 按置信度区间统计
        report.append("\n" + "=" * 70)
        report.append("置信度区间准确率统计")
        report.append("=" * 70)
        report.append(f"{'置信度区间':<12} {'总数':<8} {'正确':<8} {'准确率':<10} {'编辑相似度':<12}")
        report.append("-" * 65)
        
        for range_name, stats in confidence_stats.items():
            if stats['total'] > 0:
                report.append(f"{range_name:<12} {stats['total']:<8} {stats['correct']:<8} "
                            f"{stats['accuracy']*100:<8.2f}% {stats['edit_similarity']*100:<10.2f}%")
        
        # 错误模式分析
        if error_patterns:
            report.append("\n" + "=" * 50)
            report.append("错误模式分析")
            report.append("=" * 50)
            
            for pattern_name, samples in error_patterns.items():
                if pattern_name == 'character_error_stats':
                    continue
                if isinstance(samples, list):
                    report.append(f"{self._get_pattern_name(pattern_name)}: {len(samples)}")
            
            # 常见字符错误
            if 'character_error_stats' in error_patterns and error_patterns['character_error_stats']:
                report.append("\n常见字符识别错误 (前10个):")
                report.append("-" * 30)
                for error, count in list(error_patterns['character_error_stats'].items())[:10]:
                    report.append(f"{error}: {count}次")
        
        # 部分错误样本
        if self.stats['incorrect_samples']:
            report.append(f"\n错误样本示例 (前10个):")
            report.append("-" * 70)
            report.append(f"{'真实车牌':<15} {'识别结果':<15} {'置信度':<10} {'文件名':<25}")
            report.append("-" * 70)
            
            for sample in self.stats['incorrect_samples'][:10]:
                report.append(f"{sample['true_plate']:<15} {sample['predicted_plate']:<15} "
                            f"{sample['confidence']:<10.4f} {sample['filename']:<25}")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"详细报告已保存到: {save_path}")
        
        return report_text
    
    def _get_pattern_name(self, pattern_key: str) -> str:
        """获取错误模式的中文名称"""
        pattern_names = {
            'missing_province': '缺少省份简称',
            'wrong_province': '省份简称错误',
            'character_substitution': '字符替换错误',
            'length_mismatch': '长度不匹配',
            'partial_recognition': '部分识别',
            'other': '其他错误'
        }
        return pattern_names.get(pattern_key, pattern_key)
    
    def save_error_samples(self, save_path: str):
        """
        保存错误样本到文件
        
        输出文件包含以下信息：
        - 真实车牌：标准答案
        - 识别结果：模型预测结果  
        - 置信度：模型对预测的信心程度
        - 编辑距离：归一化编辑距离，衡量错误程度
          * 0.0 = 完全正确
          * 0.1-0.3 = 轻微错误（1-2个字符差异）
          * 0.4-0.7 = 中等错误
          * 0.8-1.0 = 严重错误
        """
        if not hasattr(self, 'stats') or not self.stats.get('incorrect_samples'):
            print("没有错误样本可保存")
            return
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("真实车牌\t识别结果\t置信度\t编辑距离\t文件名\t图像路径\n")
            f.write("# 编辑距离说明：0.0=完全正确, 0.1-0.3=轻微错误, 0.4-0.7=中等错误, 0.8-1.0=严重错误\n")
            for sample in self.stats['incorrect_samples']:
                true_plate = sample['true_plate'].upper()
                predicted_plate = sample['predicted_plate'].upper()
                # 计算该样本的编辑距离，用于错误程度分析
                edit_distance = Levenshtein.normalized_distance(true_plate, predicted_plate)
                
                f.write(f"{sample['true_plate']}\t{sample['predicted_plate']}\t"
                       f"{sample['confidence']:.4f}\t{edit_distance:.4f}\t"
                       f"{sample['filename']}\t{sample['image_path']}\n")
        
        print(f"错误样本已保存到: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='计算车牌识别完全正确率 - 增强版')
    parser.add_argument('result_file', help='识别结果文件路径')
    parser.add_argument('--confidence', type=float, default=0.0,
                       help='置信度阈值，低于此值的结果将被过滤 (默认: 0.0)')
    parser.add_argument('--report', action='store_true',
                       help='保存详细报告到文件 (输入文件同目录下的accuracy_report.txt)')
    parser.add_argument('--save-errors', action='store_true',
                       help='保存错误样本到文件 (输入文件同目录下的error_samples.txt)')
    parser.add_argument('--json', action='store_true',
                       help='以JSON格式保存统计结果 (输入文件同目录下的accuracy_stats.json)')
    
    args = parser.parse_args()
    
    # 获取输入文件的目录
    input_dir = os.path.dirname(os.path.abspath(args.result_file))
    input_basename = os.path.splitext(os.path.basename(args.result_file))[0]
    
    # 设置保存路径（只有当相应的开关打开时才设置）
    report_path = None
    errors_path = None
    json_path = None
    
    if args.report:
        report_path = os.path.join(input_dir, f"{input_basename}_accuracy_report.txt")
    
    if args.save_errors:
        errors_path = os.path.join(input_dir, f"{input_basename}_error_samples.txt")
        
    if args.json:
        json_path = os.path.join(input_dir, f"{input_basename}_accuracy_stats.json")
    
    # 创建分析器
    analyzer = PlateAccuracyAnalyzer(confidence_threshold=args.confidence)
    
    # 解析结果文件
    if not analyzer.parse_result_file(args.result_file):
        return
    
    print(f"成功解析 {len(analyzer.results)} 个识别结果")
    
    # 生成报告
    report = analyzer.generate_report(save_path=report_path)
    print(report)
    
    # 保存错误样本
    if args.save_errors:
        analyzer.save_error_samples(errors_path)
    
    # 保存JSON格式结果
    if args.json:
        json_data = {
            'basic_stats': analyzer.stats,
            'province_stats': analyzer.analyze_by_province(),
            'confidence_stats': analyzer.analyze_by_confidence_range(),
            'error_patterns': analyzer.analyze_error_patterns()
        }
        
        # 转换不能JSON序列化的对象
        def json_serializable(obj):
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            return str(obj)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2, default=json_serializable)
        
        print(f"JSON统计结果已保存到: {json_path}")


if __name__ == '__main__':
    main()
