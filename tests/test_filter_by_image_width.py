import pytest
import numpy as np


# Mock logger for standalone testing
class MockLogger:
    def info(self, msg):
        print(f"INFO: {msg}")

def get_logger():
    return MockLogger()


# Standalone FilterByImageWidth implementation for testing
class FilterByImageWidth(object):
    """根据图像宽度过滤数据样本，并收集过滤统计信息"""

    def __init__(self, width_range=None, **kwargs):
        """
        初始化宽度过滤器
        
        Args:
            width_range (list or None): 宽度过滤范围
                - [min, max]: 保留 min <= width <= max 的图片
                - [min, ]: 保留 width >= min 的图片  
                - None: 不进行过滤（默认）
        """
        _ = kwargs  # Suppress unused parameter warning
        self.width_range = width_range
        self.logger = get_logger()
        
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
        img = data["image"]
        assert isinstance(img, np.ndarray), "invalid input 'img' in FilterByImageWidth, expected numpy array, got {}".format(type(img))
        
        # 增加总计数
        self.total_count += 1
        
        # 如果未设置过滤条件，直接返回
        if self.width_range is None:
            return data
        
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


class TestFilterByImageWidth:
    """FilterByImageWidth操作符的全面单元测试"""
    
    def create_sample_data(self, width, height=100):
        """创建指定宽度的测试图像数据"""
        image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        return {"image": image, "label": "test_label"}
    
    def test_closed_range_filtering(self):
        """测试[min, max]闭区间过滤"""
        # 测试正常的闭区间过滤
        filter_op = FilterByImageWidth(width_range=[50, 200])
        
        # 测试宽度在范围内的图像 - 应该保留
        data_100 = self.create_sample_data(100)
        result = filter_op(data_100)
        assert result is not None
        assert result["image"].shape[1] == 100
        
        # 测试边界值 - 应该保留
        data_50 = self.create_sample_data(50)
        result = filter_op(data_50)
        assert result is not None
        assert result["image"].shape[1] == 50
        
        data_200 = self.create_sample_data(200)
        result = filter_op(data_200)
        assert result is not None
        assert result["image"].shape[1] == 200
        
        # 测试宽度小于最小值 - 应该过滤掉
        data_30 = self.create_sample_data(30)
        result = filter_op(data_30)
        assert result is None
        
        # 测试宽度大于最大值 - 应该过滤掉
        data_300 = self.create_sample_data(300)
        result = filter_op(data_300)
        assert result is None
    
    def test_open_range_filtering(self):
        """测试[min, ]开区间过滤"""
        # 测试只有最小宽度限制
        filter_op = FilterByImageWidth(width_range=[100, None])
        
        # 测试宽度等于最小值 - 应该保留
        data_100 = self.create_sample_data(100)
        result = filter_op(data_100)
        assert result is not None
        assert result["image"].shape[1] == 100
        
        # 测试宽度大于最小值 - 应该保留
        data_500 = self.create_sample_data(500)
        result = filter_op(data_500)
        assert result is not None
        assert result["image"].shape[1] == 500
        
        # 测试宽度小于最小值 - 应该过滤掉
        data_50 = self.create_sample_data(50)
        result = filter_op(data_50)
        assert result is None
        
        # 测试只有最大宽度限制
        filter_op = FilterByImageWidth(width_range=[None, 150])
        
        # 测试宽度等于最大值 - 应该保留
        data_150 = self.create_sample_data(150)
        result = filter_op(data_150)
        assert result is not None
        assert result["image"].shape[1] == 150
        
        # 测试宽度小于最大值 - 应该保留
        data_50 = self.create_sample_data(50)
        result = filter_op(data_50)
        assert result is not None
        assert result["image"].shape[1] == 50
        
        # 测试宽度大于最大值 - 应该过滤掉
        data_200 = self.create_sample_data(200)
        result = filter_op(data_200)
        assert result is None
    
    def test_no_filtering(self):
        """测试None配置（无过滤）"""
        filter_op = FilterByImageWidth(width_range=None)
        
        # 测试各种宽度的图像都应该通过
        for width in [10, 50, 100, 500, 1000]:
            data = self.create_sample_data(width)
            result = filter_op(data)
            assert result is not None
            assert result["image"].shape[1] == width
    
    def test_boundary_values(self):
        """测试边界值处理"""
        filter_op = FilterByImageWidth(width_range=[100, 200])
        
        # 测试临界值
        boundary_values = [99, 100, 101, 199, 200, 201]
        expected_results = [None, "pass", "pass", "pass", "pass", None]
        
        for width, expected in zip(boundary_values, expected_results):
            data = self.create_sample_data(width)
            result = filter_op(data)
            if expected is None:
                assert result is None, f"Width {width} should be filtered out"
            else:
                assert result is not None, f"Width {width} should pass through"
                assert result["image"].shape[1] == width
    
    def test_invalid_config(self):
        """测试无效配置的错误处理"""
        
        # 测试非列表/元组类型
        with pytest.raises(ValueError, match="width_range must be a list or tuple"):
            FilterByImageWidth(width_range="invalid")
        
        with pytest.raises(ValueError, match="width_range must be a list or tuple"):
            FilterByImageWidth(width_range=100)
        
        # 测试错误的列表长度
        with pytest.raises(ValueError, match="width_range should have exactly 2 elements"):
            FilterByImageWidth(width_range=[100])
        
        with pytest.raises(ValueError, match="width_range should have exactly 2 elements"):
            FilterByImageWidth(width_range=[100, 200, 300])
        
        # 测试最小值大于最大值
        with pytest.raises(ValueError, match="min_width .* should be less than or equal to max_width"):
            FilterByImageWidth(width_range=[200, 100])
        
        # 测试都为None的情况
        with pytest.raises(ValueError, match="width_range cannot be"):
            FilterByImageWidth(width_range=[None, None])
    
    def test_invalid_input_data(self):
        """测试无效输入数据的错误处理"""
        filter_op = FilterByImageWidth(width_range=[50, 200])
        
        # 测试非numpy数组输入
        invalid_data = {"image": "not_an_array", "label": "test"}
        with pytest.raises(AssertionError, match="invalid input 'img' in FilterByImageWidth"):
            filter_op(invalid_data)
        
        # 测试缺少image键
        with pytest.raises(KeyError):
            filter_op({"label": "test"})
    
    def test_statistics_collection(self):
        """测试统计信息收集的准确性"""
        filter_op = FilterByImageWidth(width_range=[100, 200])
        
        # 准备测试数据：宽度为50, 100, 150, 200, 250的图像
        test_widths = [50, 100, 150, 200, 250]
        
        # 处理测试数据
        for width in test_widths:
            data = self.create_sample_data(width)
            filter_op(data)  # 只需要处理数据，不需要检查结果
        
        # 验证统计信息
        stats = filter_op.get_statistics()
        
        assert stats['total_count'] == 5
        assert stats['filtered_count'] == 2  # 50和250被过滤
        assert stats['kept_count'] == 3      # 100, 150, 200被保留
        assert stats['filter_rate'] == 40.0  # 2/5 * 100
        assert stats['keep_rate'] == 60.0    # 3/5 * 100
        assert stats['width_range'] == [100, 200]
        
        # 测试空统计信息（没有处理任何数据时）
        empty_filter = FilterByImageWidth(width_range=[100, 200])
        empty_stats = empty_filter.get_statistics()
        
        assert empty_stats['total_count'] == 0
        assert empty_stats['filtered_count'] == 0
        assert empty_stats['kept_count'] == 0
        assert empty_stats['filter_rate'] == 0.0
        assert empty_stats['keep_rate'] == 0.0
    
    def test_statistics_with_no_filtering(self):
        """测试无过滤配置下的统计信息"""
        filter_op = FilterByImageWidth(width_range=None)
        
        # 处理一些数据
        for width in [50, 100, 200]:
            data = self.create_sample_data(width)
            result = filter_op(data)
            assert result is not None  # 无过滤时都应该通过
        
        stats = filter_op.get_statistics()
        
        # 无过滤时，所有图像都应该被保留
        assert stats['total_count'] == 3
        assert stats['filtered_count'] == 0
        assert stats['kept_count'] == 3
        assert stats['filter_rate'] == 0.0
        assert stats['keep_rate'] == 100.0
        assert stats['width_range'] is None
    
    def test_print_statistics(self):
        """测试打印统计信息功能"""
        # 测试有过滤配置时的打印
        filter_op = FilterByImageWidth(width_range=[100, 200])
        
        # 处理一些数据
        for width in [50, 150, 250]:
            data = self.create_sample_data(width)
            filter_op(data)
        
        # 这里我们只能测试方法不抛异常，因为它只是打印日志
        try:
            filter_op.print_statistics()
        except Exception as e:
            pytest.fail(f"print_statistics should not raise exception: {e}")
        
        # 测试无过滤配置时的打印
        no_filter_op = FilterByImageWidth(width_range=None)
        try:
            no_filter_op.print_statistics()
        except Exception as e:
            pytest.fail(f"print_statistics should not raise exception: {e}")
    
    def test_multiple_operators_isolation(self):
        """测试多个操作符实例之间的统计信息隔离"""
        filter1 = FilterByImageWidth(width_range=[50, 100])
        filter2 = FilterByImageWidth(width_range=[150, 200])
        
        # 用filter1处理数据
        data1 = self.create_sample_data(75)
        filter1(data1)
        
        # 用filter2处理数据  
        data2 = self.create_sample_data(175)
        filter2(data2)
        
        # 验证统计信息互不影响
        stats1 = filter1.get_statistics()
        stats2 = filter2.get_statistics()
        
        assert stats1['total_count'] == 1
        assert stats1['kept_count'] == 1
        assert stats1['width_range'] == [50, 100]
        
        assert stats2['total_count'] == 1
        assert stats2['kept_count'] == 1
        assert stats2['width_range'] == [150, 200]
    
    def test_edge_case_configurations(self):
        """测试边界配置情况"""
        
        # 测试极小范围过滤
        tiny_filter = FilterByImageWidth(width_range=[100, 100])
        
        data_100 = self.create_sample_data(100)
        result = tiny_filter(data_100)
        assert result is not None  # 正好等于边界应该通过
        
        data_99 = self.create_sample_data(99)
        result = tiny_filter(data_99)
        assert result is None  # 小于边界应该过滤
        
        data_101 = self.create_sample_data(101)
        result = tiny_filter(data_101)
        assert result is None  # 大于边界应该过滤
        
        # 测试极大数值
        large_filter = FilterByImageWidth(width_range=[10000, None])
        data_small = self.create_sample_data(5000)
        result = large_filter(data_small)
        assert result is None  # 小于最小值应该过滤
    
    def test_data_integrity(self):
        """测试数据完整性保持"""
        filter_op = FilterByImageWidth(width_range=[50, 200])
        
        # 创建带有额外数据的测试样本
        original_data = {
            "image": np.random.randint(0, 256, (100, 150, 3), dtype=np.uint8),
            "label": "test_label",
            "extra_field": "extra_value",
            "numeric_field": 42
        }
        
        result = filter_op(original_data)
        
        # 验证数据完整性
        assert result is not None
        assert np.array_equal(result["image"], original_data["image"])
        assert result["label"] == original_data["label"]
        assert result["extra_field"] == original_data["extra_field"]
        assert result["numeric_field"] == original_data["numeric_field"]
        
        # 验证图像属性没有被修改
        assert result["image"].shape == original_data["image"].shape
        assert result["image"].dtype == original_data["image"].dtype


if __name__ == "__main__":
    pytest.main([__file__])