# 数据预处理和增强

PaddleOCR 支持丰富的数据预处理和增强功能，通过在配置文件中的 `transforms` 字段配置各种变换操作。这些操作用于在训练、评估和推理过程中对输入数据进行预处理。

## 基础变换

### DecodeImage

图像解码器，用于将图像数据从字节流解码为numpy数组。

```yaml
- DecodeImage:
    img_mode: BGR          # 图像模式：BGR、RGB
    channel_first: false   # 是否使用CHW格式，false表示使用HWC格式
```

### RecResizeImg

识别任务图像尺寸调整器。

```yaml
- RecResizeImg:
    image_shape: [3, 32, 320]  # 目标图像尺寸 [C, H, W]
```

### KeepKeys

保留指定的数据字段。

```yaml
- KeepKeys:
    keep_keys:
      - image
      - label_ctc
      - length
```

## 数据增强

### RecAug

识别任务的随机数据增强，包含多种增强策略。

```yaml
- RecAug:
    # 随机应用多种增强方法
```

### IaaAugment

基于imgaug库的数据增强。

```yaml
- IaaAugment:
    augmenter_args:
      - {'type': Fliplr, 'args': {'p': 0.5}}
      - {'type': Affine, 'args': {'rotate': [-10, 10]}}
```

## 数据过滤

### FilterByImageWidth

根据图像宽度过滤数据，用于聚焦特定尺寸范围的图像。

#### 功能说明

FilterByImageWidth操作符用于根据图片宽度过滤数据集，在模型评估时可以聚焦于特定尺寸范围的图片，有助于分析模型在不同图像尺寸下的性能表现。

#### 配置参数

- `width_range`: 宽度范围配置
  - `[min, max]`: 保留宽度在min到max之间的图片（包含边界值）
  - `[min, ]`: 保留宽度大于等于min的图片
  - `None`: 禁用过滤（默认值）

#### 使用示例

```yaml
transforms:
  - DecodeImage:
      img_mode: BGR
      channel_first: false
  - FilterByImageWidth:
      width_range: [50, 300]  # 只保留宽度50-300像素的图片
  - RecResizeImg:
      image_shape: [3, 32, 320]
  - KeepKeys:
      keep_keys:
        - image
        - label_ctc
        - length
```

#### 使用场景

1. **移动端小图片性能评估**
   ```yaml
   - FilterByImageWidth:
       width_range: [0, 80]  # 聚焦小尺寸图片
   ```

2. **高分辨率图片测试**
   ```yaml
   - FilterByImageWidth:
       width_range: [1000, ]  # 测试大尺寸图片性能
   ```

3. **标准尺寸范围测试**
   ```yaml
   - FilterByImageWidth:
       width_range: [100, 500]  # 常见尺寸范围
   ```

4. **特定应用场景**
   ```yaml
   - FilterByImageWidth:
       width_range: [200, 800]  # 文档扫描场景
   ```

#### 注意事项

- **使用位置**: 必须在 `DecodeImage` 之后使用，确保图像已经被解码
- **统计信息**: 过滤统计信息会自动输出到日志，包括原始数据量、过滤后数据量和过滤比例
- **性能影响**: 过滤操作会减少实际参与训练/评估的数据量，需要根据具体需求调整范围参数
- **参数调整**: 建议根据具体应用场景和数据集特点调整宽度范围参数

## 标签编码

### MultiLabelEncode

多标签编码器，支持CTC和Attention等多种编码方式。

```yaml
- MultiLabelEncode:
    gtc_encode: NRTRLabelEncode  # 指定标签编码方式
```

### CTCLabelEncode

CTC标签编码器。

```yaml
- CTCLabelEncode:
    use_space_char: false
```

## 配置示例

### 训练配置

```yaml
Train:
  dataset:
    name: SimpleDataSet
    data_dir: ./datasets/train_data/
    label_file_list:
      - ./datasets/train_data/rec_gt_train.txt
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: false
      - FilterByImageWidth:
          width_range: [50, 800]  # 过滤异常尺寸图片
      - RecAug:                    # 数据增强
      - CTCLabelEncode:
          use_space_char: false
      - RecResizeImg:
          image_shape: [3, 32, 320]
      - KeepKeys:
          keep_keys:
            - image
            - label_ctc
            - length
```

### 评估配置

```yaml
Eval:
  dataset:
    name: SimpleDataSet
    data_dir: ./datasets/eval_data/
    label_file_list:
      - ./datasets/eval_data/rec_gt_eval.txt
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: false
      - FilterByImageWidth:
          width_range: [100, 500]  # 评估特定尺寸范围
      - CTCLabelEncode:
          use_space_char: false
      - RecResizeImg:
          image_shape: [3, 32, 320]
      - KeepKeys:
          keep_keys:
            - image
            - label_ctc
            - length
```

## 自定义变换

如需添加新的数据变换操作，可以参考现有变换的实现方式：

1. 在 `ppocr/data/imaug/` 目录下创建新的变换类
2. 在 `ppocr/data/imaug/__init__.py` 中注册新的变换
3. 在配置文件中使用新的变换

更多信息请参考 [新增算法文档](../algorithm/add_new_algorithm.md)。

## 常见问题和故障排除

### FilterByImageWidth 常见问题

**Q1: FilterByImageWidth 操作符没有过滤任何数据？**

A: 请检查以下几点：
- 确保 `FilterByImageWidth` 放在 `DecodeImage` 之后
- 检查 `width_range` 参数设置是否正确
- 查看日志输出，确认过滤统计信息

```yaml
# 错误示例：FilterByImageWidth 在 DecodeImage 之前
- FilterByImageWidth:
    width_range: [100, 500]
- DecodeImage:
    img_mode: BGR

# 正确示例：FilterByImageWidth 在 DecodeImage 之后  
- DecodeImage:
    img_mode: BGR
- FilterByImageWidth:
    width_range: [100, 500]
```

**Q2: 过滤后数据量过少影响训练效果？**

A: 建议调整策略：
- 适当放宽宽度范围参数
- 根据数据集特点调整过滤条件
- 考虑在训练和评估中使用不同的过滤策略

```yaml
# 训练时：较宽松的过滤条件
- FilterByImageWidth:
    width_range: [30, 1000]

# 评估时：针对特定场景的过滤条件
- FilterByImageWidth:
    width_range: [100, 500]
```

**Q3: 如何查看过滤的统计信息？**

A: FilterByImageWidth 会自动在日志中输出统计信息，包括：
- 原始数据量
- 过滤后数据量  
- 过滤比例
- 被过滤掉的图片宽度分布

**Q4: 在多数据源配置中如何使用 FilterByImageWidth？**

A: 每个数据源可以独立配置过滤条件：

```yaml
Train:
  dataset:
    name: MultiScaleDataSet
    data_dir: 
      - ./datasets/source1/
      - ./datasets/source2/
    label_file_list:
      - ./datasets/source1/train.txt
      - ./datasets/source2/train.txt
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: false
      # 统一的过滤策略应用到所有数据源
      - FilterByImageWidth:
          width_range: [50, 800]
```

### 一般数据增强问题

**Q5: 数据增强导致训练速度变慢？**

A: 优化建议：
- 合理设置数据加载的 `num_workers` 参数
- 避免过于复杂的增强策略组合
- 考虑使用GPU加速的数据增强库

**Q6: 如何调试数据增强效果？**

A: 调试方法：
- 保存增强后的图像样本查看效果
- 使用 `debug: true` 开启调试模式
- 对比有无增强的训练效果

```yaml
Global:
  debug: true  # 开启调试模式，会保存中间处理结果
```

**Q7: 配置文件中的 transforms 顺序重要吗？**

A: 是的，transforms 按配置顺序依次执行，顺序错误可能导致：
- 程序报错
- 数据处理结果异常
- 模型性能下降

推荐的基本顺序：
1. DecodeImage（图像解码）
2. FilterByImageWidth（数据过滤，可选）
3. 数据增强操作（RecAug、IaaAugment等）
4. 标签编码（CTCLabelEncode等）
5. 图像尺寸调整（RecResizeImg等）
6. KeepKeys（保留需要的字段）

### 性能优化建议

1. **数据加载优化**
   ```yaml
   loader:
     num_workers: 4        # 根据CPU核数调整
     use_shared_memory: false
     batch_size_per_card: 64
   ```

2. **内存使用优化**
   - 适当调整batch_size
   - 避免同时使用多个重型增强操作
   - 合理设置图像尺寸

3. **训练稳定性**
   - 在评估数据集中避免使用随机增强
   - 保持训练和评估的预处理流程一致性
   - 记录和监控过滤统计信息

如果遇到其他问题，请参考 [FAQ](../../../FAQ.md) 或在 [GitHub Issues](https://github.com/PaddlePaddle/PaddleOCR/issues) 中寻求帮助。