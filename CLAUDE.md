# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 常用命令

### 训练相关
- **训练模型**: `python tools/train.py -c <config_file>`
- **分布式训练**: `python -m paddle.distributed.launch --gpus '0,1,2,3' tools/train.py -c <config_file>`
- **评估模型**: `python tools/eval.py -c <config_file> -o Global.pretrained_model=<model_path>`

### 推理和预测
- **文本检测**: `python tools/infer_det.py -c <config_file> -o Global.pretrained_model=<model_path> Global.infer_img=<image_path>`
- **文本识别**: `python tools/infer_rec.py -c <config_file> -o Global.pretrained_model=<model_path> Global.infer_img=<image_path>`
- **系统预测**: `python tools/infer/predict_system.py --image_dir=<image_path> --det_model_dir=<det_model> --rec_model_dir=<rec_model>`

### 模型导出
- **导出推理模型**: `python tools/export_model.py -c <config_file> -o Global.pretrained_model=<model_path> Global.save_inference_dir=<output_dir>`

### 测试
- **运行pytest测试**: `pytest tests/`
- **运行单个测试**: `pytest tests/test_<specific_test>.py`

### 安装和环境
- **安装依赖**: `pip install -r requirements.txt`
- **安装PaddleOCR**: `pip install paddleocr`

## 代码架构概览

### 主要目录结构
- **paddleocr/**: 新版本OCR库，提供高级接口和管道化功能
  - `_models/`: 各种模型实现（检测、识别、分类等）
  - `_pipelines/`: 处理管道（OCR、文档结构化、ChatOCR等）
- **ppocr/**: 传统OCR核心库，包含训练和推理代码
  - `data/`: 数据加载和预处理
  - `modeling/`: 深度学习模型架构
  - `losses/`: 各种损失函数
  - `metrics/`: 评估指标
  - `postprocess/`: 后处理算法
- **configs/**: 各种模型和任务的配置文件
- **tools/**: 训练、评估、推理工具脚本
- **deploy/**: 部署相关代码（C++、移动端、服务化等）
- **docs/**: 文档资源
- **tests/**: 测试代码

### 核心功能模块
1. **文本检测**: DB、EAST、PSE、SAST等算法
2. **文本识别**: CRNN、SVTR、ABINet等算法
3. **文档结构化**: 表格识别、版面分析、公式识别
4. **关键信息提取**: LayoutLM系列模型
5. **端到端OCR**: PGNet等算法

### 模型配置系统
- 配置文件采用YAML格式，位于`configs/`目录
- 支持全局配置、架构配置、损失函数配置、优化器配置等
- 配置文件按任务类型分类：det（检测）、rec（识别）、cls（分类）、kie（关键信息提取）等

### 数据处理流程
1. **数据加载**: 支持LMDB、简单数据集等格式
2. **数据增强**: 丰富的图像变换和增强策略
3. **标签处理**: 各种标签格式的转换和处理
4. **批次整理**: collate_fn处理变长数据

### 训练流程
1. **配置解析**: 从YAML文件加载配置
2. **数据准备**: 创建数据加载器
3. **模型构建**: 根据配置构建网络架构
4. **优化器设置**: 配置学习率调度和优化策略
5. **训练循环**: 前向传播、损失计算、反向传播
6. **模型保存**: 定期保存检查点和最佳模型

## 开发注意事项

### 添加新模型
- 在`ppocr/modeling/`对应目录下添加模型实现
- 在`__init__.py`中注册新模型
- 创建对应的配置文件在`configs/`目录
- 添加相应的损失函数和后处理方法

### 添加新数据集
- 在`ppocr/data/`下实现数据集加载器
- 实现必要的数据预处理和标签转换
- 更新配置文件以使用新数据集

### 代码风格
- 使用PaddlePaddle深度学习框架
- 遵循现有的代码结构和命名规范
- 添加适当的文档字符串和注释