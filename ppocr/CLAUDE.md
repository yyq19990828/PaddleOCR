[Root Directory](../CLAUDE.md) > **ppocr**

# PP-OCR Core Library Module

## Module Responsibilities

The `ppocr/` module is the low-level core library that provides training, fine-tuning, and research-oriented functionality for PaddleOCR. It maintains backward compatibility with 2.x APIs while serving as the foundation for the modern 3.x interface.

**Key Functions:**
- Model training and fine-tuning infrastructure
- Low-level model architectures and components
- Research-oriented algorithm implementations
- Data processing and augmentation pipelines
- Loss functions and optimization strategies

## Entry and Startup

### Module Structure
- **Core Init**: `__init__.py` - Minimal initialization for backward compatibility
- **Training Entry**: Called via `tools/train.py` and `tools/eval.py`
- **Inference Entry**: Used by higher-level `paddleocr/` pipelines

### Component Registration System
Components are registered through `__init__.py` files in each submodule:
1. **Models**: `modeling/architectures/__init__.py`
2. **Losses**: `losses/__init__.py`
3. **Metrics**: `metrics/__init__.py`
4. **Postprocessors**: `postprocess/__init__.py`
5. **Data Loaders**: `data/__init__.py`

## External Interfaces

### Training Interface
```python
# Via tools/train.py
python tools/train.py -c configs/det/det_mv3_db.yml
```

### Model Building API
```python
from ppocr.modeling.architectures import build_model
from ppocr.losses import build_loss
from ppocr.metrics import build_metric

model = build_model(config['Architecture'])
loss_fn = build_loss(config['Loss'])
metric_fn = build_metric(config['Metric'])
```

### Data Processing API
```python
from ppocr.data import build_dataloader
dataloader = build_dataloader(config, mode='train')
```

## Key Dependencies and Configuration

### Core Dependencies
- **PaddlePaddle**: Deep learning framework
- **OpenCV**: Image processing
- **Pillow**: Image manipulation
- **NumPy**: Numerical computing
- **Shapely**: Geometric operations

### Configuration System
- **YAML-based**: All configurations in `configs/` directory
- **Modular Design**: Separate configs for Architecture, Loss, Optimizer, etc.
- **Inheritance**: Support for config inheritance and overrides

### Configuration Structure
```yaml
Global:
  use_gpu: true
  epoch_num: 500
  save_epoch_step: 100

Architecture:
  model_type: det
  algorithm: DB
  Backbone:
    name: MobileNetV3
  # ... more architecture config

Loss:
  name: DBLoss
  # ... loss config
```

## Data Models

### Model Architecture Components
- **Backbone**: Feature extraction networks (ResNet, MobileNet, etc.)
- **Neck**: Feature fusion networks (FPN, etc.)
- **Head**: Task-specific output layers (DBHead, CTCHead, etc.)

### Detection Models
- **DB (Differentiable Binarization)**: Real-time text detection
- **EAST**: Efficient and Accurate Scene Text detection
- **PSE (Progressive Scale Expansion)**: Multi-scale text detection
- **SAST**: Single-shot Arbitrarily-shaped Text detection

### Recognition Models
- **CRNN**: CNN + RNN for sequence recognition
- **SVTR (Scene Text Recognition Transformer)**: Transformer-based recognition
- **ABINet**: Autonomous, Bidirectional and Iterative Language Modeling
- **SEED**: Self-supervised Text Recognition

### Data Structures
```python
# Detection Label Format
{
    'image': numpy.ndarray,
    'polys': List[List[List[float]]],  # Polygon coordinates
    'ignore_tags': List[bool],
    'texts': List[str]
}

# Recognition Label Format
{
    'image': numpy.ndarray,
    'label': str,  # Ground truth text
    'length': int  # Text length
}
```

## Testing and Quality

### Test Infrastructure
- **Legacy Tests**: Basic functionality tests
- **Integration**: Via tools testing with sample data
- **Model Tests**: Architecture consistency checks

### Validation Strategy
- **Accuracy Metrics**: Detection (Precision, Recall, F1), Recognition (Accuracy)
- **Speed Benchmarks**: FPS measurements on standard hardware
- **Memory Profiling**: Resource usage monitoring

### Quality Standards
- **Model Convergence**: Training curves and loss progression
- **Numerical Stability**: Gradient flow and optimization behavior
- **Reproducibility**: Seed control and deterministic operations

## Frequently Asked Questions (FAQ)

### Training and Fine-tuning
**Q: How to train a custom detection model?**
A: Create a config file based on existing templates, prepare data in required format, and run `python tools/train.py -c your_config.yml`

**Q: How to resume training from checkpoint?**
A: Use `Global.checkpoints` parameter in config or `-o Global.checkpoints=path/to/checkpoint`

### Model Architecture
**Q: How to add a new backbone?**
A: Implement in `ppocr/modeling/backbones/`, register in `__init__.py`, and reference in config file

**Q: How to customize loss functions?**
A: Add to `ppocr/losses/`, follow existing patterns, and register the loss class

### Data Processing
**Q: What data formats are supported?**
A: LMDB databases, simple text files with image paths, and custom dataset classes

**Q: How to add data augmentation?**
A: Implement transforms in `ppocr/data/imaug/` and add to transform pipeline in config

## Related File List

### Core Architecture
- `__init__.py` - Module initialization
- `modeling/architectures/` - Model architecture definitions
  - `base_model.py` - Base model class
  - `__init__.py` - Model registry and builder
- `modeling/backbones/` - Feature extraction networks
- `modeling/necks/` - Feature fusion networks
- `modeling/heads/` - Task-specific output layers
- `modeling/transforms/` - Model-level transformations

### Training Infrastructure
- `losses/__init__.py` - Loss function registry
- `losses/` - Various loss implementations (DBLoss, CTCLoss, etc.)
- `metrics/__init__.py` - Metric registry
- `metrics/` - Evaluation metrics
- `optimizer/` - Optimization strategies and schedulers

### Data Processing
- `data/__init__.py` - Data loading infrastructure
- `data/collate_fn.py` - Batch collation functions
- `data/imaug/` - Image augmentation and preprocessing
  - `operators.py` - Data transformation operators
  - `text_image_aug/` - Text-specific augmentations

### Post-processing
- `postprocess/__init__.py` - Post-processor registry
- `postprocess/` - Detection and recognition post-processing
  - `db_postprocess.py` - DB detection post-processing
  - `rec_postprocess.py` - Recognition post-processing

### Utilities
- `utils/` - Utility functions
  - `save_load.py` - Model saving and loading
  - `utility.py` - General utilities
  - `logging/` - Logging configuration
- `ext_op/` - External operations and custom ops

## Change Log (Changelog)

### 2025-09-26: Module Documentation Created
- Comprehensive core library documentation
- Training and research workflow documentation
- Component architecture and registration system overview
- Data processing and model building API documentation