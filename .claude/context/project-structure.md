---
created: 2025-09-03T03:19:26Z
last_updated: 2025-09-03T03:19:26Z
version: 1.0
author: Claude Code PM System
---

# Project Structure: PaddleOCR

## Root Directory Structure

```
PaddleOCR/
├── .claude/                    # Claude Code PM and context system
├── configs/                    # Model configuration files
├── deploy/                     # Deployment and serving code
├── docs/                       # Documentation and resources
├── paddleocr/                  # New OCR library (high-level interface)
├── ppocr/                      # Core OCR library (training/inference)
├── tests/                      # Test suites
├── tools/                      # Training and utility scripts
├── benchmark/                  # Performance benchmarking
├── applications/               # Application examples
├── requirements.txt            # Python dependencies
├── setup.py                   # Package installation script
├── README.md                  # Project documentation
└── CLAUDE.md                  # Development guidelines
```

## Core Library Structure

### paddleocr/ - High-Level Interface
```
paddleocr/
├── _models/                    # Model implementations
│   ├── detection/              # Text detection models
│   ├── recognition/            # Text recognition models
│   ├── classification/         # Text classification models
│   └── structure/              # Document structure models
├── _pipelines/                 # Processing pipelines
│   ├── ocr.py                 # Main OCR pipeline
│   ├── structure.py           # Document structure pipeline
│   └── chatocr.py             # ChatOCR pipeline
├── tools/                      # Utility functions
└── __init__.py                # Package initialization
```

### ppocr/ - Core Training Library
```
ppocr/
├── data/                       # Data loading and preprocessing
│   ├── imaug/                 # Image augmentation
│   ├── simple_dataset.py      # Dataset implementations
│   └── lmdb_dataset.py        # LMDB dataset support
├── losses/                     # Loss functions
│   ├── det_*.py               # Detection losses
│   ├── rec_*.py               # Recognition losses
│   └── cls_*.py               # Classification losses
├── metrics/                    # Evaluation metrics
│   ├── det_metric.py          # Detection metrics
│   ├── rec_metric.py          # Recognition metrics
│   └── cls_metric.py          # Classification metrics
├── modeling/                   # Model architectures
│   ├── architectures/         # Complete model definitions
│   ├── backbones/             # Backbone networks
│   ├── necks/                 # Feature pyramid networks
│   └── heads/                 # Task-specific heads
├── optimizer/                  # Optimizers and schedulers
├── postprocess/               # Post-processing algorithms
└── utils/                     # Utility functions
    ├── dict/                  # Language dictionaries
    ├── logging.py             # Logging utilities
    └── save_load.py           # Model saving/loading
```

## Configuration System

### configs/ Directory Structure
```
configs/
├── det/                        # Detection model configs
│   ├── db/                    # DB algorithm configs
│   ├── east/                  # EAST algorithm configs
│   └── pse/                   # PSE algorithm configs
├── rec/                        # Recognition model configs
│   ├── crnn/                  # CRNN configs
│   ├── svtr/                  # SVTR configs
│   └── abinet/                # ABINet configs
├── cls/                        # Classification configs
├── kie/                        # Key information extraction
├── table/                      # Table recognition
└── _base_/                    # Base configuration templates
```

### Configuration File Pattern
- **Task Type Prefix**: `det_`, `rec_`, `cls_`, `kie_`
- **Algorithm Name**: Algorithm identifier (e.g., `db`, `crnn`)
- **Dataset/Language**: Dataset or language specific suffixes
- **File Extension**: `.yml` for all configuration files

## Documentation Structure

### docs/ Directory
```
docs/
├── version3.x/                 # Version 3.x documentation
│   ├── deployment/            # Deployment guides
│   ├── installation.md        # Installation instructions
│   ├── quick_start.md         # Quick start guide
│   └── tutorials/             # Detailed tutorials
├── images/                     # Documentation images
├── algorithm_overview/         # Algorithm documentation
└── FAQ.md                     # Frequently asked questions
```

### Multi-Language Documentation
- **English**: Primary documentation language
- **Chinese**: Simplified and Traditional Chinese versions
- **Other Languages**: Japanese, Korean, French, Russian, Spanish, Arabic
- **File Naming**: `README_{lang}.md` pattern for translations

## Deployment Structure

### deploy/ Directory
```
deploy/
├── cpp_infer/                  # C++ inference implementation
│   ├── src/                   # Source code
│   ├── include/               # Header files
│   └── CMakeLists.txt         # Build configuration
├── android_demo/              # Android deployment
├── ios_demo/                  # iOS deployment
├── hubserving/                # PaddleHub serving
├── pdserving/                 # PaddleServing deployment
├── slim/                      # Model compression
│   ├── quantization/          # Quantization scripts
│   └── prune/                # Pruning scripts
└── docker/                    # Docker configurations
```

## Tool Organization

### tools/ Directory
```
tools/
├── train.py                   # Main training script
├── eval.py                    # Evaluation script
├── export_model.py            # Model export utility
├── infer/                     # Inference tools
│   ├── predict_det.py         # Detection inference
│   ├── predict_rec.py         # Recognition inference
│   ├── predict_cls.py         # Classification inference
│   └── predict_system.py      # End-to-end system
├── program.py                 # Training program utilities
└── infer_det.py              # Detection inference (legacy)
```

## Data Organization Patterns

### Dataset Structure
```
datasets/
├── train_data/                # Training datasets
│   ├── icdar2015/            # ICDAR 2015 dataset
│   ├── mlt2017/              # MLT 2017 dataset
│   └── custom/               # Custom datasets
├── test_data/                 # Test datasets
└── pretrained_models/         # Pre-trained model weights
```

### Annotation Formats
- **Detection**: Text file with bounding box coordinates
- **Recognition**: Text file with image paths and labels
- **LMDB**: Binary database format for large datasets
- **JSON**: Structured annotation format

## Module Dependencies

### Import Hierarchy
1. **External Dependencies**: Third-party libraries
2. **PaddlePaddle Framework**: Core ML framework
3. **ppocr.utils**: Utility functions
4. **ppocr.data**: Data loading components
5. **ppocr.modeling**: Model architectures
6. **ppocr.losses**: Loss functions
7. **ppocr.metrics**: Evaluation metrics

### Circular Dependency Prevention
- Clear separation between layers
- Abstract base classes for interfaces
- Dependency injection for configurable components
- Factory patterns for model creation

## File Naming Conventions

### Python Files
- **Snake Case**: `model_name.py`
- **Descriptive Names**: Clear indication of functionality
- **Task Prefixes**: `det_`, `rec_`, `cls_` for task-specific files

### Configuration Files
- **YAML Format**: All configs use `.yml` extension
- **Hierarchical Naming**: `task_algorithm_dataset.yml`
- **Base Configs**: `_base_` directory for shared configurations

### Documentation Files
- **Markdown Format**: `.md` extension for documentation
- **Language Suffixes**: `_en.md`, `_cn.md` for translations
- **Descriptive Names**: Clear indication of content

## Directory Conventions

### Package Structure
- **`__init__.py`**: Present in all Python packages
- **Private Modules**: Underscore prefix for internal modules
- **Public API**: Clear separation of public interface

### Asset Organization
- **Images**: Grouped by usage (docs, examples, tests)
- **Models**: Organized by task and algorithm
- **Data**: Separated by dataset type and split

### Build Artifacts
- **`__pycache__/`**: Python bytecode cache (gitignored)
- **`build/`**: Build artifacts (gitignored)
- **`dist/`**: Distribution packages (gitignored)
- **`.egg-info/`**: Package metadata (gitignored)