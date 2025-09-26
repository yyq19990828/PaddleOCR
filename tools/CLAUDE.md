[Root Directory](../CLAUDE.md) > **tools**

# Training and Inference Tools Module

## Module Responsibilities

The `tools/` directory contains command-line utilities for the complete model lifecycle: training, evaluation, inference, and export. These tools provide the interface between configuration files and the core `ppocr/` library functionality.

**Key Functions:**
- Model training and distributed training coordination
- Model evaluation and performance assessment
- Single-task inference (detection, recognition, classification)
- Model export and format conversion
- End-to-end system inference

## Entry and Startup

### Main Training Tools
- **`train.py`**: Main training script with support for single-GPU and distributed training
- **`eval.py`**: Model evaluation and validation
- **`export_model.py`**: Export trained models to inference format
- **`export_center.py`**: Export detection center point models

### Inference Tools
- **`infer_det.py`**: Text detection inference
- **`infer_rec.py`**: Text recognition inference
- **`infer_cls.py`**: Text line orientation classification
- **`infer_e2e.py`**: End-to-end OCR inference
- **`infer_kie.py`**: Key information extraction inference
- **`infer_sr.py`**: Super-resolution inference
- **`infer_table.py`**: Table recognition inference

### System Tools
- **`program.py`**: Common training program utilities
- **`test_hubserving.py`**: Hub serving functionality testing

## External Interfaces

### Training Interface
```bash
# Single GPU training
python tools/train.py -c configs/det/det_mv3_db.yml

# Distributed training
python -m paddle.distributed.launch --gpus '0,1,2,3' tools/train.py -c config.yml

# Resume training
python tools/train.py -c config.yml -o Global.checkpoints=./output/db_mv3/best_accuracy
```

### Evaluation Interface
```bash
# Model evaluation
python tools/eval.py -c config.yml -o Global.pretrained_model=./pretrain_models/model

# Specific dataset evaluation
python tools/eval.py -c config.yml -o Global.pretrained_model=model_path Eval.dataset.data_dir=./test_data/
```

### Inference Interface
```bash
# Detection inference
python tools/infer_det.py -c config.yml -o Global.pretrained_model=model_path Global.infer_img=./test_imgs/

# Recognition inference
python tools/infer_rec.py -c config.yml -o Global.pretrained_model=model_path Global.infer_img=./test_imgs/

# System inference (detection + recognition)
python tools/infer/predict_system.py --image_dir=./test_imgs/ --det_model_dir=./det_model/ --rec_model_dir=./rec_model/
```

### Model Export Interface
```bash
# Export inference model
python tools/export_model.py -c config.yml -o Global.pretrained_model=./pretrain_models/model Global.save_inference_dir=./inference/

# Export with specific input shape
python tools/export_model.py -c config.yml -o Global.pretrained_model=model_path Global.save_inference_dir=inference_dir
```

## Key Dependencies and Configuration

### Core Dependencies
- **PaddlePaddle**: Framework for model training and inference
- **ppocr**: Core library components
- **Configuration Management**: YAML-based configuration system
- **Logging**: Training progress and performance logging

### Configuration Override System
Tools support configuration overrides via command-line:
```bash
-o Global.pretrained_model=path/to/model
-o Architecture.Backbone.name=ResNet50
-o Train.dataset.data_dir=./custom_data/
```

### Environment Setup
- **Distributed Training**: Automatic GPU detection and allocation
- **Mixed Precision**: Support for FP16 training
- **Gradient Synchronization**: For multi-GPU training

## Data Models

### Training Data Flow
```
Config File → Data Loader → Model → Loss → Optimizer → Metrics → Checkpoints
```

### Inference Data Flow
```
Input Images → Preprocessing → Model → Postprocessing → Results
```

### Model Export Flow
```
Trained Model → Static Graph → Inference Model → Deployment Format
```

### Common Data Structures
```python
# Training configuration structure
config = {
    'Global': {'use_gpu': True, 'epoch_num': 500},
    'Architecture': {'model_type': 'det', 'algorithm': 'DB'},
    'Loss': {'name': 'DBLoss'},
    'Optimizer': {'name': 'Adam', 'lr': 0.001},
    'Train': {'dataset': {...}},
    'Eval': {'dataset': {...}}
}

# Inference results
det_results = [
    {'points': [[x1,y1], [x2,y2], [x3,y3], [x4,y4]], 'confidence': 0.95},
    # ... more detections
]

rec_results = [
    {'text': 'recognized_text', 'confidence': 0.92},
    # ... more recognitions
]
```

## Testing and Quality

### Test Coverage
- **Integration Testing**: Full pipeline testing with sample data
- **Performance Testing**: Speed and accuracy benchmarks
- **Configuration Validation**: Config file parsing and validation

### Validation Workflows
- **Training Validation**: Monitor loss curves and accuracy metrics
- **Inference Validation**: Compare outputs with expected results
- **Export Validation**: Verify exported model functionality

### Quality Metrics
- **Training Metrics**: Loss convergence, validation accuracy
- **Inference Speed**: FPS measurements across different hardware
- **Model Size**: Parameter count and memory usage

## Frequently Asked Questions (FAQ)

### Training Issues
**Q: Training stops with out-of-memory error?**
A: Reduce batch size in config, enable gradient accumulation, or use smaller image sizes.

**Q: How to monitor training progress?**
A: Check log files in output directory, use VisualDL for visualization, or monitor via TensorBoard.

**Q: Distributed training not working?**
A: Verify GPU availability, check CUDA_VISIBLE_DEVICES, ensure consistent config across nodes.

### Inference Issues
**Q: Inference results are poor?**
A: Check model path, verify input image preprocessing, ensure model-config compatibility.

**Q: How to batch process images?**
A: Use `Global.infer_img` with directory path, or modify scripts for batch processing.

### Export and Deployment
**Q: Exported model size too large?**
A: Use model pruning, quantization, or smaller backbone architectures.

**Q: Inference speed too slow?**
A: Enable TensorRT, use FP16 inference, or optimize preprocessing pipeline.

## Related File List

### Core Training Scripts
- `train.py` - Main training script
- `eval.py` - Model evaluation
- `program.py` - Training utilities and common functions

### Inference Scripts
- `infer_det.py` - Text detection inference
- `infer_rec.py` - Text recognition inference
- `infer_cls.py` - Classification inference
- `infer_e2e.py` - End-to-end inference
- `infer_kie.py` - Key information extraction
- `infer_sr.py` - Super-resolution inference
- `infer_table.py` - Table recognition

### System Integration
- `infer/predict_system.py` - Complete OCR system inference
- `infer/predict_det.py` - Detection-only system inference
- `infer/predict_rec.py` - Recognition-only system inference
- `infer/predict_cls.py` - Classification-only system inference

### Model Export and Conversion
- `export_model.py` - Export to inference format
- `export_center.py` - Export detection center models

### Utilities and Testing
- `test_hubserving.py` - Hub serving tests
- `naive_sync_bn.py` - Batch normalization utilities
- `__init__.py` - Module initialization

### Specialized Tools
- `end2end/` - End-to-end training and inference tools

## Change Log (Changelog)

### 2025-09-26: Module Documentation Created
- Comprehensive tools documentation
- Training, evaluation, and inference workflow documentation
- Configuration override and environment setup guidelines
- FAQ section for common issues and solutions