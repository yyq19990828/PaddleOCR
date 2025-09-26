[Root Directory](../CLAUDE.md) > **paddleocr**

# PaddleOCR High-Level API Module

## Module Responsibilities

The `paddleocr/` module provides the modern 3.x API interface for PaddleOCR, built on the PaddleX framework. It offers user-friendly pipelines for end-to-end OCR workflows with automatic model management, caching, and optimization.

**Key Functions:**
- Unified pipeline interfaces for OCR tasks
- Automatic model downloading and caching
- Hardware acceleration abstraction
- Production-ready inference optimization
- Integration with latest PP-OCR series models

## Entry and Startup

### Main Entry Point
- **Module Init**: `__init__.py` - Exposes all public APIs and pipelines
- **CLI Entry**: `__main__.py` - Command-line interface via `paddleocr` command
- **Version Management**: `_version.py` - Version information and compatibility

### Core Initialization Flow
1. Import model classes from `_models/`
2. Import pipeline classes from `_pipelines/`
3. Initialize logging via `_utils.logging`
4. Expose public API through `__all__` list

## External Interfaces

### Pipeline APIs
- **PaddleOCR**: Main OCR pipeline for text detection and recognition
- **PPStructureV3**: Document parsing and structure analysis
- **PPChatOCRv4Doc**: Intelligent information extraction with ERNIE 4.5
- **DocUnderstanding**: Document comprehension and analysis
- **FormulaRecognitionPipeline**: Mathematical formula recognition
- **SealRecognition**: Official seal detection and recognition
- **TableRecognitionPipelineV2**: Table structure recognition
- **DocPreprocessor**: Document preprocessing pipeline
- **PPDocTranslation**: Document translation services

### Model Components
- **TextDetection**: Text detection models (PP-OCRv5 Det)
- **TextRecognition**: Text recognition models (PP-OCRv5 Rec)
- **LayoutDetection**: Document layout analysis
- **TableStructureRecognition**: Table structure parsing
- **FormulaRecognition**: Mathematical formula parsing
- **ChartParsing**: Chart and diagram analysis

### Command Line Interface
```bash
paddleocr --image_dir <path> --lang <language> [options]
```

## Key Dependencies and Configuration

### Core Dependencies
- **PaddleX Framework**: `paddlex[ocr-core]>=3.2.0,<3.3.0`
- **Configuration**: `PyYAML>=6` for model configs
- **Type Support**: `typing-extensions>=4.12`

### Optional Dependencies
- **Document Parsing**: `paddlex[ocr]>=3.2.0,<3.3.0`
- **Information Extraction**: `paddlex[ie]>=3.2.0,<3.3.0`
- **Translation**: `paddlex[trans]>=3.2.0,<3.3.0`

### Configuration Management
- Model configurations managed by PaddleX framework
- Automatic model download and caching
- Hardware-specific optimizations (CPU/GPU/XPU/NPU)
- Memory and performance tuning

## Data Models

### Pipeline Input/Output
```python
# OCR Pipeline
ocr = PaddleOCR(use_gpu=True, lang='ch')
result = ocr.ocr(img_path, det=True, rec=True, cls=True)
# Result: List[List[Tuple[List[List[float]], Tuple[str, float]]]]

# Document Structure
parser = PPStructureV3()
result = parser(pdf_path)
# Result: Dict with markdown content and metadata

# Information Extraction
extractor = PPChatOCRv4Doc()
result = extractor(image_path, query="Extract invoice number")
# Result: Dict with extracted information and confidence
```

### Model Data Structures
- **Detection Results**: Bounding boxes with coordinates and confidence
- **Recognition Results**: Text strings with confidence scores
- **Structure Results**: Hierarchical document structure
- **Extraction Results**: Key-value pairs with confidence and context

## Testing and Quality

### Test Coverage
- **Pipeline Tests**: `tests/pipelines/test_*.py`
  - `test_ocr.py`: Main OCR pipeline testing
  - `test_pp_structurev3.py`: Document structure parsing
  - `test_pp_chatocrv4_doc.py`: Information extraction
  - `test_doc_understanding.py`: Document comprehension

### Test Strategy
- Integration tests for end-to-end pipelines
- Performance benchmarking for inference speed
- Accuracy validation against reference datasets
- Memory usage and resource monitoring

### Quality Assurance
- Automated model validation
- Hardware compatibility testing
- Version compatibility checks
- Performance regression testing

## Frequently Asked Questions (FAQ)

### Model Management
**Q: How are models downloaded and cached?**
A: Models are automatically downloaded on first use and cached locally. Use `model_dir` parameter to specify custom cache location.

**Q: How to use custom models?**
A: Pass custom model paths via pipeline initialization parameters or use PaddleX model management APIs.

### Performance Optimization
**Q: How to optimize inference speed?**
A: Use GPU acceleration, enable TensorRT, adjust batch size, and configure appropriate hardware backends.

**Q: Memory usage too high?**
A: Reduce image resolution, use smaller models, or enable memory optimization flags in pipeline configuration.

### Integration
**Q: How to integrate with existing applications?**
A: Use pipeline APIs for batch processing, REST API for service deployment, or direct model calls for custom workflows.

## Related File List

### Core Architecture
- `__init__.py` - Main module initialization and public API
- `__main__.py` - Command-line entry point
- `_version.py` - Version management
- `_constants.py` - Global constants
- `_common_args.py` - Common argument definitions
- `_abstract.py` - Abstract base classes
- `_cli.py` - CLI implementation
- `_env.py` - Environment configuration

### Pipeline Implementation
- `_pipelines/ocr.py` - Main OCR pipeline
- `_pipelines/pp_structurev3.py` - Document structure analysis
- `_pipelines/pp_chatocrv4_doc.py` - Information extraction
- `_pipelines/doc_understanding.py` - Document comprehension
- `_pipelines/formula_recognition.py` - Formula recognition
- `_pipelines/seal_recognition.py` - Seal recognition
- `_pipelines/table_recognition_v2.py` - Table recognition

### Model Components
- `_models/text_detection.py` - Text detection models
- `_models/text_recognition.py` - Text recognition models
- `_models/layout_detection.py` - Layout analysis models
- `_models/table_structure_recognition.py` - Table structure models
- `_models/formula_recognition.py` - Formula recognition models

### Utilities
- `_utils/logging.py` - Logging configuration
- `_utils/` - Additional utility functions

## Change Log (Changelog)

### 2025-09-26: Module Documentation Created
- Comprehensive module documentation
- API interface documentation
- Integration guidelines and FAQ
- Test strategy and quality assurance overview