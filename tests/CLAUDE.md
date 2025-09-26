[Root Directory](../CLAUDE.md) > **tests**

# Testing and Quality Assurance Module

## Module Responsibilities

The `tests/` directory contains comprehensive testing infrastructure for PaddleOCR, covering unit tests, integration tests, pipeline tests, and quality assurance. It ensures reliability, performance, and correctness across all components and deployment scenarios.

**Key Functions:**
- Unit testing for individual components and models
- Integration testing for end-to-end pipelines
- Performance benchmarking and regression testing
- Quality assurance for accuracy and reliability
- Automated testing infrastructure for CI/CD

## Entry and Startup

### Test Organization
- **Model Tests**: `models/` - Individual model component testing
- **Pipeline Tests**: `pipelines/` - End-to-end pipeline testing
- **Component Tests**: Root level - Utility and component testing
- **Test Utilities**: `testing_utils.py` - Common testing infrastructure

### Test Execution
```bash
# Run all tests (excluding resource-intensive)
pytest tests/

# Run specific test module
pytest tests/pipelines/test_ocr.py

# Run with resource-intensive tests
pytest tests/ -m ""

# Verbose output for debugging
pytest tests/test_specific.py -v -s
```

## External Interfaces

### Testing Framework
- **pytest**: Primary testing framework with fixtures and parametrization
- **unittest**: Legacy test support for backward compatibility
- **Markers**: Resource-intensive test marking and filtering

### Test Categories
```python
# Unit tests for models
@pytest.mark.parametrize("model_type", ["text_detection", "text_recognition"])
def test_model_inference(model_type):
    # Test individual model components
    pass

# Integration tests for pipelines
def test_ocr_pipeline():
    # Test complete OCR workflow
    pass

# Performance tests
@pytest.mark.resource_intensive
def test_inference_speed():
    # Test performance benchmarks
    pass
```

### CI/CD Integration
- **Automated Testing**: GitHub Actions integration
- **Test Reports**: Coverage and performance reporting
- **Regression Detection**: Automatic quality monitoring

## Key Dependencies and Configuration

### Testing Dependencies
- **pytest**: Modern testing framework
- **pytest-markers**: Test categorization and filtering
- **numpy/opencv**: Test data generation and manipulation
- **paddleocr**: Import and test the main package

### Configuration
- **pytest.ini**: Test configuration in `pyproject.toml`
- **Markers**: `resource_intensive` for heavy tests
- **Fixtures**: Shared test setup and teardown

### Test Data Management
- **Sample Images**: Test images for different scenarios
- **Mock Models**: Lightweight models for unit testing
- **Reference Data**: Expected outputs for validation

## Data Models

### Test Data Structures
```python
# Test image data
test_image = {
    'path': 'tests/data/test_image.jpg',
    'expected_text': 'Expected OCR result',
    'expected_boxes': [[x1, y1, x2, y2, x3, y3, x4, y4]],
    'confidence_threshold': 0.8
}

# Model test configuration
model_test_config = {
    'model_name': 'text_detection',
    'input_shape': (3, 640, 640),
    'output_format': 'detection_boxes',
    'precision': 'fp32'
}

# Performance benchmark data
benchmark_result = {
    'fps': 25.3,
    'memory_usage': 1024,  # MB
    'accuracy': 0.95,
    'latency': 0.04  # seconds
}
```

### Test Results Format
```python
# Test execution results
{
    'test_name': 'test_ocr_pipeline',
    'status': 'passed',
    'duration': 1.23,
    'assertions': 5,
    'performance_metrics': {
        'fps': 20.5,
        'accuracy': 0.93
    }
}
```

## Testing and Quality

### Test Categories

#### Unit Tests (`tests/models/`)
- **Model Component Tests**: Individual model testing
  - `test_text_detection.py` - Detection model tests
  - `test_text_recognition.py` - Recognition model tests
  - `test_layout_detection.py` - Layout analysis tests
  - `test_table_structure_recognition.py` - Table parsing tests

#### Integration Tests (`tests/pipelines/`)
- **Pipeline Tests**: End-to-end workflow testing
  - `test_ocr.py` - Main OCR pipeline
  - `test_pp_structurev3.py` - Document structure parsing
  - `test_pp_chatocrv4_doc.py` - Information extraction
  - `test_doc_understanding.py` - Document comprehension

#### Performance Tests
- **Speed Benchmarks**: Inference speed testing
- **Memory Profiling**: Resource usage monitoring
- **Accuracy Validation**: Quality regression testing

### Quality Standards
- **Code Coverage**: Minimum 80% coverage target
- **Performance Thresholds**: Speed and accuracy baselines
- **Regression Prevention**: Automated quality monitoring
- **Cross-platform Testing**: Multi-environment validation

### Test Automation
- **Continuous Integration**: Automated test execution
- **Nightly Testing**: Resource-intensive test runs
- **Release Testing**: Comprehensive validation before releases

## Frequently Asked Questions (FAQ)

### Test Execution
**Q: Tests are taking too long to run?**
A: Use `pytest tests/ -m "not resource_intensive"` to skip heavy tests during development.

**Q: How to run tests for specific functionality?**
A: Use pytest with specific paths: `pytest tests/pipelines/test_ocr.py::test_specific_function`

### Test Development
**Q: How to write tests for new models?**
A: Follow existing patterns in `tests/models/`, use fixtures for model loading, and include performance validation.

**Q: How to add test data?**
A: Place test images in appropriate subdirectories, document expected results, and use version control for consistency.

### Debugging Tests
**Q: Tests failing locally but passing in CI?**
A: Check environment differences, dependency versions, and hardware-specific issues.

**Q: How to debug test failures?**
A: Use `pytest -v -s` for verbose output, add print statements, and use pytest debugging features.

### Performance Testing
**Q: How to benchmark new features?**
A: Add performance tests with `@pytest.mark.resource_intensive`, use consistent test data, and document baselines.

## Related File List

### Model Testing
- `models/test_text_detection.py` - Text detection model tests
- `models/test_text_recognition.py` - Text recognition model tests
- `models/test_textline_orientation_classifcation.py` - Text orientation tests
- `models/test_text_image_unwarping.py` - Image unwarping tests
- `models/test_layout_detection.py` - Layout detection tests
- `models/test_table_structure_recognition.py` - Table structure tests
- `models/test_table_cells_detection.py` - Table cell detection tests
- `models/test_formula_recognition.py` - Formula recognition tests
- `models/test_doc_vlm.py` - Document VLM tests
- `models/test_seal_text_detection.py` - Seal detection tests

### Pipeline Testing
- `pipelines/test_ocr.py` - Main OCR pipeline tests
- `pipelines/test_pp_structurev3.py` - Document structure tests
- `pipelines/test_pp_chatocrv4_doc.py` - Information extraction tests
- `pipelines/test_pp_doctranslation.py` - Document translation tests
- `pipelines/test_doc_understanding.py` - Document comprehension tests
- `pipelines/test_doc_preprocessor.py` - Document preprocessing tests
- `pipelines/test_formula_recognition.py` - Formula recognition tests
- `pipelines/test_seal_rec.py` - Seal recognition tests
- `pipelines/test_table_recognition_v2.py` - Table recognition tests

### Component Testing
- `test_cls_postprocess.py` - Classification post-processing tests
- `test_iaa_augment.py` - Image augmentation tests
- `test_formula_model.py` - Formula model tests
- `test_filter_by_image_width.py` - Image filtering tests
- `test_ppstructure.py` - PP-Structure tests

### Test Infrastructure
- `__init__.py` - Test module initialization
- `testing_utils.py` - Common testing utilities and fixtures
- `models/__init__.py` - Model test module initialization
- `pipelines/__init__.py` - Pipeline test module initialization

## Change Log (Changelog)

### 2025-09-26: Module Documentation Created
- Comprehensive testing infrastructure documentation
- Test organization and execution guidelines
- Quality assurance standards and automation
- FAQ section for common testing scenarios