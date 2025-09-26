[Root Directory](../CLAUDE.md) > **deploy**

# Deployment Solutions Module

## Module Responsibilities

The `deploy/` directory provides production-ready deployment solutions for PaddleOCR across different platforms, environments, and integration scenarios. It enables seamless transition from research and development to production deployment.

**Key Functions:**
- Multi-platform deployment support (C++, Python, Mobile, Cloud)
- Hardware acceleration optimization (CPU, GPU, XPU, NPU)
- Service integration and API deployment
- Model optimization and compression
- Cross-platform compatibility solutions

## Entry and Startup

### Deployment Options
- **C++ Inference**: `cpp_infer/` - High-performance C++ implementation
- **Mobile Deployment**: `android_demo/`, `ios_demo/` - Mobile platform integration
- **Service Deployment**: `hubserving/` - PaddleHub-based service deployment
- **Cloud Deployment**: Various cloud platform integrations
- **Edge Computing**: `lite/` - Paddle Lite for edge devices

### Container Solutions
- **Docker**: `docker/` - Containerized deployment
- **Kubernetes**: Cloud-native orchestration support
- **Service Mesh**: Integration with service mesh architectures

## External Interfaces

### C++ Inference API
```cpp
#include "ocr_det.h"
#include "ocr_rec.h"
#include "ocr_cls.h"

// Detection
OCRDet det("model_dir", use_gpu, gpu_id);
std::vector<OCRPredictResult> det_results = det.ocr(img_list);

// Recognition
OCRRec rec("model_dir", use_gpu, gpu_id);
std::vector<OCRPredictResult> rec_results = rec.ocr(img_list);
```

### Python Service API
```python
# PaddleHub serving
hub install deploy/hubserving/ocr_system/
hub serving start -m ocr_system

# REST API call
import requests
response = requests.post(
    'http://127.0.0.1:8866/predict/ocr_system',
    files={'images': open('test.jpg', 'rb')}
)
```

### Mobile Integration API
```java
// Android integration
OCRPredictorNative predictor = new OCRPredictorNative(
    detModelPath, recModelPath, clsModelPath,
    cpuThreadNum, cpuPowerMode
);
ArrayList<OcrResultModel> results = predictor.runModel(bitmap);
```

## Key Dependencies and Configuration

### C++ Dependencies
- **OpenCV**: Image processing and computer vision
- **PaddleInference**: PaddlePaddle inference library
- **CUDA/TensorRT**: GPU acceleration (optional)
- **OpenMP**: Parallel processing

### Python Service Dependencies
- **PaddleHub**: Service framework
- **Flask/FastAPI**: Web service frameworks
- **Gunicorn**: WSGI HTTP server
- **Docker**: Containerization

### Mobile Dependencies
- **Paddle Lite**: Mobile inference framework
- **Android NDK**: Android native development
- **iOS SDK**: iOS development tools

### Hardware Optimization
- **Intel MKL-DNN**: CPU optimization
- **NVIDIA TensorRT**: GPU acceleration
- **Kunlun XPU**: Baidu XPU support
- **Ascend NPU**: Huawei NPU support

## Data Models

### C++ Data Structures
```cpp
struct OCRPredictResult {
    std::vector<std::vector<int>> box;
    std::string text;
    float confidence;
    float text_score;
    std::vector<std::vector<float>> raw_scores;
};

struct OCRConfig {
    bool use_gpu = false;
    int gpu_id = 0;
    int cpu_math_library_num_threads = 10;
    bool use_mkldnn = true;
    // ... more config options
};
```

### Service Response Format
```json
{
    "results": [
        {
            "text": "detected text",
            "confidence": 0.95,
            "text_region": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        }
    ],
    "status": "success",
    "processing_time": 0.123
}
```

### Mobile Data Models
- **Bitmap/UIImage**: Native image formats
- **Model Results**: Text detection and recognition results
- **Configuration**: Model paths and inference parameters

## Testing and Quality

### Performance Testing
- **Throughput Testing**: QPS measurements under load
- **Latency Testing**: Response time analysis
- **Resource Usage**: CPU, memory, and GPU utilization
- **Concurrent Testing**: Multi-client stress testing

### Accuracy Validation
- **Cross-Platform Consistency**: Results comparison across platforms
- **Precision Validation**: FP32 vs FP16 vs INT8 accuracy
- **Model Optimization**: Pruning and quantization impact

### Integration Testing
- **API Testing**: RESTful service endpoint testing
- **Container Testing**: Docker deployment validation
- **Mobile Testing**: Device-specific testing

### Production Readiness
- **Load Testing**: High-traffic scenario testing
- **Failover Testing**: Error handling and recovery
- **Monitoring**: Performance metrics and alerting

## Frequently Asked Questions (FAQ)

### C++ Deployment
**Q: How to optimize C++ inference speed?**
A: Enable MKL-DNN, use TensorRT, optimize OpenCV, and tune thread count.

**Q: Memory usage too high in C++ deployment?**
A: Use model quantization, reduce input resolution, and optimize memory allocation.

### Service Deployment
**Q: How to scale OCR service horizontally?**
A: Use load balancers, container orchestration, and stateless service design.

**Q: API response time too slow?**
A: Implement batch processing, use async processing, and optimize model loading.

### Mobile Deployment
**Q: Mobile app performance issues?**
A: Use Paddle Lite quantized models, optimize preprocessing, and implement background processing.

**Q: Model size too large for mobile?**
A: Apply model compression, pruning, and use mobile-specific model variants.

### Cloud Deployment
**Q: How to deploy on Kubernetes?**
A: Use provided Docker images, configure resource limits, and implement health checks.

**Q: Auto-scaling configuration?**
A: Monitor CPU/memory metrics, set appropriate scaling policies, and use horizontal pod autoscaling.

## Related File List

### C++ Inference
- `cpp_infer/src/` - C++ inference implementation
  - `ocr_det.cpp/.h` - Text detection
  - `ocr_rec.cpp/.h` - Text recognition
  - `ocr_cls.cpp/.h` - Text classification
  - `main.cpp` - Main inference program
- `cpp_infer/tools/` - Build and utility tools
- `cpp_infer/docs/` - C++ deployment documentation

### Service Deployment
- `hubserving/` - PaddleHub service modules
  - `ocr_system/` - Complete OCR system service
  - `ocr_det/` - Detection-only service
  - `ocr_rec/` - Recognition-only service
  - `structure_*` - Document structure services
- `pdserving/` - PaddleServing deployment (if available)

### Mobile Deployment
- `android_demo/` - Android application demo
  - `app/src/main/` - Android source code
  - `app/libs/` - Native libraries
- `ios_demo/` - iOS application demo
- `lite/` - Paddle Lite deployment guides

### Optimization and Compression
- `slim/` - Model optimization tools
  - `quantization/` - Model quantization
  - `prune/` - Model pruning
  - `auto_compression/` - Automatic compression tools

### Format Conversion
- `paddle2onnx/` - ONNX format conversion
- `avh/` - ARM Virtual Hardware deployment

### Container and Cloud
- `docker/` - Docker deployment configurations
- `paddlecloud/` - PaddleCloud deployment (if available)

### Documentation and Guides
- `README.md` - Main deployment guide
- `README_ch.md` - Chinese deployment guide
- Platform-specific README files

## Change Log (Changelog)

### 2025-09-26: Module Documentation Created
- Comprehensive deployment solutions documentation
- Multi-platform deployment guidelines
- Performance optimization and testing strategies
- FAQ section for common deployment challenges