# ONNX Model Export Guide

This guide explains how to export your trained PyTorch models to ONNX format for deployment in C++ and OpenCV applications.

## Quick Start

### Export the default best model:
```bash
python export_to_onnx.py
```

This will export `checkpoints/best_model.pth` to `checkpoints/best_model.onnx`.

## Command-Line Options

### Export a specific model:
```bash
python export_to_onnx.py --model checkpoints/epoch_10.pth
```

### Specify custom output path:
```bash
python export_to_onnx.py --model checkpoints/best_model.pth --output models/qrcode_classifier.onnx
```

### Change input image size (e.g., 224x224):
```bash
python export_to_onnx.py --input-size 224
```

### Use a different ONNX opset version:
```bash
python export_to_onnx.py --opset 13
```

### Disable dynamic batch size:
```bash
python export_to_onnx.py --no-dynamic
```

### Quiet mode (minimal output):
```bash
python export_to_onnx.py --quiet
```

## Using from Python Code

You can also use the export function directly in your Python code:

```python
from src.utils import export_to_onnx

# Export default model
onnx_path = export_to_onnx()

# Export with custom settings
onnx_path = export_to_onnx(
    model_path='checkpoints/epoch_10.pth',
    output_path='models/qrcode_classifier.onnx',
    input_size=(3, 224, 224),
    opset_version=13
)
```

## Using ONNX Model in C++ with OpenCV

Once you have exported your model to ONNX format, you can use it in C++ with OpenCV's DNN module:

```cpp
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

int main() {
    // Load the ONNX model
    cv::dnn::Net net = cv::dnn::readNetFromONNX("checkpoints/best_model.onnx");
    
    // Load and preprocess image
    cv::Mat image = cv::imread("test_image.jpg");
    
    // Create blob from image (normalize and resize)
    cv::Mat blob = cv::dnn::blobFromImage(
        image,
        1.0/255.0,                              // Scale factor
        cv::Size(350, 350),                     // Target size
        cv::Scalar(0.485, 0.456, 0.406),       // Mean values (ImageNet)
        true,                                   // Swap RB channels
        false                                   // Crop
    );
    
    // Note: OpenCV uses different normalization, you may need to adjust
    // the mean and std values based on your training configuration
    
    // Set input and run inference
    net.setInput(blob);
    cv::Mat output = net.forward();
    
    // Get prediction
    cv::Point classIdPoint;
    double confidence;
    cv::minMaxLoc(output.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
    int classId = classIdPoint.x;
    
    std::cout << "Predicted class: " << classId << std::endl;
    std::cout << "Confidence: " << confidence << std::endl;
    
    return 0;
}
```

### Important Notes for C++ Deployment:

1. **Normalization**: The model expects ImageNet-normalized inputs:
   - Mean: [0.485, 0.456, 0.406]
   - Std: [0.229, 0.224, 0.225]
   
   You may need to manually apply std normalization after using `blobFromImage`.

2. **Input Size**: Make sure to use the same input size as during training (default: 350x350).

3. **Color Channels**: The model expects RGB images, but OpenCV loads images in BGR format. Use `swapRB=true` in `blobFromImage`.

4. **Output Format**: The model outputs logits (raw scores). Apply softmax if you need probabilities:
   ```cpp
   cv::exp(output, output);
   cv::Scalar sum = cv::sum(output);
   output /= sum[0];  // Now contains probabilities
   ```

## Compilation Example

To compile the C++ example:

```bash
g++ -o inference inference.cpp `pkg-config --cflags --libs opencv4`
```

Or with CMake:

```cmake
find_package(OpenCV REQUIRED)
add_executable(inference inference.cpp)
target_link_libraries(inference ${OpenCV_LIBS})
```

## Troubleshooting

### Model file not found
Make sure you have trained a model first:
```bash
python main.py
```

### ONNX package not installed
Install the ONNX package for model verification:
```bash
pip install onnx
```

### OpenCV DNN module not available
Make sure you have OpenCV compiled with DNN support:
```bash
python -c "import cv2; print(cv2.dnn.getAvailableBackends())"
```

## Additional Resources

- [ONNX Documentation](https://onnx.ai/)
- [OpenCV DNN Module](https://docs.opencv.org/master/d2/d58/tutorial_table_of_content_dnn.html)
- [PyTorch ONNX Export](https://pytorch.org/docs/stable/onnx.html)
