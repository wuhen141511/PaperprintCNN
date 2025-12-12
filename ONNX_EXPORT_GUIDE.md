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

### ⚠️ Important: Correct Preprocessing

**The model expects the SAME preprocessing as PyTorch training:**
```
normalized = (pixel / 255.0 - mean) / std
```
Where:
- mean = [0.485, 0.456, 0.406]
- std = [0.229, 0.224, 0.225]

**Problem**: OpenCV's `blobFromImage` can only do `(pixel * scale - mean)`, it **cannot divide by std**.

**Solution**: You must manually divide by std after using `blobFromImage`.

### Complete C++ Example

```cpp
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

int main() {
    // Load the ONNX model
    cv::dnn::Net net = cv::dnn::readNetFromONNX("checkpoints/best_model.onnx");
    
    // Check if model loaded successfully
    if (net.empty()) {
        std::cerr << "Failed to load ONNX model" << std::endl;
        return -1;
    }
    
    // Load and preprocess image
    cv::Mat image = cv::imread("test_image.jpg");
    if (image.empty()) {
        std::cerr << "Failed to load image" << std::endl;
        return -1;
    }
    
    // CRITICAL: Correct preprocessing to match PyTorch
    // Step 1: Convert BGR to RGB and resize
    cv::Mat rgb_image;
    cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
    
    // Step 2: Create blob with scale=1/255 and ImageNet mean
    // Note: blobFromImage does (pixel * scalefactor - mean)
    cv::Mat blob = cv::dnn::blobFromImage(
        rgb_image,
        1.0/255.0,                              // Scale to [0, 1]
        cv::Size(224, 224),                     // Target size (must match training)
        cv::Scalar(0.485, 0.456, 0.406),       // ImageNet mean (RGB order)
        false,                                  // swapRB=false (already converted to RGB)
        false                                   // crop=false
    );
    
    // Step 3: CRITICAL - Divide by std (OpenCV can't do this automatically)
    // PyTorch normalization: (x - mean) / std
    // After blobFromImage we have: (x - mean)
    // Now we need to divide by std
    cv::Scalar std_values(0.229, 0.224, 0.225);  // ImageNet std (RGB order)
    
    // Divide each channel by its std
    // blob shape is [1, 3, 224, 224] - need to divide each channel
    float* blob_data = (float*)blob.data;
    int channel_size = 224 * 224;  // pixels per channel
    
    for (int c = 0; c < 3; c++) {
        float std_val = (c == 0) ? std_values[0] : (c == 1) ? std_values[1] : std_values[2];
        for (int i = 0; i < channel_size; i++) {
            blob_data[c * channel_size + i] /= std_val;
        }
    }
    
    // Step 4: Run inference
    net.setInput(blob);
    cv::Mat output = net.forward();
    
    // Step 5: Apply softmax to get probabilities
    cv::Mat probs;
    cv::exp(output, probs);
    cv::Scalar sum = cv::sum(probs);
    probs /= sum[0];
    
    // Step 6: Get prediction
    cv::Point classIdPoint;
    double confidence;
    cv::minMaxLoc(probs.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
    int classId = classIdPoint.x;
    
    // Step 7: Print results
    std::cout << "Predicted class: " << classId << std::endl;
    std::cout << "Confidence: " << (confidence * 100.0) << "%" << std::endl;
    std::cout << "\nClass probabilities:" << std::endl;
    for (int i = 0; i < probs.cols; i++) {
        std::cout << "  Class " << i << ": " 
                  << (probs.at<float>(0, i) * 100.0) << "%" << std::endl;
    }
    
    return 0;
}
```

### Alternative: Helper Function for Preprocessing

```cpp
// Helper function to preprocess image exactly like PyTorch
cv::Mat preprocessImage(const cv::Mat& image, int target_size = 224) {
    // Convert BGR to RGB
    cv::Mat rgb_image;
    cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
    
    // Resize
    cv::Mat resized;
    cv::resize(rgb_image, resized, cv::Size(target_size, target_size));
    
    // Convert to float and normalize to [0, 1]
    cv::Mat float_image;
    resized.convertTo(float_image, CV_32F, 1.0/255.0);
    
    // ImageNet normalization
    cv::Scalar mean(0.485, 0.456, 0.406);
    cv::Scalar std(0.229, 0.224, 0.225);
    
    // Subtract mean
    float_image -= mean;
    
    // Divide by std (per channel)
    std::vector<cv::Mat> channels(3);
    cv::split(float_image, channels);
    channels[0] /= std[0];  // R
    channels[1] /= std[1];  // G
    channels[2] /= std[2];  // B
    cv::merge(channels, float_image);
    
    // Convert to blob format [1, 3, H, W]
    cv::Mat blob = cv::dnn::blobFromImage(float_image, 1.0, 
                                          cv::Size(target_size, target_size),
                                          cv::Scalar(0, 0, 0), false, false);
    
    return blob;
}

// Usage:
cv::Mat image = cv::imread("test.jpg");
cv::Mat blob = preprocessImage(image, 224);
net.setInput(blob);
cv::Mat output = net.forward();
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
