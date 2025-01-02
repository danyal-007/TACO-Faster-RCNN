# TACO Faster R-CNN Trash Detection

A comprehensive implementation of a Faster R-CNN model for trash detection using the TACO (Trash Annotations in Context) dataset. Built with PyTorch and torchvision.

## 🚀 Features

- Complete training pipeline for trash detection using Faster R-CNN
- Memory-efficient data handling with dynamic batch size adjustment
- Pre-trained ResNet50 backbone with customizable classification head
- Comprehensive evaluation metrics and visualization tools
- Easy-to-follow training and inference workflows

## 📁 Repository Structure

```
.
├── TACO FasterRCNN.ipynb    # Main notebook with implementation
├── data/                    # Dataset directory (create this)
│   ├── images/             # TACO dataset images
│   └── annotations.json    # Dataset annotations
├── FasterRCNN/             # Utility modules
│   └── utils.py           # Helper functions
└── TACO_FasterRCNN.pth     # Trained model weights (generated after training)
```

## 🛠️ Installation

1. Create and activate a Python environment
2. Install required dependencies:

```bash
pip install torch torchvision pandas matplotlib tqdm scikit-learn opencv-python pillow
```

## 📦 Dependencies

The implementation relies on the following Python libraries:

- **Deep Learning**: PyTorch, torchvision
- **Data Processing**: pandas, numpy, PIL, opencv-python
- **Visualization**: matplotlib, tqdm
- **Utilities**: scikit-learn, json, pickle
- **System**: os, pathlib, gc, threading, queue, time

## 🔧 Setup

1. **Dataset Preparation**:
   - Download the TACO dataset
   - Create a `data/` directory in the project root
   - Place images and `annotations.json` in the data directory

2. **Verify Structure**:
   ```
   data/
   ├── images/
   └── annotations.json
   ```

## 💻 Implementation Details

### Data Handling

- Custom `TACODataset` class with:
  - Efficient image preprocessing
  - Annotation management
  - Coordinate transformations
  - Aspect ratio preservation
- Memory-optimized data loaders with configurable batch sizes
- Train/validation/test split functionality

### Model Architecture

- Faster R-CNN with ResNet50 backbone
- Feature Pyramid Network (FPN) for multi-scale detection
- Customizable classification head
- COCO-pretrained weights

### Training Pipeline

- Dynamic batch size adjustment
- Automatic Mixed Precision (AMP) training
- Learning rate scheduling with warmup
- Gradient clipping
- Memory management optimizations
- Progress tracking and metric logging

### Evaluation Features

- Average Precision (AP) and Average Recall (AR) metrics
- Visualization of predictions vs ground truth
- Confidence threshold filtering
- Single image inference support

### Model Management

- Checkpoint saving and loading
- State dictionary preservation
- Architecture reconstruction support

## 🚦 CUDA Configuration

- Automatic GPU detection and utilization
- Memory usage monitoring
- Enhanced error reporting with `CUDA_LAUNCH_BLOCKING`
- Optimized CUDNN benchmarking

## 📝 Usage

1. Open `TACO FasterRCNN.ipynb` in Jupyter
2. Follow the step-by-step instructions
3. Ensure correct file paths before execution
4. Execute cells sequentially for:
   - Dataset loading
   - Model initialization
   - Training
   - Evaluation
   - Visualization

## 🔍 Notes

- Performance can be enhanced through:
  - Extended training epochs
  - Hyperparameter tuning
  - Batch size optimization
- Memory management is handled automatically
- Dataset statistics and category distribution are displayed during execution
- Visualization includes both predictions and ground truth

## 🤝 Contributing

Feel free to open issues or submit pull requests for improvements or bug fixes.
