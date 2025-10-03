# Vision AI Fundamentals - Digit Recognizer

A comprehensive machine learning project implementing handwritten digit recognition using both neural networks from scratch and TensorFlow/Keras on the MNIST dataset.
<img width="1452" height="593" alt="Sample MNIST image" src="https://github.com/user-attachments/assets/8cc57d6d-3927-4e5c-9799-065545268af1" /> <img width="1439" height="593" alt="Model Prediction" src="https://github.com/user-attachments/assets/2085f87e-4c0f-4439-a716-863904c2da11" /><img width="1050" height="390" alt="test with random number (a)" src="https://github.com/user-attachments/assets/5458ceca-d379-4626-a599-d639f818b850" />

## Project Overview

This project demonstrates fundamental computer vision and deep learning concepts by building a digit recognition system that achieves 95%+ accuracy. It includes two implementations:
- Neural Network built from scratch using only NumPy
- Optimized TensorFlow/Keras model with batch normalization and dropout

## Features

- Complete neural network implementation from scratch (forward/backward propagation)
- Professional TensorFlow model with modern architecture
- Interactive prediction interface for custom images
- Comprehensive data visualization and analysis
- Model performance comparison and confusion matrix
- Trained on 60,000 MNIST images

## Dataset

**MNIST Dataset**: 70,000 grayscale images of handwritten digits (0-9)
- Training set: 55,000 images
- Validation set: 5,000 images  
- Test set: 10,000 images
- Image size: 28x28 pixels

## Tech Stack

- **Python 3.7+**
- **TensorFlow/Keras** - Deep learning framework
- **NumPy** - Numerical computing
- **Matplotlib** - Data visualization
- **Pillow** - Image processing
- **Google Colab** - Development environment

## Getting Started

### Option 1: Run in Google Colab (Recommended)

1. Click the badge below to open in Colab:

   [![Open In Colab](https://colab.research.google.com/drive/18ssr4-mcqJbDhCSCGP_X1uqPi0loRwmL?usp=sharing)

2. Run all cells in sequence
3. Upload custom digit images for testing

### Option 2: Local Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/Digit-Recognizer.git
cd Digit-Recognizer
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the notebook:
```bash
jupyter notebook Digit_Recognizer.ipynb
```

## Project Structure

```
digit-recognizer/
│
├── digit_recognizer.ipynb    # Main Colab notebook
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── images/                    # Sample images and screenshots
│   ├── sample_predictions.png
│   └── training_curves.png
└── models/                    # Saved model weights (optional)
    ├── nn_scratch_model.pkl
    └── tf_model.h5
```

## Model Architecture

### Neural Network from Scratch
- Input Layer: 784 neurons (28x28 flattened)
- Hidden Layer 1: 128 neurons (ReLU activation)
- Hidden Layer 2: 64 neurons (ReLU activation)
- Output Layer: 10 neurons (Softmax activation)

### TensorFlow Model
- Input Layer: 784 neurons
- Dense Layer: 256 neurons + BatchNorm + Dropout (0.3)
- Dense Layer: 128 neurons + BatchNorm + Dropout (0.2)
- Dense Layer: 64 neurons + Dropout (0.2)
- Output Layer: 10 neurons (Softmax)

## Results

| Model | Test Accuracy | Training Time |
|-------|---------------|---------------|
| Neural Network (Scratch) | 96.5% | ~5 minutes |
| TensorFlow Model | 98.2% | ~3 minutes |

## Usage Examples

### Training the Model

The notebook automatically trains both models when executed. You can adjust hyperparameters:

```python
# Neural Network from Scratch
nn_scratch = NeuralNetworkScratch([784, 128, 64, 10])
nn_scratch.train(x_train, y_train, epochs=30, learning_rate=0.1)

# TensorFlow Model
tf_model = build_tensorflow_model()
tf_model.fit(x_train, y_train, epochs=20, batch_size=128)
```

### Making Predictions

```python
# Upload and predict custom image
uploaded = files.upload()
for filename, content in uploaded.items():
    predict_custom_image({'content': content}, tf_model, model_type='tf')
```

## Key Learning Outcomes

- Understanding of neural network fundamentals (forward/backward propagation)
- Implementation of gradient descent optimization
- Experience with TensorFlow/Keras framework
- Data preprocessing and normalization techniques
- Model evaluation and performance metrics
- Visualization of training dynamics

## Future Improvements

- [ ] Implement Convolutional Neural Networks (CNN)
- [ ] Add data augmentation techniques
- [ ] Deploy as web application using Streamlit
- [ ] Implement real-time drawing canvas
- [ ] Add model export for mobile deployment
- [ ] Support for other digit datasets (e.g., SVHN)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## Acknowledgments

- MNIST Dataset: Yann LeCun, Corinna Cortes, and Christopher J.C. Burges
- TensorFlow/Keras documentation and tutorials
- Inspired by Andrew Ng's Deep Learning Specialization


