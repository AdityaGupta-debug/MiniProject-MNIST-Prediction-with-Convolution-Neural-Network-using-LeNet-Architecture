# MNIST Handwritten Digit Classification using CNN with LeNet Architecture

This project implements a Convolutional Neural Network (CNN) based on the LeNet architecture to classify handwritten digits from the MNIST dataset.

## About the Dataset

The MNIST dataset is a large database of handwritten digits commonly used for training and testing machine learning algorithms. It contains:

- **Training set**: 60,000 images
- **Test set**: 10,000 images
- **Image dimensions**: 28x28 pixels (grayscale)
- **Classes**: 10 digits (0-9)

Each image is a 28x28 pixel grayscale image of a handwritten digit, normalized to fit into a 28x28 pixel bounding box and anti-aliased. The dataset is widely used as a benchmark for image classification tasks.

## Model Architecture

This implementation uses the classic LeNet architecture, which consists of:

1. **Convolutional Layer 1**: 6 filters, 5x5 kernel, tanh activation
2. **Average Pooling Layer 1**: 2x2 pool size, stride 2
3. **Convolutional Layer 2**: 16 filters, 5x5 kernel, tanh activation
4. **Average Pooling Layer 2**: 2x2 pool size, stride 2
5. **Flatten Layer**: Converts 2D feature maps to 1D
6. **Dense Layer 1**: 120 neurons, tanh activation
7. **Dense Layer 2**: 84 neurons, tanh activation
8. **Output Layer**: 10 neurons, softmax activation

## Implementation

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense

model = Sequential()
model.add(Conv2D(6, kernel_size=(5,5), padding='valid', activation='tanh', input_shape=(28,28,1)))
model.add(AveragePooling2D(pool_size=(2,2), strides=2, padding='valid'))

model.add(Conv2D(16, kernel_size=(5,5), padding='valid', activation='tanh'))
model.add(AveragePooling2D(pool_size=(2,2), strides=2, padding='valid'))

model.add(Flatten())

model.add(Dense(120, activation='tanh'))
model.add(Dense(84, activation='tanh'))

model.add(Dense(10, activation='softmax'))
```

## Results

The model achieved impressive performance on the MNIST dataset:

- **Training Duration**: 10 epochs
- **Final Accuracy**: 98.76%

This high accuracy demonstrates the effectiveness of the LeNet architecture for handwritten digit classification tasks.

## Architecture Details

The LeNet architecture used in this project follows the original design principles:

- **Feature Extraction**: Two convolutional layers with average pooling extract hierarchical features from the input images
- **Classification**: Three fully connected layers perform the final classification
- **Activation Functions**: Tanh activation is used throughout (except the output layer which uses softmax)
- **Pooling**: Average pooling is used instead of max pooling, staying true to the original LeNet design

## Key Features

- Implementation of the classic LeNet CNN architecture
- Efficient training on MNIST dataset
- High accuracy achievement in minimal epochs
- Clean and readable code structure
- Proper use of padding and pooling strategies

## Usage

1. Load and preprocess the MNIST dataset
2. Build the model using the provided architecture
3. Compile the model with appropriate loss function and optimizer
4. Train the model for 10 epochs
5. Evaluate performance on test data

This implementation serves as an excellent starting point for understanding CNN architectures and their application to image classification tasks.
