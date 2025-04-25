# Implementation of MNIST Classification Using Multilayer Perceptron and Convolutional Neural Networks

## Introduction to Deep Learning CS4851/6851 - Dr. Yue Wang

### Reza Mansouri `rmansouri1[at]student.gsu.edu`
### Pranjal Patil `ppatil8[at]student.gsu.edu`
### Ven Mendelssohn `gmendelssohn1[at]student.gsu.edu`

## 1. Introduction

### 1.1 Background and Motivation
Deep learning has revolutionized fields such as computer vision, natural language processing, and signal processing. Among its many architectures, **Multilayer Perceptron (MLP)** and **Convolutional Neural Network (CNN)** are foundational. MLPs serve as universal function approximators, while CNNs excel in spatial data processing, especially image classification.

This project explores the construction of both MLP and CNN models **from scratch using only NumPy**. By re-implementing these architectures without relying on high-level libraries like PyTorch or TensorFlow, we gain a deeper understanding of how data flows through layers and how gradients are propagated during learning.

### 1.2 Problem Definition
Our task is to classify handwritten digits from the MNIST dataset using two approaches:
- A fully connected MLP with one hidden layer.
- A CNN incorporating convolutional and fully connected layers.

### 1.3 System Overview
The system includes the following components:
1. **Data preprocessing** using TorchVision, normalization, and batching.
2. **MLP implementation:** input layer, one hidden layer, and an output softmax layer.
3. **CNN implementation:** convolution, activation, flattening, and a fully connected output layer.
4. **Manual backpropagation** for both architectures, with weight updates using vanilla gradient descent.
5. **Evaluation** of model performance on the MNIST test subset.

---

## 2. Methodology

### 2.1 MLP Architecture

The MLP processes a flattened 784-dimensional input (28×28 image) through:

- One hidden layer with a variable number of neurons,
- An output layer with 10 neurons,
- Sigmoid activation in the hidden layer and softmax activation in the output layer.

**MLP Forward Pass:**
w
```python
def forward(self, x):
    self.hidden_layer_input = np.dot(x, self.weights1) + self.bias1
    self.hidden_layer_output = sigmoid(self.hidden_layer_input)
    self.output_layer_input = np.dot(self.hidden_layer_output, self.weights2) + self.bias2
    outputs = softmax(self.output_layer_input)
    return outputs
```

Remarks:

- Proper initialization of weights helps avoid vanishing gradients when using sigmoid.
- The hidden layer activation is sigmoid, which can saturate if not carefully tuned.

**MLP Backward Pass:**

```python
def backward(self, x, y, pred):
    one_hot_y = np.eye(self.output_size)[y]
    d_output = pred - one_hot_y
    d_weights2 = np.dot(self.hidden_layer_output.T, d_output)
    d_bias2 = np.sum(d_output, axis=0)
    d_hidden = np.dot(d_output, self.weights2.T) * (self.hidden_layer_output * (1 - self.hidden_layer_output))
    d_weights1 = np.dot(x.T, d_hidden)
    d_bias1 = np.sum(d_hidden, axis=0)

    self.weights2 -= self.lr * d_weights2
    self.bias2 -= self.lr * d_bias2
    self.weights1 -= self.lr * d_weights1
    self.bias1 -= self.lr * d_bias1
```

Remarks:

- Correct computation of the derivative of the sigmoid function is essential for stable training.
- One-hot encoding must match the softmax output dimensions to compute loss gradients properly.

Example `MLP_3.py` output with hidden size = 256:

```
The number of training data: 60000
The number of testing data: 10000
Epoch 1/100, Train Loss: 0.8850 Train Accuracy: 0.7027
Epoch 2/100, Train Loss: 0.2825 Train Accuracy: 0.9169
Epoch 3/100, Train Loss: 0.2201 Train Accuracy: 0.9349
Epoch 4/100, Train Loss: 0.1861 Train Accuracy: 0.9439
Epoch 5/100, Train Loss: 0.1630 Train Accuracy: 0.9516
...
Epoch 96/100, Train Loss: 0.0044 Train Accuracy: 0.9997
Epoch 97/100, Train Loss: 0.0042 Train Accuracy: 0.9997
Epoch 98/100, Train Loss: 0.0042 Train Accuracy: 0.9997
Epoch 99/100, Train Loss: 0.0041 Train Accuracy: 0.9997
Epoch 100/100, Train Loss: 0.0041 Train Accuracy: 0.9997
Test Accuracy: 0.9701
```

---

### 2.2 CNN Architecture

The CNN model processes a 28×28 grayscale image through:

- A single convolutional layer with multiple filters,
- ReLU activation after convolution,
- Flattening,
- A fully connected layer with 10 outputs.

No pooling operation is used; convolution alone reduces spatial dimensions.

**CNN Forward Pass:**

```python
def forward(self, x):
    self.x = x
    self.conv = self.convolve(x, self.filters)
    self.relu = relu(self.conv)
    self.flat = self.relu.reshape(x.shape[0], -1)
    self.logits = np.dot(self.flat, self.fc_weights) + self.fc_bias
    self.probs = softmax(self.logits)
    return self.probs
```

Remarks:

- Without pooling, the convolution operation itself is responsible for reducing spatial dimensions.
- ReLU activation prevents vanishing gradients and accelerates training.

**Vectorized CNN Convolution (Efficient Implementation):**

```python
def convolve(self, x, filters):
    B, _, H, W = x.shape
    F, _, KH, KW = filters.shape
    OH, OW = H - KH + 1, W - KW + 1
    patches = np.lib.stride_tricks.as_strided(
        x,
        shape=(B, OH, OW, KH, KW),
        strides=(x.strides[0], x.strides[2], x.strides[3], x.strides[2], x.strides[3]),
        writeable=False
    ).reshape(B, OH * OW, KH * KW)
    filters_flat = filters.reshape(F, KH * KW)
    out = np.matmul(patches, filters_flat.T)
    return out.transpose(0, 2, 1).reshape(B, F, OH, OW)
```

Remarks:

- Matrix multiplication of reshaped patches with filters avoids four nested loops, greatly improving computational speed.
- Careful use of `stride_tricks` ensures that memory is shared efficiently without copying data.

**CNN Backward Pass:**

```python
def backward(self, x, y, pred):
    B = y.shape[0]
    y_onehot = np.eye(self.fc_output_size)[y]
    d_logits = (pred - y_onehot) / B

    d_fc_weights = np.dot(self.flat.T, d_logits)
    d_fc_bias = np.sum(d_logits, axis=0)
    d_flat = np.dot(d_logits, self.fc_weights.T)
    d_relu = d_flat.reshape(self.relu.shape)
    d_conv = d_relu * (self.conv > 0)

    B, F, OH, OW = d_conv.shape
    patches = np.lib.stride_tricks.as_strided(
        self.x,
        shape=(B, OH, OW, self.kernel_size, self.kernel_size),
        strides=(self.x.strides[0], self.x.strides[2], self.x.strides[3], self.x.strides[2], self.x.strides[3]),
        writeable=False
    ).reshape(B, OH * OW, self.kernel_size * self.kernel_size)

    d_out_flat = d_conv.reshape(B, F, OH * OW)

    d_filters = np.zeros_like(self.filters)
    for b in range(B):
        d_filters += np.matmul(d_out_flat[b], patches[b]).reshape(F, 1, self.kernel_size, self.kernel_size)

    self.fc_weights -= self.lr * d_fc_weights
    self.fc_bias -= self.lr * d_fc_bias
    self.filters -= self.lr * d_filters
```

Remarks:

- The backward pass through ReLU correctly zeros gradients where activations were negative.
- Convolutional gradient computation requires careful reshaping and batch handling to accumulate the correct updates.

Example `CNN_3.py` output with kernel size = 7:

```
Training samples: 60000
Testing samples: 10000
Epoch 1/5, Train Loss: 0.6728 Train Accuracy: 0.7876
Epoch 2/5, Train Loss: 0.3537 Train Accuracy: 0.8979
Epoch 3/5, Train Loss: 0.3120 Train Accuracy: 0.9103
Epoch 4/5, Train Loss: 0.2893 Train Accuracy: 0.9170
Epoch 5/5, Train Loss: 0.2736 Train Accuracy: 0.9219
Test Accuracy: 0.9252
```

### 2.3. Formulas

Cross-entropy loss:
$$
L = -\sum y_i \log(\hat{y}_i) \Rightarrow \frac{\partial L}{\partial z} = \hat{y} - y
$$

**2. Fully Connected Gradients:**

$$
\frac{\partial L}{\partial W_{fc}} = x^\top (\hat{y} - y)
$$

**3. CNN Convolution Gradient:**

$$
\frac{\partial L}{\partial W_{conv}} = \sum_{b=1}^{B} x_{patch} \cdot \delta_{conv}
$$

## 3. Evaluation

### 3.1 Experiment Settings

- **Data**: 60,000 training and 10,000 testing images from MNIST
- **MLP**: 784-input, one hidden layer with varying sizes, 10-output, softmax
- **Learning rate**: 0.01
- **Batch size**: 128
- **CNN**: 28x28-input, one convolution kernel with varying sizes, ReLU, FC(10)
- **Epochs**:
    - **MLP**: 100
    - **CNN**: 5

We performed multiple experiments on the MLP architecture by varying the `hidden_size` parameter to understand its effect on model performance.

The five values tested for `hidden_size` were: **32, 64, 128, 256, 512**.

---

### 3.2 MLP Learning Curves

Below are the training loss and accuracy plots for each hidden size:

![](mlp_results/train_accuracy.png)

![](mlp_results/train_losses.png)

### 3.3 CNN Learning Curves

Below are the training loss and accuracy plots for each hidden size:

![](cnn_results/train_accuracy.png)

![](cnn_results/train_losses.png)


### 3.4 MLP Accuracy Results

The table below shows test set accuracy for each `hidden_size` value:

| Hidden Layer Size | Test Accuracy |
|-------------------|---------------|
| 32                | 95.69%        |
| 64                |   96.94%      |
| 128               |   96.79%      |
| 256               |  ***97.01%*** |
| 512               |    96.40%     |

We found that increasing the hidden layer size generally improved accuracy, but performance plateaued beyond 256. The best result was obtained with `hidden_size = 256`.


### 3.5 CNN Accuracy Results

The table below shows test set accuracy for each `kernel_size` value:

| Kernel Size | Test Accuracy |
|-------------------|---------------|
| 3 x 3                | 91.16%        |
| 5 x 5                |   90.47%      |
| 7 x 7               |   ***92.77%***      |
| 9 x 9               |  91.29%      |
| 11 x 11               |    92.52%     |

As can be seen, the best result was obtained with `kernel_size = 7`.

---

## 4. Learning Outcomes

### 4.1 Tasks Accomplished

- Implemented both MLP and CNN architectures from scratch in NumPy.
- Derived and implemented full forward and backward propagation manually.
- Designed a full training and evaluation pipeline.
- Gained practical experience debugging gradient flow in neural nets.

### 4.2 Team Contributions

This project was a team effort by three members:

- **Reza Mansouri**: Developed the convolutional operation and CNN backpropagation logic.
- **Pranjal Patil**: Implemented the MLP architecture and trained both models.
- **Ven Mendelssohn**: Wrote the training/evaluation scripts and composed the final report.

All members contributed to testing, debugging, and system integration.

### 4.3 Challenges and Lessons Learned

- Implementing convolutional backpropagation without autograd was complex.
- Proper memory management and vectorization were crucial for speed.
- Manual implementation helped us understand how modern frameworks work under the hood.

### 4.4 Conclusion

Rebuilding MLP and CNN architectures from scratch allowed us to deeply understand the fundamental building blocks of deep learning. From activation functions to convolution operations, every detail required precise implementation and validation. Manually constructing and optimizing both models also highlighted the critical roles of proper initialization, gradient flow, and efficient computation. This project not only enhanced our confidence in the theory and application of deep learning models but also provided a strong foundation for further exploration into modern deep learning frameworks.