# DigitRecognitionNN

## 🧠 Custom C# Neural Network for Digit Recognition

**DigitRecognitionNN** is a fully custom implementation of a neural network in pure **C#**, designed to recognize handwritten digits from the [MNIST dataset](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
. It uses no external machine learning libraries or frameworks.

This project is ideal for learning purposes, experimentation with neural network architecture, SIMD acceleration, and custom model serialization.

---

## 🗂 Project Structure

```
DigitRecognitionNN/
├── DigitRecognitionNN/         # Core library
│   ├── Data/
│   │   ├── DataHandler.cs      # CSV data loading and handling
│   │   └── mnist_train.csv     # CSV-formatted MNIST training data
│   ├── Models/
│   │   ├── NeuralNetwork.cs    # Main neural network logic
│   │   ├── DataPoint.cs        # Data + label container
│   │   ├── Matrix.cs           # SIMD-optimized matrix operations
│   │   ├── ModelData.cs        # JSON model serialization structure
│   └── Utils/
│       ├── ActivationFunction.cs # ReLU, Softmax, and derivatives
│       └── MathUtils.cs          # CrossEntropy, ArgMax, RandomWeight
│
├── TestApp/                    # Console application for training/testing
│   ├── Data/                   # (Optional) duplicate data folder
│   └── Program.cs              # Entry point
```

---

## 🔧 Requirements

- [.NET SDK 9.0+](https://dotnet.microsoft.com/en-us/download)
- `mnist_train.csv` with 784 pixel values + label per row (see format below)

---

## 🚀 Getting Started

### Option 1: Use the Prebuilt Executable (Windows)

1. Go to the [Releases](https://github.com/hslt0/Cs-Digit-Recognition-NN/releases/tag/v1.0) page.
2. Download the latest `.zip` archive.
3. Extract it and run:

```bash
TestApp.exe
```

### Option 2: Build from Source

### 1. Clone the repository

```bash
git clone https://github.com/your-username/DigitRecognitionNN.git
cd DigitRecognitionNN
```

### 2. Build the project

```bash
dotnet build
```

### 3. Run the console app

```bash
dotnet run --project TestApp
```

You will be prompted with:

> Do you want load model from JSON? (true/false):

- `true`: Load existing model from `trained_model.json`
- `false`: Train a new model from scratch

---

## 🧠 Neural Network Architecture

- **Input Layer**: 784 neurons (28 × 28 grayscale pixels)
- **Hidden Layer 1**: 16 neurons
- **Hidden Layer 2**: 16 neurons
- **Output Layer**: 10 neurons (digits 0–9)
- **Activations**:
  - ReLU for hidden layers
  - Softmax for output
- **Loss Function**: Cross-Entropy
- **Training Algorithm**: SGD with backpropagation (manually implemented)

---

## 🧮 Matrix Class

The core of all neural operations is the custom `Matrix` class.

### Features:

- Stored as a flat `float[]` array for performance
- Indexable via `matrix[row, col]`
- Supports:
  - `+`, `-` (element-wise)
  - `*` (scalar and matrix multiplication)
  - `.Transpose()`, `.Copy()`, `.ToArray()`, `.FromArray()`
  - `.ToJaggedArray()`, `.FromJaggedArray()`
- SIMD acceleration using `System.Numerics.Vector<float>`
- Used throughout the network for all forward/backward calculations

---

## 🧪 Sample Output

```
Training nn for digit recognition
Loading data... 
Training data: 50000
Test data: 10000
Do you want load model from JSON? (true/false):
false
Start training
Epoch 1/10 completed. Avg Loss: 1.4753
...
Accuracy on test data: 92.14%
Model is saved

Test with random data:
Real digit: 7, Predict digit: 7, Confidence: 0.98
Real digit: 4, Predict digit: 4, Confidence: 0.94
...
```

---

## 📁 Data Format (mnist_train.csv)

Expected format:

```
label,pixel0,pixel1,...,pixel783
5,0,0,0,...,128
0,0,0,0,...,0
...
```

- First column is the label (0–9)
- Remaining 784 columns represent grayscale pixel values (0–255)
- Each row = one flattened 28×28 image

---

## 💾 Saving and Loading the Model

The model can be saved and loaded as JSON via:

```csharp
network.SaveModel("trained_model.json");
network.LoadModel("trained_model.json");
```

Weights and biases are converted to jagged arrays (`float[][]`) using the `ModelData` class.

---

## 📊 Metrics

- **Loss**: Cross-Entropy
- **Prediction**: ArgMax of output layer
- **Accuracy**: Evaluated on test set

---

## 📜 License

This project is licensed under the MIT License.

---

## ✍️ Author

**Viktor Herasymenko**  
Created for learning, experimentation, and educational demonstration of low-level neural network logic in C#.
