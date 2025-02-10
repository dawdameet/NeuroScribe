# NEUROSCRIBE - AI-Native Programming Language

NEUROSCRIBE is a YAML-like AI-native programming language designed for defining and training machine learning models with minimal Python code. With NEUROSCRIBE, you can specify your entire model architecture, dataset, training parameters, and optimizations in a `.meet` configuration file, making AI development more intuitive and efficient.

## 🚀 Features

### **Current Version: v1.0.1**
- ✅ Define and train any neural network model entirely using `.meet` files.
- ✅ Supports dataset loading, preprocessing, and training via configuration.
- ✅ Self-sufficient: No need to modify Python code to experiment with different models.
- ✅ Handles categorical encoding, missing values, and validation split automatically.
- ✅ Seamless integration with PyTorch.

### **Changelogs**
#### **v0.2.0**
- ✅ Batch Normalization for stabilized training.
- ✅ LeakyReLU to prevent dead neurons.
- ✅ Improved Dropout for overfitting reduction.
- ✅ Early Stopping for optimized training.
- ✅ Automatic Model Saving.
- ✅ More flexibility in activation functions, optimizers, and dropout settings.

#### **v0.3.0**
- ✅ Xavier/He Weight Initialization.
- ✅ ReduceLROnPlateau for adaptive learning rate adjustment.
- ✅ Gradient Clipping to prevent exploding gradients.
- ✅ AdamW Optimizer with weight decay.
- ✅ Dataset switching (MNIST, CIFAR-10, etc.).
- ✅ Data augmentation for CIFAR-10.
- ✅ Improved logging and debugging.

---

## 📜 Quick Start

### **Installation**
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/NEUROSCRIBE.git
   cd NEUROSCRIBE
   ```
2. Create a virtual environment and install dependencies:
   ```sh
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

### **Usage**
Define your model in a `.meet` file (e.g., `config.meet`):
```yaml
model:
  type: "MLP"
  layers: [128, 64, 10]
  activation: "relu"
optimizer:
  type: "adam"
  learning_rate: 0.001
dataset:
  path: "mnist_train.csv"
  features: [c0, c1, ..., c63]
  target_column: "target"
```

Run training with:
```sh
python stable.py config.meet
```

---

## 🤖 How NEUROSCRIBE Works
1. Parses the `.meet` file (YAML-based syntax).
2. Loads dataset, preprocesses features, and applies encoding.
3. Defines the neural network architecture dynamically.
4. Trains the model using the specified optimizer and loss function.
5. Saves the trained model for inference.

---

## 💡 Why NEUROSCRIBE?
🔹 **No Python Code Needed** – Define models purely using `.meet` files.
🔹 **Flexible & Modular** – Switch datasets, architectures, and training settings seamlessly.
🔹 **Optimized for AI/ML** – Native support for deep learning best practices.

---

## 📌 Roadmap
🔜 Custom AI model inference from `.meet` files.
🔜 Support for RNNs, CNNs, and Transformers.
🔜 AutoML features for hyperparameter tuning.

---

## 🤝 Contributing
We welcome contributions! Feel free to fork the repo, create a new branch, and submit a PR. 🚀

---

## 📄 License
MIT License. See [LICENSE](LICENSE) for details.

---

## 🌟 Show Your Support
If you find this project useful, give it a ⭐ on GitHub and share it with others!
