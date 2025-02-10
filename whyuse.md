### **What This Project is About**  
This project is a **configurable neural network trainer** built using **PyTorch**. Instead of manually writing the model code, it **reads a YAML file (`config.yaml`)** to define the model structure, training settings, and optimizer.  

It currently:  
✅ Loads the **model architecture** from `config.yaml`  
✅ Uses the **MNIST dataset** (handwritten digits) for training  
✅ Trains a simple **feedforward neural network**  
✅ **Validates** the model’s accuracy after each training epoch  

---

### **How It Works (Step by Step)**
1. **Reads `config.yaml`** → Gets model structure (layers, activation functions, etc.).  
2. **Builds the model dynamically** → Creates a `Sequential` PyTorch model from the YAML file.  
3. **Loads the MNIST dataset** → Splits into **training** (80%) and **validation** (20%).  
4. **Trains the model** → Runs multiple epochs, updating weights using an optimizer (like Adam).  
5. **Validates the model** → Checks accuracy on unseen data after each epoch.  

---

### **What It’s Currently Doing**  
📌 Right now, it’s training a simple neural network on the **MNIST dataset**. It has:  
🔹 **2 fully connected layers** (Linear layers)  
🔹 **ReLU activation** for non-linearity  
🔹 **Dropout** to prevent overfitting  
🔹 **Adam optimizer** with a learning rate of `0.001`  
🔹 **CrossEntropyLoss** for multi-class classification  

After training, the model prints the **validation accuracy** to see how well it learned.  

**Want to Improve It?**  
- Add more layers or different activation functions in `config.yaml`  
- Change the dataset (e.g., CIFAR-10 for images)  
- Save the trained model for later use  

This project is a **beginner-friendly neural network trainer** that can be modified easily! 🚀🔥