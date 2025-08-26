# Neural Network in C.

Practising by implementing a **feedforward neural network**, so far, in C. The network uses dynamic memory allocation to support different architectures.

- **Architecture:** 3 layers (input, hidden, and output).
- **Activation Function:** Sigmoid
- **Memory Management:**.


## **Structure of c directory**
| **File**            | **Purpose**                                               |
|--------------------|-----------------------------------------------------------|
| `main.c`           | Entry point for testing the network with multiple inputs.  |
| `Model.h`          | Core structure definition for the neural network.          |
| `Model.c`          | Implements forward propagation logic.                      |
| `init.h`, `init.c` | Network initialization and memory allocation.              |
| `math_utils.h`, `math_utils.c` | Contains **matrix multiplication** and **sigmoid activation**. |
| `debug_utils.h`, `debug_utils.c` | Functions for printing activations and debugging.          |
| `CMakeLists.txt`   | Build configuration for the project.                       |

---


## Build and Run

### **1. Clone the repository**
```bash
git clone <repository-url>
cd <repository-folder>
```

### **2. Create a build directory and run CMake**
```bash
mkdir build
cd build
cmake ..
```

### **3. Build the project**
```bash
cmake --build .
```

### **4. Run the executable**
```bash
Debug\nn_executable.exe  # Windows
./nn_executable          # Linux/Mac
```

## **How It Works**
### **Step 1: Initialization**
- The network is initialized using **random weights and biases** between -0.5 and 0.5.
- The architecture is defined in **`main.c`** using the **`NeuralNetwork` struct.**

### **Step 2: Forward Propagation**
- For each layer, the activations are computed using **matrix multiplication**:
  
  \[ z = W \cdot A + b \]
  
  - The sigmoid activation function is applied to compute the output of each neuron:

  \[ \text{activation} = \frac{1}{1 + e^{-z}} \]

### **Step 3: Output**
- The network propagates the input through the hidden layer to the output layer, and the activations are printed using **debug utilities.**

### **Step 4: Cleanup**
- Dynamically allocated memory is safely deallocated using **`free_network()`** to avoid memory leaks.


## **Test Cases and Results**

| **Test Case**         | **Input**         | **Hidden Layer Activations**                            | **Output Activation** |
|----------------------|-------------------|--------------------------------------------------------|-----------------------|
| **Test Case 1**       | `{0.5, -0.3}`     | `{0.486, 0.520, 0.451, 0.481}`                         | `0.440`               |
| **Test Case 2**       | `{1.0, 0.8}`      | `{0.578, 0.448, 0.347, 0.405}`                         | `0.468`               |
| **Test Case 3**       | `{-0.2, 0.5}`     | `{0.534, 0.467, 0.503, 0.492}`                         | `0.441`               |
| **Test Case 4**       | `{0.0, 0.0}`      | `{0.500, 0.500, 0.500, 0.500}`                         | `0.436`               |
| **Test Case 5**       | `{-1.0, -0.8}`    | `{0.422, 0.552, 0.653, 0.595}`                         | `0.405`               |

---

## **Coming Improvements**
1. **Implement Backpropagation:** gradient descent and backpropagation to train the network.
2. **Expand the Architecture:** Test with more layers and neurons.
3. **Optimize Memory Usage:** static memory allocation for embedded systems.

---

## 🔗 **License**
[MIT License](LICENSE)
