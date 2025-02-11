# Neural Network in C.

## **Project Overview**
This project implements a **feedforward neural network**, so far, in C. The network uses **dynamic memory allocation** to support different architectures.

- **Architecture:** 3 layers (input, hidden, and output).
- **Activation Function:** Sigmoid
- **Memory Management:** Dynamic allocation with cleanup.


## **Project Structure**
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
### **Analysis of Test Case Outputs:**

1. **Test Case 1: `{0.5, -0.3}`**
   - Hidden layer activations:  
     ```
     Activation[0] = 0.486
     Activation[1] = 0.520
     Activation[2] = 0.451
     Activation[3] = 0.481
     ```
   - Output activation:  
     ```
     Activation[0] = 0.440
     ```

   - The activations **vary across neurons** in the hidden layer, reflecting that the **weights and biases are contributing correctly.**

2. **Test Case 2: `{1.0, 0.8}` (Larger positive inputs)**
   - Hidden layer activations:  
     ```
     Activation[0] = 0.578
     Activation[1] = 0.448
     Activation[2] = 0.347
     Activation[3] = 0.405
     ```
   - Output activation:  
     ```
     Activation[0] = 0.468
     ```

   - The activations respond differently due to larger positive inputs, showing that the **network handles diverse inputs correctly.**

3. **Test Case 3: `{-0.2, 0.5}` (Mixed signs)**
   - Hidden layer activations:  
     ```
     Activation[0] = 0.534
     Activation[1] = 0.467
     Activation[2] = 0.503
     Activation[3] = 0.492
     ```
   - Output activation:  
     ```
     Activation[0] = 0.441
     ```

   - The mixed signs result in **slightly different activations**, proving that the network correctly propagates input variations.

4. **Test Case 4: `{0.0, 0.0}` (All zeros)**
   - Hidden layer activations:  
     ```
     Activation[0] = 0.500
     Activation[1] = 0.500
     Activation[2] = 0.500
     Activation[3] = 0.500
     ```
   - Output activation:  
     ```
     Activation[0] = 0.436
     ```

   Since all inputs are zero, the hidden layer activations are **solely influenced by the biases**, which results in a balanced activation of **0.5** across the hidden neurons.

5. **Test Case 5: `{-1.0, -0.8}` (Large negative inputs)**
   - Hidden layer activations:  
     ```
     Activation[0] = 0.422
     Activation[1] = 0.552
     Activation[2] = 0.653
     Activation[3] = 0.595
     ```
   - Output activation:  
     ```
     Activation[0] = 0.405
     ```
- Negative inputs cause varied activations, i.e., The network is responding properly to both positive and negative ranges.

## In general:

- The activations across the hidden and output layers **are not uniform**— they change dynamically based on the inputs, which is exactly what we expect in a properly functioning network.
- The **sigmoid activation values are in the range [0, 1]**, as expected.


## 🔧 **How to Build and Run**

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

## 🚀 **How It Works**
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

## 🌟 **Future Improvements**
1. **Implement Backpropagation:** Add gradient descent and backpropagation to train the network.
2. **Expand the Architecture:** Test the network with more layers and neurons.
3. **Optimize Memory Usage:** Implement static memory allocation for embedded systems.

---

## 🔗 **Contributing**
Contributions are welcome! Feel free to fork the project and submit pull requests.

---

## 🔗 **License**
[MIT License](LICENSE)
