import math
from dense_nn import DenseNN
from dense_layer import DenseLayer
from utils import MSE
from optim import update
from activations import tanh, tanh_prime, identity, identity_prime
import matplotlib.pyplot as plt
import numpy as np

# regressio§n task #

# Contains sample points from 0 to 2π
dataset = []
for i in range(50):   # 50 training samples
    x = (i / 50) * 2 * math.pi   # evenly spaced points
    y = math.sin(x)
    dataset.append(([x], [y]))   # input and output (lists)

num_epochs = 5000   
lr = 0.01

# 1. Build the model.
net = DenseNN([
    DenseLayer(input_s=1, n_neurons=10, activation=tanh, activation_prime=tanh_prime, init="xavier"),  # hidden layer
    DenseLayer(input_s=10, n_neurons=1, activation=identity, activation_prime=identity_prime, init="xavier")  # output layer (linear)
])

# Print initial weights
print("\n--- Initial Weights ---")
for i, layer in enumerate(net.layers, start=1):
    for j, neuron in enumerate(layer.neurons, start=1):
        print(f"Layer {i}, Neuron {j}: Weights={neuron.weights}, Bias={neuron.bias}")

print("\n--- Training started ---\n")

# 2. Training loop
for epoch in range(num_epochs):
    total_loss = 0
    for x, y_true in dataset:
        # Forward
        y_pred = net.forward(x)

        # Loss
        loss = MSE(y_true, y_pred)

        # Backward + Update
        grads = net.backward(y_true, y_pred)
        update(net, grads, lr)

        total_loss += loss

    if epoch % 500 == 0 or epoch == num_epochs - 1:
        print(f"Epoch {epoch}, Avg Loss: {total_loss / len(dataset)}")

print("\n--- Training finished ---\n")

# Print final weights
print("\n--- Final Weights ---")
for i, layer in enumerate(net.layers, start=1):
    for j, neuron in enumerate(layer.neurons, start=1):
        print(f"Layer {i}, Neuron {j}: Weights={neuron.weights}, Bias={neuron.bias}")

# 3. Testing on some sample points.
print("\nTesting sin(x) approximation:")
test_points = [0, math.pi/2, math.pi, 3*math.pi/2, 2*math.pi]
for x in test_points:
    y_true = math.sin(x)
    y_pred = net.forward([x])
    print(f"x={x:.2f} -> Predicted: {y_pred[0]:.4f}, Expected: {y_true:.4f}")


#test points.
x_vals = np.linspace(0, 2 * math.pi, 200)
y_true = np.sin(x_vals)
y_pred = [net.forward([x])[0] for x in x_vals]

# Plot
plt.figure(figsize=(8,5))
plt.plot(x_vals, y_true, label="True sin(x)", color="blue")
plt.plot(x_vals, y_pred, label="Predicted", color="red", linestyle="--")
plt.scatter([0, math.pi/2, math.pi, 3*math.pi/2, 2*math.pi],
            [0, 1, 0, -1, 0], color="black", zorder=5, label="Checkpoints")

plt.title("sin(x) approximation by DenseNN")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
