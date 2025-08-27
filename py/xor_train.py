from dense_nn import DenseNN
from dense_layer import DenseLayer
from utils import MSE
from optim import update
from activations import sigmoid, sigmoid_prime, tanh, tanh_prime

dataset = [
   ([0,0], [0]),
   ([0,1], [1]),
   ([1,0], [1]),
   ([1,1], [0])
]

num_epochs = 2000 
lr = 0.5

# 1. Build the model (n_neurons = outputs)
net = DenseNN([
    DenseLayer(input_s=2, n_neurons=2, activation=tanh, activation_prime=tanh_prime, init="xavier"),   # hidden layer 1
    DenseLayer(input_s=2, n_neurons=2, activation=tanh, activation_prime=tanh_prime, init="xavier"),   # hidden layer 2
    DenseLayer(input_s=2, n_neurons=1, activation=sigmoid, activation_prime=sigmoid_prime, init="xavier")  # output layer
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

    # keep logs readable.
    if epoch % 100 == 0 or epoch == num_epochs - 1:
        print(f"Epoch {epoch}, Avg Loss: {total_loss / len(dataset)}")

print("\n--- Training finished ---\n")

# Print final weights
print("\n--- Final Weights ---")
for i, layer in enumerate(net.layers, start=1):
    for j, neuron in enumerate(layer.neurons, start=1):
        print(f"Layer {i}, Neuron {j}: Weights={neuron.weights}, Bias={neuron.bias}")

# 3. Testing final outputs
print("\nFinal XOR predictions:")
for x, y_true in dataset:
    y_pred = net.forward(x)
    print(f"{x} -> Predicted: {y_pred}, Expected: {y_true}")
