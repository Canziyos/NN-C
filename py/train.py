from dense_nn import DenseNN
from dense_layer import DenseLayer
from utils import MSE
from optim import update
from activations import sigmoid_prime, sigmoid, tanh, tanh_prime

dataset = [
   ([0,0], [0]),
   ([0,1], [1]),
   ([1,0], [1]),
   ([1,1], [0])
]

num_epochs = 2000   # longer training for XOR
lr = 0.5

# 1. Build the model (n_neurons = outputs)
net = DenseNN([
    DenseLayer(input_s=2, n_neurons=2, activation=tanh, activation_prime=tanh_prime),   # hidden layer 1
    DenseLayer(input_s=2, n_neurons=2, activation=tanh, activation_prime=tanh_prime),   # hidden layer 2
    DenseLayer(input_s=2, n_neurons=1, activation=sigmoid, activation_prime=sigmoid)    # output layer
])

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

    # Print every 100 epochs (to keep logs readable)
    if epoch % 100 == 0 or epoch == num_epochs - 1:
        print(f"Epoch {epoch}, Avg Loss: {total_loss / len(dataset)}")

print("\n--- Training finished ---\n")

# 3. Testing final outputs
print("Final XOR predictions:")
for x, y_true in dataset:
    y_pred = net.forward(x)
    print(f"{x} -> Predicted: {y_pred}, Expected: {y_true}")
