import random, matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
import optim
from dense_nn import DenseNN
from dense_layer import DenseLayer
from utils import cross_entropy
from activations import tanh, tanh_prime, softmax

# ----------------------------
# Dataset
# ----------------------------
X, y = make_circles(n_samples=300, noise=0.2, factor=0.5, random_state=42)

# standardize X.
X_arr = np.array(X)
mu = X_arr.mean(axis=0)
sigma = X_arr.std(axis=0) + 1e-8
X = [list((x - mu) / sigma) for x in X_arr]

def one_hot(y, num_classes):
    return [[1 if i == label else 0 for i in range(num_classes)] for label in y]

Y = one_hot(y, 2)
dataset = list(zip(X, Y))

# ----------------
# Plot dataset.
# ----------------
# X_plot, y_plot = zip(*zip(X, y))  # unzip
# X0 = [X_plot[i][0] for i in range(len(X_plot))]
# X1 = [X_plot[i][1] for i in range(len(X_plot))]

# plt.scatter(X0, X1, c=y, cmap=plt.cm.Spectral, s=30)
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.title("make_circles dataset")
# plt.show()

# ----------------------------
# Model
# ----------------------------

net = DenseNN([
    DenseLayer(input_s=2, n_neurons=16, activation=tanh, activation_prime=tanh_prime, init="xavier"),
    DenseLayer(input_s=16, n_neurons=8, activation=tanh, activation_prime=tanh_prime, init="xavier"),
    DenseLayer(input_s=8, n_neurons=2, activation=softmax, activation_prime=None, init="xavier")
])



# ----------------------------
# Training
# ----------------------------
optim.momentum_state = {}
num_epochs = 2000

print("\n--- Training started ---\n")
for epoch in range(num_epochs):
    random.shuffle(dataset)
    total_loss = 0.0

    for x, y_true in dataset:
        y_pred = net.forward(x)
        loss = cross_entropy(y_true, y_pred)
        grads = net.backward(y_true, y_pred)
        optim.update(net, grads, lr=0.01, momentum=0.8, clip=0.8, weight_decay=1e-4)
        total_loss += loss

    if epoch % 200 == 0 or epoch == num_epochs - 1:
        avg_loss = total_loss / len(dataset)
        w_sample = net.layers[0].neurons[0].weights[0]
        g1_sample = grads[0]["w"][0][0]
        gL_sample = grads[-1]["w"][0][0]
        print(f"Epoch {epoch}, Avg Loss: {avg_loss:.4f} | w00={w_sample:.4f}  g1={g1_sample:.4f}  gL={gL_sample:.4f}")

# ----------------------------
# Evaluation
# ----------------------------
correct = 0
pred_classes = []
for x, y_true in dataset:
    y_pred = net.forward(x)
    pred_class = y_pred.index(max(y_pred))
    pred_classes.append(pred_class)
    true_class = y_true.index(1)
    if pred_class == true_class:
        correct += 1

accuracy = correct / len(dataset)
print(f"\nTraining accuracy: {accuracy*100:.2f}%")

# --- Decision boundary ---
h = 0.02  # step size.
X_arr = np.array([x for x, _ in dataset])
y_arr = np.array([np.argmax(y) for _, y in dataset])

# Create a mesh grid covering the feature space
x_min, x_max = X_arr[:, 0].min() - 0.5, X_arr[:, 0].max() + 0.5
y_min, y_max = X_arr[:, 1].min() - 0.5, X_arr[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict class for each point in the grid
grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = []
for point in grid_points:
    out = net.forward(point)
    Z.append(np.argmax(out))
Z = np.array(Z).reshape(xx.shape)

# Plot
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.6)
plt.scatter(X_arr[:, 0], X_arr[:, 1], c=y_arr, cmap=plt.cm.Spectral, edgecolors="k")
plt.title("Decision Boundary on Circles (95% Net)")
plt.show()
