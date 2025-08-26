from dense_nn import DenseNN
from dense_layer import DenseLayer
from utils import MSE

dataset = [
   ( [0,0], [0] ),  
   ( [0,1], [1] ),  
   ( [1,0], [1] ),  
   ( [1,1], [0] )  
]
num_epochs = 10

# 1. Build the model (n_neurons = outputs)
net = DenseNN([
    DenseLayer(input_s=2, n_neurons=2),
    DenseLayer(input_s=2, n_neurons=2),
    DenseLayer(input_s=2, n_neurons=1)
])


# 2. Training loop.
for epoch in range(num_epochs):
    total_loss = 0
    for x, y_true in dataset:
        y_pred = net.forward(x)
        loss = MSE(y_pred, y_true) # loss
      
        # TODO: add backward() implementation for output layer
        # TODO: add update() function for SGD.
        #grads = backward(loss)     # Backpropagation (gradients)
        #update(net, grads, lr)     # Gradient descent step
        total_loss += loss
    print(f"Epoch {epoch}, Loss: {total_loss}")

# 3. Testing
for x in dataset:
    print(x, "->", net.forward(x))
