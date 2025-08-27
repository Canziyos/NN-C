def update(net, grads, lr):
    for l, layer in enumerate(net.layers):
        for j, neuron in enumerate(layer.neurons):
            # Update each weight
            for i in range(len(neuron.weights)):
                neuron.weights[i] -= lr * grads[l]["w"][j][i]
            # Update bias
            neuron.bias -= lr * grads[l]["b"][j]
