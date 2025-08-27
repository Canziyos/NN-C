momentum_state = {}

def reset_momentum():
    momentum_state.clear()

# optim.py
momentum_state = {}

def update(net, grads, lr, momentum=0.9, clip=1.0, weight_decay=0.0):
    global momentum_state
    for l, layer in enumerate(net.layers):
        for j, neuron in enumerate(layer.neurons):
            key_w = (l, j, "w"); key_b = (l, j, "b")
            if key_w not in momentum_state:
                momentum_state[key_w] = [0.0 for _ in neuron.weights]
            if key_b not in momentum_state:
                momentum_state[key_b] = 0.0

            # Weights
            for i in range(len(neuron.weights)):
                g = grads[l]["w"][j][i]
                if clip is not None:
                    g = max(min(g, clip), -clip)
                v = momentum_state[key_w][i]
                v = momentum * v - lr * g
                # decoupled weight decay (AdamW style)
                neuron.weights[i] = neuron.weights[i] * (1 - lr * weight_decay) + v
                momentum_state[key_w][i] = v

            # Bias (no weight decay on bias)
            gb = grads[l]["b"][j]
            if clip is not None:
                gb = max(min(gb, clip), -clip)
            v_b = momentum_state[key_b]
            v_b = momentum * v_b - lr * gb
            neuron.bias += v_b
            momentum_state[key_b] = v_b
