
def MSE(y_true, y_pred):
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("MSE: y_true and y_pred must be non-empty and equal length.")
    n = len(y_true)
    return sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / n

# I know how wrong I was.
def output_layer_delta(y_true, y_pred, z_values, activation_prime):
    n = len(y_true)
    return [(2/n)*(yp - yt) * activation_prime(z)
            for yp, yt, z in zip(y_pred, y_true, z_values)]


def output_gradient(delta, prev_activations):
    # For each neuron j in the output layer:
    weight_grads = [[delta[j] * a for a in prev_activations]
                    for j in range(len(delta))]
    bias_grads = delta[:]  # just copy.
    return weight_grads, bias_grads

# How wrong I was depends on how much I influenced
# the next layer’s mistakes, scaled by my own sensitivity.
def hidden_layer_delta(z_values, activation_prime, next_layer_weights, next_layer_delta):
    deltas = []
    for i, z in enumerate(z_values):
        # sum over next-layer neurons j: w_ij * delta_j.
        downstream = sum(next_layer_weights[j][i] * next_layer_delta[j] 
                         for j in range(len(next_layer_delta)))
        deltas.append(activation_prime(z) * downstream)
    return deltas

