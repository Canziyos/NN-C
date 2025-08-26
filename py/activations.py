import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1 - s)


def relu(z):
    return max(0, z)

def relu_prime(z):
    return 1 if z > 0 else 0


def identity(z):
    return z

def identity_prime(z):
    return 1


def tanh(z):
    return (math.exp(z) - math.exp(-z)) / (math.exp(z) + math.exp(-z))

def tanh_prime(z):
    t = tanh(z)
    return 1 - t**2
