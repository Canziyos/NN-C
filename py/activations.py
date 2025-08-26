import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def relu(z):
    return max(0, z)

def identity(z):
    return z

def tanh(z):
    return (math.exp(z) - math.exp(-z)) / (math.exp(z) + math.exp(-z))
