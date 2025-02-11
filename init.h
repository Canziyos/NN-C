#ifndef INIT_H
#define INIT_H

#include "Model.h"

// Network initialization
void initialize_network(NeuralNetwork *nn, int num_layers, int *neurons_per_layer);
void free_network(NeuralNetwork *nn);


#endif
