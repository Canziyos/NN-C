#ifndef MATH_UTILS_H
#define MATH_UTILS_H

void matrix_multiply(double **weights, double *input, double *output, int rows, int cols);

// Sigmoid func (activation function).
double sigmoid(double z);

#endif
