#include <math.h>
#include "math_utils.h"

void matrix_multiply(double **weights, double *input, double *output, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        output[i] = 0.0;  // Initialize output (z)
        for (int j = 0; j < cols; j++) {
            output[i] += weights[j][i] * input[j];
        }
    }
}



double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}
