#include "uwnet.h"
#include <assert.h>
#include <math.h>

// Run an activation function on each element in a matrix,
// modifies the matrix in place
// matrix m: Input to activation function
// ACTIVATION a: function to run
void activate_matrix(matrix m, ACTIVATION a)
{
    int i, j;
    for (i = 0; i < m.rows; ++i) {
        double sum = 0;
        for (j = 0; j < m.cols; ++j) {
            double x = m.data[i * m.cols + j];
            if (a == LOGISTIC) {
                x = 1 / (1 + exp(-x));
            } else if (a == RELU) {
                if (x <= 0) {
                    x = 0;
                }
            } else if (a == LRELU) {
                if (x <= 0) {
                    x = 0.1 * x;
                }
            } else if (a == SOFTMAX) {
                x = exp(x);
            }
            m.data[i * m.cols + j] = x;
            sum += m.data[i * m.cols + j];
        }
        if (a == SOFTMAX) {
            for (j = 0; j < m.cols; ++j) {
                m.data[i * m.cols + j] = m.data[i * m.cols + j] / sum;
            }
        }
    }
}

// Calculates the gradient of an activation function and multiplies it into
// the delta for a layer
// matrix m: an activated layer output
// ACTIVATION a: activation function for a layer
// matrix d: delta before activation gradient
void gradient_matrix(matrix m, ACTIVATION a, matrix d)
{
    int i, j;
    for (i = 0; i < m.rows; ++i) {
        for (j = 0; j < m.cols; ++j) {
            double x = m.data[i * m.cols + j];
            if (a == LOGISTIC) {
                x = x * (1 - x);
            }
            if (a == RELU) {
                if (x <= 0) {
                    x = 0;
                } else {
                    x = 1;
                }
            }
            if (a == LRELU) {
                if (x <= 0) {
                    x = 0.1;
                } else {
                    x = 1;
                }
            }
            if (a == SOFTMAX) {
                x = 1;
            }

            d.data[i * m.cols + j] *= x;
            /* m.data[i * m.cols + j] = x; */
        }
    }
}
