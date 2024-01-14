#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    int steps = m / batch;
    for (int step = 0; step < steps; step++) {
        int start_idx_X = step * batch * n;
        int start_idx_y = step * batch;

        float X_copy[batch * n];
        for (int i = 0; i < batch * n; i++) {
          X_copy[i] = X[start_idx_X + i];
        }
        unsigned char y_copy[batch];
        for (int i = 0; i < batch; i++) {
          y_copy[i] = y[start_idx_y + i];
        }
        // logits = np.dot(X_batch, theta) # [b_size, class]
        float logits[batch * k];
        for (int i = 0; i < batch; i++) {
            for (int k_idx = 0; k_idx < k; k_idx++) {
                float logit = 0;
                for (int j = 0; j < n; j++) {
                    logit += X_copy[i*n + j] * theta[j*k + k_idx];
                }
                logits[i*k + k_idx] = logit;                
            }
        }
        // normalizer = np.sum(np.exp(logits), axis=1)
        float exp[batch * k];
        for (int i = 0; i < batch; i++) {
            for (int k_idx = 0; k_idx < k; k_idx++) {
                exp[i*k+k_idx] = std::exp(logits[i*k+k_idx]);
            }
        }
        float normalizer[batch];
        for (int i = 0; i < batch; i++) {
            float sum_ = 0.0;
            for (int k_idx = 0; k_idx < k; k_idx++) {
                sum_ += exp[i*k+k_idx];
            }
            normalizer[i] = sum_;
        }
        // prob = np.exp(logits) / normalizer[:, None] # [b_size, class]
        float prob[batch * k];
        for (int i = 0; i < batch; i++) {
            for (int k_idx = 0; k_idx < k; k_idx++) {
                prob[i*k+k_idx] = exp[i*k+k_idx] / normalizer[i];
            }
        } 
        // one_hot_label = np.eye(classes)[y_batch] # [b_size, class]
        // prob_minus_label = prob - one_hot_label 
        // here we just modify the prob matrix in place
        for (int i = 0; i < batch; i++) {
            int label_idx = y_copy[i];
            prob[i*k + label_idx] -= 1.0;
        }
        // gradient = np.dot(X_batch.transpose(), prob - one_hot_label) 
        // gradient /= batch
        float gradient[n * k];
        for (int n_idx = 0; n_idx < n; n_idx++) {
            for (int k_idx = 0; k_idx < k; k_idx++) {
                float gradient_n_k = 0.0;
                for (int i = 0; i < batch; i ++) {
                    gradient_n_k += X_copy[i*n + n_idx] * prob[i*k + k_idx];
                }
                gradient_n_k = gradient_n_k / batch;
                gradient[n_idx * k + k_idx] = gradient_n_k;
                theta[n_idx * k + k_idx] -= lr * gradient_n_k;                
            }
        }

    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
