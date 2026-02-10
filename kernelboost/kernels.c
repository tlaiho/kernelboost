#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// C code for CPU kernel operations (float32 for GPU consistency)

// kernel type constants: 0 = gaussian, 1 = laplace
static inline float gaussian_weight(float sq_diff, float precision) {
    return expf(-precision * sq_diff);
}

static inline float laplace_weight(float sq_diff, float precision) {
    return expf(-precision * sqrtf(sq_diff));
}

void predict(
    float * predictions,
    float * training_dependent,
    float * training_features,
    float * prediction_features,
    float precision,
    int training_obs,
    int prediction_obs,
    int dimension,
    int kernel_type) {

    int i;

    #pragma omp parallel
    {
        int j, k;
        float weight_sum, dependent_sum, sq_diff, w;

        #pragma omp for
        for (i = 0; i < prediction_obs; i++) {
            weight_sum = 0;
            dependent_sum = 0;
            for (j = 0; j < training_obs; j++) {
                sq_diff = 0;
                for (k = 0; k < dimension; k++) {
                    float diff = prediction_features[i * dimension + k] - training_features[j * dimension + k];
                    sq_diff += diff * diff;
                }
                if (kernel_type == 0) w = gaussian_weight(sq_diff, precision);
                else w = laplace_weight(sq_diff, precision);
                weight_sum += w;
                dependent_sum += w * training_dependent[j];
            }
            // guard against division by zero: default to predicting 0.0
            predictions[i] = (weight_sum > 0) ? dependent_sum / weight_sum : 0.0f;
        }
    }
}

// index macro: for i <= j, maps to linear index in upper triangle storage
// i*n - i*(i-1)/2 + (j - i) = full_rows - subtract_lower_triangle + offset_current_row
#define TRI_IDX(i, j, n) ((i) * (n) - ((i) * ((i) - 1)) / 2 + ((j) - (i)))

float loo_mse(
    float * training_dependent,
    float * training_features,
    float precision,
    int training_obs,
    int dimension,
    int kernel_type,
    float mean_y) {
    // Leave-one-out loss using symmetry of the kernel weights.  
    // Weights computed with rescaling by 1 - self_weight.

    int n = training_obs;
    size_t tri_size = (size_t)n * (n + 1) / 2;

    // allocate upper triangle storage
    float *upper = (float *)malloc(tri_size * sizeof(float));
    if (!upper) return -1.0f;

    // first pass: compute upper triangle kernel values
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            float sq_diff = 0;
            for (int k = 0; k < dimension; k++) {
                float diff = training_features[i * dimension + k]
                           - training_features[j * dimension + k];
                sq_diff += diff * diff;
            }
            float w;
            if (kernel_type == 0) w = gaussian_weight(sq_diff, precision);
            else w = laplace_weight(sq_diff, precision);

            upper[TRI_IDX(i, j, n)] = w;
        }
    }

    // second pass: compute LOO errors using symmetry
    float cv_error = 0;
    #pragma omp parallel for reduction(+:cv_error)
    for (int i = 0; i < n; i++) {
        float weight_sum = 0;
        float dependent_sum = 0;

        for (int j = 0; j < n; j++) {
            // symmetry: K(i,j) stored at min(i,j), max(i,j)
            float w = (i <= j) ? upper[TRI_IDX(i, j, n)] // i is min, j is max
                               : upper[TRI_IDX(j, i, n)]; // j is min, i is max
            weight_sum += w;
            dependent_sum += w * training_dependent[j];
        }

        // self-weight = exp(0) = 1.0, normalized = 1/weight_sum
        float self_weight_norm = (weight_sum > 0) ? 1.0f / weight_sum : 1.0f;

        if (weight_sum > 0 && self_weight_norm <= 1.0f - 1e-4f) {
            // normal LOO calculation
            float prediction = dependent_sum / weight_sum;
            float scaled_error = (prediction - training_dependent[i])
                               / (1.0f - self_weight_norm);
            cv_error += scaled_error * scaled_error;
        } else {
            // no neighbors or isolated: use mean fallback penalty
            float residual = training_dependent[i] - mean_y;
            cv_error += residual * residual;
        }
    }

    free(upper);
    return cv_error / n;
}

void estimate_similarity(
    float * weight_sums,
    float * training_features,
    float * prediction_features,
    float precision,
    int training_obs,
    int prediction_obs,
    int dimension,
    int kernel_type) {

    int i;

    #pragma omp parallel
    {
        int j, k;
        float sq_diff, w, ws;

        #pragma omp for
        for (i = 0; i < prediction_obs; i++) {
            ws = 0;
            for (j = 0; j < training_obs; j++) {
                sq_diff = 0;
                for (k = 0; k < dimension; k++) {
                    float diff = prediction_features[i * dimension + k] - training_features[j * dimension + k];
                    sq_diff += diff * diff;
                }
                if (kernel_type == 0) w = gaussian_weight(sq_diff, precision);
                else w = laplace_weight(sq_diff, precision);
                ws += w;
            }
            weight_sums[i] = ws;
        }
    }
}

void predict_with_variance(
    float * predictions,
    float * variances,
    float * training_dependent,
    float * training_features,
    float * prediction_features,
    float precision,
    int training_obs,
    int prediction_obs,
    int dimension,
    int kernel_type) {

    int i;

    #pragma omp parallel
    {
        int j, k;
        float weight_sum, dependent_sum, dependent_sq_sum, sq_diff, w;

        #pragma omp for
        for (i = 0; i < prediction_obs; i++) {
            weight_sum = 0;
            dependent_sum = 0;
            dependent_sq_sum = 0;
            for (j = 0; j < training_obs; j++) {
                sq_diff = 0;
                for (k = 0; k < dimension; k++) {
                    float diff = prediction_features[i * dimension + k] - training_features[j * dimension + k];
                    sq_diff += diff * diff;
                }
                if (kernel_type == 0) w = gaussian_weight(sq_diff, precision);
                else w = laplace_weight(sq_diff, precision);
                weight_sum += w;
                dependent_sum += w * training_dependent[j];
                dependent_sq_sum += w * training_dependent[j] * training_dependent[j];
            }
            predictions[i] = (weight_sum > 0) ? dependent_sum / weight_sum : 0.0f;
            float expected_y_sq = (weight_sum > 0) ? dependent_sq_sum / weight_sum : 0.0f;
            // Variance: E[Y²|X] - E[Y|X]²
            variances[i] = expected_y_sq - predictions[i] * predictions[i];
            if (variances[i] < 0) variances[i] = 0;  // numerical stability
        }
    }
}

void get_weights(
    float * weights,
    float * training_features,
    float * prediction_features,
    float precision,
    int training_obs,
    int prediction_obs,
    int dimension,
    int kernel_type) {

    int i;

    #pragma omp parallel
    {
        int j, k;
        float sq_diff, w, weight_sum;

        #pragma omp for
        for (i = 0; i < prediction_obs; i++) {
            weight_sum = 0;
            // first pass: compute unnormalized weights
            for (j = 0; j < training_obs; j++) {
                sq_diff = 0;
                for (k = 0; k < dimension; k++) {
                    float diff = prediction_features[i * dimension + k] - training_features[j * dimension + k];
                    sq_diff += diff * diff;
                }
                if (kernel_type == 0) w = gaussian_weight(sq_diff, precision);
                else w = laplace_weight(sq_diff, precision);
                weights[i * training_obs + j] = w;
                weight_sum += w;
            }
            // second pass: normalize
            if (weight_sum > 0) {
                for (j = 0; j < training_obs; j++) {
                    weights[i * training_obs + j] /= weight_sum;
                }
            }
        }
    }
}
