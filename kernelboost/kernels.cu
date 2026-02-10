/*
 * CUDA kernels for GPU kernel operations
 */

extern "C" __global__ void gaussian_kernel(
    float *p_features,
    float *t_features,
    float *out_matrix,
    int p_count,
    int t_count,
    int dimension,
    float precision
)
{
    for (int outer_index = blockIdx.x * blockDim.x + threadIdx.x;
         outer_index < p_count; outer_index += blockDim.x * gridDim.x) {
        for (int inner_index = blockIdx.y * blockDim.y + threadIdx.y;
             inner_index < t_count; inner_index += blockDim.y * gridDim.y) {
            float s = 0;
            for (int k = 0; k < dimension; k++) {
                float diff = p_features[outer_index * dimension + k] - t_features[inner_index * dimension + k];
                s += diff * diff;
            }
            out_matrix[inner_index + outer_index * t_count] = __expf(-precision * s);
        }
    }
}

extern "C" __global__ void laplace_kernel(
    float *p_features,
    float *t_features,
    float *out_matrix,
    int p_count,
    int t_count,
    int dimension,
    float precision
)
{
    for (int outer_index = blockIdx.x * blockDim.x + threadIdx.x;
         outer_index < p_count; outer_index += blockDim.x * gridDim.x) {
        for (int inner_index = blockIdx.y * blockDim.y + threadIdx.y;
             inner_index < t_count; inner_index += blockDim.y * gridDim.y) {
            float s = 0;
            for (int k = 0; k < dimension; k++) {
                float diff = p_features[outer_index * dimension + k] - t_features[inner_index * dimension + k];
                s += diff * diff;
            }
            out_matrix[inner_index + outer_index * t_count] = __expf(-precision * sqrt(s));
        }
    }
}

extern "C" __global__ void weights(
    float *k_matrix,
    float *sums,
    int p_count,
    int t_count
)
{
    for (int outer_index = blockIdx.x * blockDim.x + threadIdx.x;
         outer_index < p_count; outer_index += blockDim.x * gridDim.x) {
        for (int inner_index = blockIdx.y * blockDim.y + threadIdx.y;
             inner_index < t_count; inner_index += blockDim.y * gridDim.y) {
            int idx = inner_index + outer_index * t_count;
            // guard against division by zero: default to 0.0 
            k_matrix[idx] = (sums[outer_index] > 0) ? k_matrix[idx] / sums[outer_index] : 0.0f;
        }
    }
}

extern "C" __global__ void loo_error(
    float *predictions,
    float *t_dependent,
    float *k_matrix,
    float *out_matrix,
    int t_count
)
{
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < t_count; index += blockDim.x * gridDim.x) {
        float diff = (t_dependent[index] - predictions[index]) / (1 - k_matrix[index * t_count + index]);
        out_matrix[index] = diff * diff;
    }
}
