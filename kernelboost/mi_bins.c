#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// C code for fast MI estimation over the whole feature set

static int find_bin(float *thresholds, int n_thresh, float val) {
//  standard binary search
    int lo = 0, hi = n_thresh - 2;
    while (lo <= hi) {                                                                   
        int mid = (lo + hi) / 2;                                       
        if (val < thresholds[mid])                                                       
            hi = mid - 1;
        else
            lo = mid + 1;
    }
    return lo - 1; 
}

void histogram_mi_batch(
    float *X,              /* (n, n_features), row-major */
    float *residuals,      /* (n,) */
    int n,
    int n_features,
    float *x_thresholds,   /* (n_features, n_thresh) */
    float *y_thresholds,   /* (n_thresh,) */
    int n_thresh,                /* n_bins + 1 */
    float *out_mi /* (n_features,) output */ ) {
    
    int n_bins = n_thresh - 1;
    size_t binsize = (size_t) n_bins * n_bins;

    #pragma omp parallel for schedule(dynamic)
    for (int f=0; f < n_features; f++) {
        float *hist = calloc(binsize, sizeof(float));
            for (int i=0; i < n; i++) {
                int xi = find_bin(x_thresholds + f * n_thresh, n_thresh, X[i * n_features + f]);
                int yi = find_bin(y_thresholds, n_thresh, residuals[i]);
                hist[xi * n_bins + yi] += 1;
            }

        // convert to probabilities, compute marginals
        double inv_n = 1.0 / n;    
        float *pxy = calloc(binsize, sizeof(float)); 
        float *px = calloc(n_bins, sizeof(float));
        float *py = calloc(n_bins, sizeof(float));

        for (int x_index=0; x_index < n_bins; x_index++){
            for (int y_index=0; y_index < n_bins; y_index++){
                float probability = hist[x_index * n_bins + y_index] * inv_n;
                pxy[x_index * n_bins + y_index] = probability;
                px[x_index] += probability;
                py[y_index] += probability;
            }
        }

        // MI
        double mi = 0.0;
        for (int x_index=0; x_index < n_bins; x_index++){
            for (int y_index=0; y_index < n_bins; y_index++){
                if (pxy[x_index * n_bins + y_index] > 0 && px[x_index] * py[y_index] > 0)
                    mi += (pxy[x_index * n_bins + y_index] * 
                        log(pxy[x_index * n_bins + y_index] / (px[x_index] * py[y_index])));
            }
        }
        out_mi[f] = fmax(0, mi);
        free(hist); free(pxy); free(px); free(py);
    }
    }                
