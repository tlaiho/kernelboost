#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// C code for fast MI estimation over the whole feature set

static int find_bin(float *thresholds, int n_thresh, float val) {
    int lo = 0, hi = n_thresh - 2;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (val < thresholds[mid])
            hi = mid - 1;
        else
            lo = mid + 1;
    }
    int bin = lo - 1;
    if (bin < 0) bin = 0;
    if (bin > n_thresh - 2) bin = n_thresh - 2;
    return bin;
}

void histogram_mi_batch(
    float *X,              /* (n, n_features), row-major */
    float *residuals,      /* (n,) */
    int n,
    int n_features,
    float *x_thresholds,   /* (n_features, n_thresh) */
    float *y_thresholds,   /* (n_thresh,) */
    int n_thresh,          /* n_bins + 1 */
    float *out_mi          /* (n_features,) */ ) {
    
    int n_bins = n_thresh - 1;
    size_t binsize = (size_t) n_bins * n_bins;

    #pragma omp parallel for schedule(dynamic)
    for (int f=0; f < n_features; f++) {
        float *hist = calloc(binsize, sizeof(float));
            for (int i=0; i < n; i++) {
                int xi = find_bin(x_thresholds + f * n_thresh, n_thresh, X[i * n_features + f]);
                int yi = find_bin(y_thresholds, n_thresh, residuals[i]);
                hist[xi * n_bins + yi]++;
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

void histogram_mi_3d_batch(
    float *z,              /* (n,) fixed first variable */
    float *X,              /* (n, n_candidates), row-major */
    float *y,              /* (n,) target */
    int n,
    int n_candidates,
    float *z_thresholds,   /* (n_thresh,) */
    float *x_thresholds,   /* (n_candidates, n_thresh) */
    float *y_thresholds,   /* (n_thresh,) */
    int n_thresh,          /* n_bins + 1 */
    float *out_mi          /* (n_candidates,) */ ) {

    int n_bins = n_thresh - 1;
    size_t hist_size = (size_t) n_bins * n_bins * n_bins;
    size_t zx_size = (size_t) n_bins * n_bins;

    #pragma omp parallel for schedule(dynamic)
    for (int f=0; f < n_candidates; f++) {
        float *hist = calloc(hist_size, sizeof(float));
            for (int i=0; i < n; i++) {
                int zi = find_bin(z_thresholds, n_thresh, z[i]);
                int xi = find_bin(x_thresholds + f * n_thresh, n_thresh, X[i * n_candidates + f]);
                int yi = find_bin(y_thresholds, n_thresh, y[i]);
                hist[zi * n_bins * n_bins + xi * n_bins + yi]++;
            }

        // convert to probabilities, compute marginals
        double inv_n = 1.0 / n;
        float *pzxy = calloc(hist_size, sizeof(float));
        float *pzx = calloc(zx_size, sizeof(float));
        float *py = calloc(n_bins, sizeof(float));
        
        for (int z_index=0; z_index < n_bins; z_index++){
            for (int x_index=0; x_index < n_bins; x_index++){
                for (int y_index=0; y_index < n_bins; y_index++){
                    float probability = hist[z_index * n_bins * n_bins + x_index * n_bins + y_index] * inv_n;
                    pzxy[z_index * n_bins * n_bins + x_index * n_bins + y_index] = probability;
                    pzx[z_index * n_bins + x_index] += probability;
                    py[y_index] += probability;
                }
            }
        }

        // MI
        double mi = 0.0;
        for (int z_index=0; z_index < n_bins; z_index++){
            for (int x_index=0; x_index < n_bins; x_index++){
                for (int y_index=0; y_index < n_bins; y_index++){
                    if (pzxy[z_index * n_bins * n_bins + x_index * n_bins + y_index] > 0 
                        && pzx[z_index * n_bins + x_index] * py[y_index] > 0)
                        mi += (pzxy[z_index * n_bins * n_bins + x_index * n_bins + y_index] *
                            log(pzxy[z_index * n_bins * n_bins + x_index * n_bins + y_index] 
                                / (pzx[z_index * n_bins + x_index] * py[y_index])));
                }
            }
        }
        out_mi[f] = fmax(0, mi);
        free(hist); free(pzxy); free(pzx); free(py);
    }
}

/* ---- Grassberger-corrected estimator (Grassberger 2003) ----
   H_G = log(n) - (1/n) * sum_i n_i * G(n_i),  
   G(k) = psi(k) + (-1)^k / (k(k+1)), where
   psi(k) is the digamma function: integral over
   exp(-t)/t - exp(-kt)/(1- e(-t)) from 0 to infinity. 

/* G table filled via the recurrence psi(k+1) = psi(k) + 1/k
   from psi(1) = -gamma (Euler-Mascheroni). */
static double *build_g_table(int n) {
    double *G = malloc(((size_t) n + 1) * sizeof(double));
    double psi = -0.57721566490153286;
    for (int k=1; k <= n; k++) {
        double correction = 1.0 / ((double) k * (k + 1));
        G[k] = psi + ((k & 1) ? -correction : correction);
        psi += 1.0 / k;
    }
    return G;
}

static double entropy_gr(const int *counts, size_t m, int n, const double *G) {
    double sum = 0.0;
    for (size_t i=0; i < m; i++)
        if (counts[i] > 0) sum += counts[i] * G[counts[i]];
    return log((double) n) - sum / n;
}

void histogram_mi_batch_gr(
    float *X,              /* (n, n_features), row-major */
    float *residuals,      /* (n,) */
    int n,
    int n_features,
    float *x_thresholds,   /* (n_features, n_thresh) */
    float *y_thresholds,   /* (n_thresh,) */
    int n_thresh,          /* n_bins + 1 */
    float *out_mi          /* (n_features,) output */ ) {

    int n_bins = n_thresh - 1;
    size_t binsize = (size_t) n_bins * n_bins;

    double *G = build_g_table(n);

    // y is candidate-independent: bin once, calculate H_G(Y) once. 
    int *ybin = malloc(n * sizeof(int));
    int *cy = calloc(n_bins, sizeof(int));  // y counts
    for (int i=0; i < n; i++) {
        ybin[i] = find_bin(y_thresholds, n_thresh, residuals[i]);
        cy[ybin[i]]++;
    }
    double hy = entropy_gr(cy, n_bins, n, G);

    #pragma omp parallel for schedule(dynamic)
    for (int f=0; f < n_features; f++) {
        int *cxy = calloc(binsize, sizeof(int));
        int *cx = calloc(n_bins, sizeof(int));
        for (int i=0; i < n; i++) {
            int xi = find_bin(x_thresholds + f * n_thresh, n_thresh, X[i * n_features + f]);
            cx[xi]++;
            cxy[xi * n_bins + ybin[i]]++;
        }

        /* I(X;Y) = H(X) + H(Y) - H(XY) */
        double mi = entropy_gr(cx, n_bins, n, G) + hy
                  - entropy_gr(cxy, binsize, n, G);
        out_mi[f] = fmax(0, mi);
        free(cxy); free(cx);
    }
    free(G); free(ybin); free(cy);
}

void histogram_mi_3d_batch_gr(
    float *z,              /* (n,) fixed first variable */
    float *X,              /* (n, n_candidates), row-major */
    float *y,              /* (n,) target */
    int n,
    int n_candidates,
    float *z_thresholds,   /* (n_thresh,) */
    float *x_thresholds,   /* (n_candidates, n_thresh) */
    float *y_thresholds,   /* (n_thresh,) */
    int n_thresh,          /* n_bins + 1 */
    float *out_mi          /* (n_candidates,)  */ ) {

    int n_bins = n_thresh - 1;
    size_t zx_size = (size_t) n_bins * n_bins;
    size_t hist_size = zx_size * n_bins;

    double *G = build_g_table(n);

    // z and y are candidate-independent: bin both once, H_G(Y) once. 
    int *zbin = malloc(n * sizeof(int));
    int *ybin = malloc(n * sizeof(int));
    int *cy = calloc(n_bins, sizeof(int));
    for (int i=0; i < n; i++) {
        zbin[i] = find_bin(z_thresholds, n_thresh, z[i]);
        ybin[i] = find_bin(y_thresholds, n_thresh, y[i]);
        cy[ybin[i]]++;
    }
    double hy = entropy_gr(cy, n_bins, n, G);

    #pragma omp parallel for schedule(dynamic)
    for (int f=0; f < n_candidates; f++) {
        int *czxy = calloc(hist_size, sizeof(int));
        int *czx = calloc(zx_size, sizeof(int));
        for (int i=0; i < n; i++) {
            int xi = find_bin(x_thresholds + f * n_thresh, n_thresh, X[i * n_candidates + f]);
            czx[zbin[i] * n_bins + xi]++;
            czxy[(size_t) zbin[i] * zx_size + (size_t) xi * n_bins + ybin[i]]++;
        }

        // I((Z,X);Y) = H(ZX) + H(Y) - H(ZXY)
        double mi = entropy_gr(czx, zx_size, n, G) + hy
                  - entropy_gr(czxy, hist_size, n, G);
        out_mi[f] = fmax(0, mi);
        free(czxy); free(czx);
    }
    free(G); free(zbin); free(ybin); free(cy);
}
