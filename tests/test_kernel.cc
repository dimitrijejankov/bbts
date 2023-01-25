#include <memory>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <immintrin.h>
#include <chrono>


void init_matrix(float *data, int32_t I, int32_t J) {
    for(size_t Idx = 0; Idx < I; ++Idx) {
        for(size_t Jdx = 0; Jdx < J; ++Jdx) {
            data[Idx * J + Jdx] = (float) (Idx * J + Jdx);
        }
    }
}

void zero_matrix(float *data, int32_t I, int32_t J) {
    for(size_t Idx = 0; Idx < I; ++Idx) {
        for(size_t Jdx = 0; Jdx < J; ++Jdx) {
            data[Idx * J + Jdx] = 0;
        }
    }
}

// a(I, K) b(K, J) c(I, J)
void wierd_mult(float *a, float *b, float *c, int I, int J, int K) {
    for (int i = 0; i < I; i++) {
        for (int j = 0; j < J; j++) {
            for (int k = 0; k < K; k++) {
                auto v = a[i * K + k] - b[j * K + k];
                c[i * J + j] += v * v;
            }
        }
    }
}

template<int kernel_rows, int kernel_cols>
void wierd_kernel(float *a, float *b, float *c, int lda, int ldb, int ldc, int K) {

    __m256 sums[kernel_rows][kernel_cols] = {};

    for (int k = 0; k < K; k += 8) {
        for (int j = 0; j < kernel_cols; j++) {
            __m256 b4 = _mm256_load_ps(b + ldb * j + k);
            for (int i = 0; i < kernel_rows; i++) {
                __m256 a4 = _mm256_load_ps(a + i * lda + k);
                auto v = _mm256_sub_ps(a4, b4);
                v = _mm256_mul_ps(v, v);
                sums[i][j] = _mm256_add_ps(v, sums[i][j]);
            }
        }
    }

    for (int i = 0; i < kernel_rows; i++) {
        for (int j = 0; j < kernel_cols; j++) {
            c[i * ldc + j] = sums[i][j][0] + sums[i][j][1] + sums[i][j][2] + sums[i][j][3] +
                             sums[i][j][4] + sums[i][j][5] + sums[i][j][6] + sums[i][j][7];
        }
    }
}


// a(I, K) b(K, J) c(I, J)
template<int kernel_rows, int kernel_cols>
void wierd_mult_with_kernel(float *a, float *b, float *c, int I, int J, int K) {

    // assert(I % kernel_rows == 0);
    // assert(J % kernel_cols == 0);

    // // avx registers are 8 floats wide
    // assert(kernel_cols % 8 == 0);

    for (int i = 0; i < I; i += kernel_rows) {

        // #pragma omp parallel for
        for (int j = 0; j < J; j += kernel_cols) {
            wierd_kernel<kernel_rows, kernel_cols>(&a[i * K], &b[j * K], &c[i * J + j], K, K, J, K);
        }
    }
}

int main() {

    // int I = 1024;
    // int K = 1024;
    // int J = 1024;
    // int I = 3008;
    // int K = 3008;
    // int J = 3008;
    // int I = 8;
    // int J = 8;
    // int K = 8;
    int I = 512;
    int J = 512;
    int K = 1024;

    auto a = (float*) std::aligned_alloc(256, I * K * sizeof(float));
    auto b = (float*) std::aligned_alloc(256, K * J * sizeof(float));
    auto c = (float*) std::aligned_alloc(256, I * J * sizeof(float));
    auto cc = (float*) std::aligned_alloc(256, I * J * sizeof(float));

    init_matrix(a, I, K);
    init_matrix(b, K, J);
    zero_matrix(c, I, J);

    auto start_basic = std::chrono::high_resolution_clock::now();
    wierd_mult(a, b, c, I, J, K);
    auto stop_basic = std::chrono::high_resolution_clock::now();

    zero_matrix(cc, I, J);
    auto start_kernel_1 = std::chrono::high_resolution_clock::now();
    wierd_mult_with_kernel<2, 2>(a, b, cc, I, J, K);
    auto stop_kernel_1 = std::chrono::high_resolution_clock::now();

    zero_matrix(cc, I, J);
    auto start_kernel_2 = std::chrono::high_resolution_clock::now();
    wierd_mult_with_kernel<2, 4>(a, b, cc, I, J, K);
    auto stop_kernel_2 = std::chrono::high_resolution_clock::now();

    zero_matrix(cc, I, J);
    auto start_kernel_3 = std::chrono::high_resolution_clock::now();
    wierd_mult_with_kernel<4, 2>(a, b, cc, I, J, K);
    auto stop_kernel_3 = std::chrono::high_resolution_clock::now();

    zero_matrix(cc, I, J);
    auto start_kernel_4 = std::chrono::high_resolution_clock::now();
    wierd_mult_with_kernel<8, 2>(a, b, cc, I, J, K);
    auto stop_kernel_4 = std::chrono::high_resolution_clock::now();

    zero_matrix(cc, I, J);
    auto start_kernel_5 = std::chrono::high_resolution_clock::now();
    wierd_mult_with_kernel<16, 2>(a, b, cc, I, J, K);
    auto stop_kernel_5 = std::chrono::high_resolution_clock::now();

    std::cout << "Time taken by function: " << std::chrono::duration_cast<std::chrono::nanoseconds>(stop_basic - start_basic).count() << " nanoseconds" << std::endl;
    std::cout << "Time taken by function: " << std::chrono::duration_cast<std::chrono::nanoseconds>(stop_kernel_1 - start_kernel_1).count() << " nanoseconds" << std::endl;
    std::cout << "Time taken by function: " << std::chrono::duration_cast<std::chrono::nanoseconds>(stop_kernel_2 - start_kernel_2).count() << " nanoseconds" << std::endl;
    std::cout << "Time taken by function: " << std::chrono::duration_cast<std::chrono::nanoseconds>(stop_kernel_3 - start_kernel_3).count() << " nanoseconds" << std::endl;
    std::cout << "Time taken by function: " << std::chrono::duration_cast<std::chrono::nanoseconds>(stop_kernel_4 - start_kernel_4).count() << " nanoseconds" << std::endl;
    std::cout << "Time taken by function: " << std::chrono::duration_cast<std::chrono::nanoseconds>(stop_kernel_5 - start_kernel_5).count() << " nanoseconds" << std::endl;

    float sum = 0.0f;
    for (int i = 0; i < I; i++) {
        for (int j = 0; j < J; j++) {
            sum += std::abs(c[i * J + j] - cc[i * J + j]);
            // std::cout << "org : " << c[i * J + j] << " kernel : " << cc[i * J + j] << '\n';
        }
    }

    std::cout << "Sum : " << sum / (I * J) << '\n';
    std::cout << "All good!.." << '\n';

    free(a);
    free(b);
    free(c);
    free(cc);

    return 0;
}