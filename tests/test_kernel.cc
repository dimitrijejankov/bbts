#include <memory>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <immintrin.h>
#include <chrono>


void init_matrix(float *data, int32_t I, int32_t J) {
    for(size_t Idx = 0; Idx < I; ++Idx) {
        for(size_t Jdx = 0; Jdx < J; ++Jdx) {
            data[Idx * J + Jdx] = (float) 0.001f * (Idx * J + Jdx);
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
                auto v = a[i * K + k] - b[k * J + j];
                c[i * J + j] += v * v;
            }
        }
    }
}

template<int kernel_rows, int kernel_cols>
void wierd_kernel(float *a, float *b, float *c, int lda, int ldb, int ldc, int K) {

    // we accumulate stuff here
    __m256 sums[4][kernel_cols / 8] = {};

    for (int k = 0; k < K; k++) {
        for (int j = 0; j < kernel_cols / 8; j++) {

            __m256 b4 = _mm256_load_ps(b + ldb * k + 8 * j);
            for (int i = 0; i < kernel_rows; i++) {
                
                __m256 a4 = _mm256_broadcast_ss(a + i * lda + k);
                auto v = _mm256_sub_ps(a4, b4);
                v = _mm256_mul_ps(v, v);
                sums[i][j] = _mm256_add_ps(v, sums[i][j]);
            }
        }
    }

    for (int i = 0; i < kernel_rows; i++) {
        for (int j = 0; j < kernel_cols / 8; j++) {
            _mm256_store_ps(&c[i * ldc + j * 8], sums[i][j]);
        }
    }
}

// a(I, K) b(K, J) c(I, J)
template<int kernel_rows, int kernel_cols>
void wierd_mult_with_kernel(float *a, float *b, float *c, int I, int J, int K) {

    assert(I % kernel_rows == 0);
    assert(J % kernel_cols == 0);

    // avx registers are 8 floats wide
    assert(kernel_cols % 8 == 0);

    for (int i = 0; i < I; i += kernel_rows) {

        #pragma omp parallel for
        for (int j = 0; j < J; j += kernel_cols) {
            wierd_kernel<kernel_rows, kernel_cols>(&a[i * K], &b[j], &c[i * J + j], K, J, J, K);
        }
    }
}

// int main() {

//     // int I = 1024;
//     // int K = 1024;
//     // int J = 1024;
//     int I = 3008;
//     int K = 3008;
//     int J = 3008;
//     // int I = 128;
//     // int J = 128;
//     // int K = 128;

//     auto a = (float*) std::aligned_alloc(256, I * K * sizeof(float));
//     auto b = (float*) std::aligned_alloc(256, K * J * sizeof(float));
//     auto c = (float*) std::aligned_alloc(256, I * J * sizeof(float));
//     auto cc = (float*) std::aligned_alloc(256, I * J * sizeof(float));

//     init_matrix(a, I, K);
//     init_matrix(b, K, J);
//     zero_matrix(c, I, J);
//     zero_matrix(cc, I, J);

//     auto start_basic = std::chrono::high_resolution_clock::now();
//     wierd_mult(a, b, c, I, J, K);
//     auto stop_basic = std::chrono::high_resolution_clock::now();


//     auto start_kernel = std::chrono::high_resolution_clock::now();
//     wierd_mult_with_kernel<4, 16>(a, b, cc, I, J, K);
//     auto stop_kernel = std::chrono::high_resolution_clock::now();

//     std::cout << "Time taken by function: " << std::chrono::duration_cast<std::chrono::microseconds>(stop_basic - start_basic).count() / 1000000.0 << " seconds" << std::endl;
//     std::cout << "Time taken by function: " << std::chrono::duration_cast<std::chrono::microseconds>(stop_kernel - start_kernel).count() / 1000000.0 << " seconds" << std::endl;


//     for (int i = 0; i < I; i++) {
//         for (int j = 0; j < J; j++) {
//             assert(std::abs(c[i * J + j] - cc[i * J + j]) < 0.001f);
//         }
//     }

//     return 0;
// }


int main() {

    // int I = 1024;
    // int K = 1024;
    // int J = 1024;
    size_t I = 262144 / 2;
    size_t K = 1024 / 2;
    size_t J = 262144 / 2;
    // int I = 128;
    // int J = 128;
    // int K = 128;

    auto a = (float*) std::aligned_alloc(256, I * K * sizeof(float));
    auto b = (float*) std::aligned_alloc(256, K * J * sizeof(float));
    auto c = (float*) std::aligned_alloc(256, I * J * sizeof(float));
    auto cc = (float*) std::aligned_alloc(256, I * J * sizeof(float));

    init_matrix(a, I, K);
    init_matrix(b, K, J);
    zero_matrix(c, I, J);
    zero_matrix(cc, I, J);


    auto start_kernel = std::chrono::high_resolution_clock::now();
    wierd_mult_with_kernel<4, 16>(a, b, cc, I, J, K);
    auto stop_kernel = std::chrono::high_resolution_clock::now();

    std::cout << "Time taken by function: " << std::chrono::duration_cast<std::chrono::microseconds>(stop_kernel - start_kernel).count() / 1000000.0 << " seconds" << std::endl;

    free(a);
    free(b);
    free(c);
    free(cc);

    return 0;
}