#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <limits>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#ifdef _MSC_VER
#define __builtin_popcount __popcnt 
#define __builtin_popcountl __popcnt
#endif

#define DEFINE_CAFFE_CPU_UNARY_FUNC(name, operation) \
  template<typename Dtype> \
  DLL_EXPORT void caffe_cpu_##name(const int n, const Dtype* x, Dtype* y) { \
    CHECK_GT(n, 0); CHECK(x); CHECK(y); \
    for (int i = 0; i < n; ++i) { \
      operation; \
    } \
  } \
  template DLL_EXPORT void caffe_cpu_##name<float>(const int n, const float* x, float* y); \
  template DLL_EXPORT void caffe_cpu_##name<double>(const int n, const double* x, double* y);

namespace caffe {
    DEFINE_CAFFE_CPU_UNARY_FUNC(sign, y[i] = caffe_sign<Dtype>(x[i]));

    DEFINE_CAFFE_CPU_UNARY_FUNC(sgnbit, \
    y[i] = static_cast<bool>((std::signbit)(x[i])));

    DEFINE_CAFFE_CPU_UNARY_FUNC(fabs, y[i] = std::fabs(x[i]));


    inline void caffe_gpu_memset(const size_t N, const int alpha, void* X) {
#ifndef CPU_ONLY
  CUDA_CHECK(cudaMemset(X, alpha, N));  // NOLINT(caffe/alt_fn)
#else
  NO_GPU;
#endif
}
template<>
void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template<>
void caffe_cpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template <>
void caffe_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_cpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) { cblas_saxpy(N, alpha, X, 1, Y, 1); }

template <>
void caffe_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) { cblas_daxpy(N, alpha, X, 1, Y, 1); }

template <typename Dtype>
void caffe_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    return;
  }
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

template DLL_EXPORT void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C);

template DLL_EXPORT void caffe_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M, const int N,
    const float alpha, const float* A, const float* x, const float beta,
    float* y);

template DLL_EXPORT void caffe_axpy<float>(const int N, const float alpha, const float* X,
    float* Y);

template DLL_EXPORT void caffe_cpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C);

template DLL_EXPORT void caffe_cpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M, const int N,
    const double alpha, const double* A, const double* x, const double beta,
    double* y);

template DLL_EXPORT void caffe_axpy<double>(const int N, const double alpha, const double* X,
    double* Y);



template DLL_EXPORT void caffe_set<int>(const int N, const int alpha, int* Y);
template DLL_EXPORT void caffe_set<float>(const int N, const float alpha, float* Y);
template DLL_EXPORT void caffe_set<double>(const int N, const double alpha, double* Y);

template <>
void caffe_add_scalar(const int N, const float alpha, float* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <>
void caffe_add_scalar(const int N, const double alpha, double* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <typename Dtype>
void caffe_copy(const int N, const Dtype* X, Dtype* Y) {
  if (X != Y) {
    if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
      // NOLINT_NEXT_LINE(caffe/alt_fn)
      CUDA_CHECK(cudaMemcpy(Y, X, sizeof(Dtype) * N, cudaMemcpyDefault));
#else
      NO_GPU;
#endif
    } else {
      memcpy(Y, X, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    }
  }
}

template DLL_EXPORT void caffe_copy<int>(const int N, const int* X, int* Y);
template DLL_EXPORT void caffe_copy<unsigned int>(const int N, const unsigned int* X,
    unsigned int* Y);
template DLL_EXPORT void caffe_copy<float>(const int N, const float* X, float* Y);
template DLL_EXPORT void caffe_copy<double>(const int N, const double* X, double* Y);

template <>
void caffe_scal<float>(const int N, const float alpha, float *X) {
  cblas_sscal(N, alpha, X, 1);
}

template <>
void caffe_scal<double>(const int N, const double alpha, double *X) {
  cblas_dscal(N, alpha, X, 1);
}

template <>
void caffe_cpu_axpby<float>(const int N, const float alpha, const float* X,
                            const float beta, float* Y) {
  cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void caffe_cpu_axpby<double>(const int N, const double alpha, const double* X,
                             const double beta, double* Y) {
  cblas_daxpby(N, alpha, X, 1, beta, Y, 1);
}
template DLL_EXPORT void caffe_cpu_axpby<float>(const int N, const float alpha, const float* X,
    const float beta, float* Y);
template DLL_EXPORT void caffe_cpu_axpby<double>(const int N, const double alpha, const double* X,
    const double beta, double* Y);
template <>
void caffe_add<float>(const int n, const float* a, const float* b,
    float* y) {
  vAdd(n, a, b, y);
}

template <>
void caffe_add<double>(const int n, const double* a, const double* b,
    double* y) {
  vAdd(n, a, b, y);
}

template <>
void caffe_sub<float>(const int n, const float* a, const float* b,
    float* y) {
  vSub(n, a, b, y);
}

template <>
void caffe_sub<double>(const int n, const double* a, const double* b,
    double* y) {
  vSub(n, a, b, y);
}

template <>
void caffe_mul<float>(const int n, const float* a, const float* b,
    float* y) {
  vMul(n, a, b, y);
}

template <>
void caffe_mul<double>(const int n, const double* a, const double* b,
    double* y) {
  vMul(n, a, b, y);
}

template <>
void caffe_div<float>(const int n, const float* a, const float* b,
    float* y) {
  vDiv(n, a, b, y);
}

template <>
void caffe_div<double>(const int n, const double* a, const double* b,
    double* y) {
  vDiv(n, a, b, y);
}

template <>
void caffe_powx<float>(const int n, const float* a, const float b,
    float* y) {
  vPowx(n, a, b, y);
}

template <>
void caffe_powx<double>(const int n, const double* a, const double b,
    double* y) {
  vPowx(n, a, b, y);
}

template <>
void caffe_sqr<float>(const int n, const float* a, float* y) {
  vSqr(n, a, y);
}

template <>
void caffe_sqr<double>(const int n, const double* a, double* y) {
  vSqr(n, a, y);
}

template <>
void caffe_exp<float>(const int n, const float* a, float* y) {
  vExp(n, a, y);
}

template <>
void caffe_exp<double>(const int n, const double* a, double* y) {
  vExp(n, a, y);
}

template <>
void caffe_log<float>(const int n, const float* a, float* y) {
  vLn(n, a, y);
}

template <>
void caffe_log<double>(const int n, const double* a, double* y) {
  vLn(n, a, y);
}

template <>
void caffe_abs<float>(const int n, const float* a, float* y) {
    vAbs(n, a, y);
}

template <>
void caffe_abs<double>(const int n, const double* a, double* y) {
    vAbs(n, a, y);
}

unsigned int caffe_rng_rand() {
  return (*caffe_rng())();
}

template DLL_EXPORT void caffe_add_scalar<float>(const int N, const float alpha, float *X);
template DLL_EXPORT void caffe_scal<float>(const int N, const float alpha, float *X);
template DLL_EXPORT void caffe_sqr<float>(const int N, const float* a, float* y);
template DLL_EXPORT void caffe_add<float>(const int N, const float* a, const float* b, float* y);
template DLL_EXPORT void caffe_sub<float>(const int N, const float* a, const float* b, float* y);
template DLL_EXPORT void caffe_mul<float>(const int N, const float* a, const float* b, float* y);
template DLL_EXPORT void caffe_div<float>(const int N, const float* a, const float* b, float* y);
template DLL_EXPORT void caffe_powx<float>(const int n, const float* a, const float b, float* y);

template DLL_EXPORT void caffe_add_scalar<double>(const int N, const double alpha, double *X);
template DLL_EXPORT void caffe_scal<double>(const int N, const double alpha, double *X);
template DLL_EXPORT void caffe_sqr<double>(const int N, const double* a, double* y);
template DLL_EXPORT void caffe_add<double>(const int N, const double* a, const double* b, double* y);
template DLL_EXPORT void caffe_sub<double>(const int N, const double* a, const double* b, double* y);
template DLL_EXPORT void caffe_mul<double>(const int N, const double* a, const double* b, double* y);
template DLL_EXPORT void caffe_div<double>(const int N, const double* a, const double* b, double* y);
template DLL_EXPORT void caffe_powx<double>(const int n, const double* a, const double b, double* y);


template <typename Dtype>
Dtype caffe_nextafter(const Dtype b) {
  return boost::math::nextafter<Dtype>(
      b, std::numeric_limits<Dtype>::max());
}

template DLL_EXPORT float caffe_nextafter(const float b);

template DLL_EXPORT double caffe_nextafter(const double b);

template <typename Dtype>
void caffe_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_LE(a, b);
  boost::uniform_real<Dtype> random_distribution(a, caffe_nextafter<Dtype>(b));
  boost::variate_generator<caffe::rng_t*, boost::uniform_real<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template DLL_EXPORT void caffe_rng_uniform<float>(const int n, const float a, const float b, float* r);

template DLL_EXPORT void caffe_rng_uniform<double>(const int n, const double a, const double b, double* r);

template <typename Dtype>
void caffe_rng_gaussian(const int n, const Dtype a,
                        const Dtype sigma, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GT(sigma, 0);
  boost::normal_distribution<Dtype> random_distribution(a, sigma);
  boost::variate_generator<caffe::rng_t*, boost::normal_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template DLL_EXPORT void caffe_rng_gaussian<float>(const int n, const float mu,
                               const float sigma, float* r);

template DLL_EXPORT void caffe_rng_gaussian<double>(const int n, const double mu,
                                const double sigma, double* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template DLL_EXPORT void caffe_rng_bernoulli<double>(const int n, const double p, int* r);

template DLL_EXPORT void caffe_rng_bernoulli<float>(const int n, const float p, int* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, unsigned int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = static_cast<unsigned int>(variate_generator());
  }
}

template DLL_EXPORT void caffe_rng_bernoulli<double>(const int n, const double p, unsigned int* r);

template DLL_EXPORT void caffe_rng_bernoulli<float>(const int n, const float p, unsigned int* r);

template <>
float caffe_cpu_strided_dot<float>(const int n, const float* x, const int incx,
    const float* y, const int incy) {
  return cblas_sdot(n, x, incx, y, incy);
}

template <>
double caffe_cpu_strided_dot<double>(const int n, const double* x,
    const int incx, const double* y, const int incy) {
  return cblas_ddot(n, x, incx, y, incy);
}

template <typename Dtype>
Dtype caffe_cpu_dot(const int n, const Dtype* x, const Dtype* y) {
  return caffe_cpu_strided_dot(n, x, 1, y, 1);
}

template DLL_EXPORT float caffe_cpu_dot<float>(const int n, const float* x, const float* y);

template DLL_EXPORT double caffe_cpu_dot<double>(const int n, const double* x, const double* y);

template <>
float caffe_cpu_asum<float>(const int n, const float* x) {
  return cblas_sasum(n, x, 1);
}

template <>
double caffe_cpu_asum<double>(const int n, const double* x) {
  return cblas_dasum(n, x, 1);
}

template <>
void caffe_cpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  cblas_scopy(n, x, 1, y, 1);
  cblas_sscal(n, alpha, y, 1);
}

template <>
void caffe_cpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  cblas_dcopy(n, x, 1, y, 1);
  cblas_dscal(n, alpha, y, 1);
}

template DLL_EXPORT float caffe_cpu_asum<float>(const int n, const float* x);
template DLL_EXPORT void caffe_cpu_scale<float>(const int n, const float alpha, const float* x, float* y);

template DLL_EXPORT double caffe_cpu_asum<double>(const int n, const double* x);
template DLL_EXPORT void caffe_cpu_scale<double>(const int n, const double alpha, const double* x, double* y);

}  // namespace caffe
