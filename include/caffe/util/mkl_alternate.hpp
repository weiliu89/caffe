#ifndef CAFFE_UTIL_MKL_ALTERNATE_H_
#define CAFFE_UTIL_MKL_ALTERNATE_H_

#ifdef USE_MKL

#include <mkl.h>

#else  // If use MKL, simply include the MKL header

extern "C" {
#include <cblas.h>
}
#include <math.h>
#include "caffe/export.hpp"

// Functions that caffe uses but are not present if MKL is not linked.

// A simple way to define the vsl unary functions. The operation should
// be in the form e.g. y[i] = sqrt(a[i])
#define DECLARE_VSL_UNARY_FUNC(name) \
    template<typename Dtype> DLL_EXPORT void v##name(const int n, const Dtype* a, Dtype* y);

DECLARE_VSL_UNARY_FUNC(Sqr);
DECLARE_VSL_UNARY_FUNC(Exp);
DECLARE_VSL_UNARY_FUNC(Ln);
DECLARE_VSL_UNARY_FUNC(Abs);


#define DECLARE_VSL_UNARY_FUNC_WITH_PARAM(name) \
    template<typename Dtype> DLL_EXPORT void v##name(const int n, const Dtype* a, const Dtype b, Dtype* y);

DECLARE_VSL_UNARY_FUNC_WITH_PARAM(Powx);


#define DECLARE_VSL_BINARY_FUNC(name) \
template<typename Dtype> DLL_EXPORT void v##name(const int n, const Dtype* a, const Dtype* b, Dtype* y);

DECLARE_VSL_BINARY_FUNC(Add);
DECLARE_VSL_BINARY_FUNC(Sub);
DECLARE_VSL_BINARY_FUNC(Mul);
DECLARE_VSL_BINARY_FUNC(Div);

// In addition, MKL comes with an additional function axpby that is not present
// in standard blas. We will simply use a two-step (inefficient, of course) way
// to mimic that.
inline void cblas_saxpby(const int N, const float alpha, const float* X,
                         const int incX, const float beta, float* Y,
                         const int incY) {
  cblas_sscal(N, beta, Y, incY);
  cblas_saxpy(N, alpha, X, incX, Y, incY);
}
inline void cblas_daxpby(const int N, const double alpha, const double* X,
                         const int incX, const double beta, double* Y,
                         const int incY) {
  cblas_dscal(N, beta, Y, incY);
  cblas_daxpy(N, alpha, X, incX, Y, incY);
}

#endif  // USE_MKL
#endif  // CAFFE_UTIL_MKL_ALTERNATE_H_
