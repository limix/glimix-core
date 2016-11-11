#ifndef BRENT_H
#define BRENT_H

#include "func_base.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>

void find_minimum(double *x0, double *fx0, func_base *f, void *args, double a,
                  double b, double rtol, double atol, int maxiter);

static inline double _neg_func_base(double x, void *args) {
  void **args_ = args;
  func_base *fb = (func_base *)args_[0];

  return -(*fb)(x, args_[1]);
}

static inline void find_maximum(double *x0, double *fx0, func_base *f,
                                void *args, double a, double b, double rtol,
                                double atol, int maxiter) {
  void *args_[] = {f, args};

  find_minimum(x0, fx0, &_neg_func_base, args_, a, b, rtol, atol, maxiter);
  *fx0 = -(*fx0);
}

#endif /* end of include guard: BRENT_H */
