#ifndef GFUNC_H
#define GFUNC_H

#include "expfam.h"
#include "normal.h"

static inline double g_function(double x, ExpFam *ef, Normal *normal) {
  const double a = x * (ef->y / ef->aphi + normal->eta);
  const double b = (normal->tau * x * x) / 2;
  const double c = ef->lp0(x) / ef->aphi;
  assert(isfinite(a));
  assert(isfinite(b));
  assert(isfinite(c));

  return (a - b) - c;
}

static inline double g_function_func_base(double x, void *args) {
  void **args_ = args;
  ExpFam *ef = args_[0];
  Normal *normal = args_[1];

  return g_function(x, ef, normal);
}

static inline double g_derivative(double x, ExpFam *ef, Normal *normal) {
  return ef->y / ef->aphi + normal->eta - x * normal->tau -
         exp(ef->lp1(x)) / ef->aphi;
}

#endif /* end of include guard: GFUNC_H */
