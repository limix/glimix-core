#include "shrink.h"

#include "constants.h"
#include "func_base.h"
#include "gfunc.h"
#include "zero.h"

static inline double g_function_root(double x, void *args) {
  void **args_ = args;
  func_base *fb = (func_base *)args_[0];
  double *fxmax = args_[1];

  return *fxmax - (*fb)(x, args_[2]) - LIK_LOG_MAX;
}

void shrink_interval(ExpFam *ef, Normal *normal, double *a, double xmax,
                     double *b, double fxmax) {
  void *args[] = {ef, normal};
  double fa = g_function_func_base(*a, args);
  double fb = g_function_func_base(*b, args);

  if (!(fa <= fxmax && fb <= fxmax))
    printf("fa fxmax fb: %g %g %g\n", fa, fxmax, fb);
  assert(fa <= fxmax && fb <= fxmax);

  if (fxmax - fa > LIK_LOG_MAX) {
    void *args_[] = {&g_function_func_base, &fxmax, args};
    *a = zero(*a, xmax, 1e-5, &g_function_root, args_);
  }

  if (fxmax - fb > LIK_LOG_MAX) {
    void *args_[] = {&g_function_func_base, &fxmax, args};
    *b = zero(*b, xmax, 1e-5, &g_function_root, args_);
  }

  assert(*a <= *b);
}
