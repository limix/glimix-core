#include "bracket.h"
#include "brent.h"
#include "gfunc.h"
#include "interval.h"
#include "shrink.h"
#include <math.h>

#define TIMES_STD 7
#define REPS 1e-5
#define AEPS 1e-5
#define MAXITER 100

static inline void find_first_interval(ExpFam *ef, Normal *normal, double *a,
                                       double *b) {
  double std = sqrt(1 / normal->tau);
  double mu = normal->eta / normal->tau;

  /* initial reasonable interval */
  *a = mu - TIMES_STD * std;
  *b = mu + TIMES_STD * std;

  *a = fmax(*a, ef->lower_bound);
  *b = fmin(*b, ef->upper_bound);

  const double smallest_step = fmax(fabs(*a), fabs(*b)) * REPS + AEPS;

  if (*b - *a < smallest_step) {
    if (*a - ef->lower_bound >= *b - ef->lower_bound)
      *a -= smallest_step;
    else
      *b += smallest_step;
  }
  assert(*b - *a >= smallest_step);
}

void find_interval(ExpFam *ef, Normal *normal, double *left, double *right) {

  double a, b;
  find_first_interval(ef, normal, &a, &b);
#ifndef NDEBUG
  printf("First interval: [%g, %g].\n", a, b);
#endif
  void *args[] = {ef, normal};
  double fleft, fright;
  find_bracket(&g_function_func_base, args, a, b, ef->lower_bound,
               ef->upper_bound, left, right, &fleft, &fright);

  assert(*left < *right);
  assert(ef->lower_bound <= *left);
  assert(*right <= ef->upper_bound);
  assert(isfinite(*left) && isfinite(*right));
  assert(isfinite(fleft) && isfinite(fright));

  a = fmin(a, *left);
  b = fmax(b, *right);
#ifndef NDEBUG
  printf("Second interval: [%g, %g].\n", a, b);
#endif

  double xmax, fxmax;

  find_maximum(&xmax, &fxmax, &g_function_func_base, args, a, b, REPS, AEPS,
               MAXITER);

  assert(isfinite(xmax));
  assert(isfinite(fxmax));
#ifndef NDEBUG
  printf("Maximum %g at %g.\n", fxmax, xmax);
  printf("g(%g), g(%g), g(%g) = %g, %g, %g\n", a, xmax, b,
         g_function_func_base(a, args), fxmax, g_function_func_base(b, args));
#endif

  assert(a <= xmax && xmax <= b);

  shrink_interval(ef, normal, &a, xmax, &b, fxmax);

#ifndef NDEBUG
  printf("Third interval: [%g, %g].\n", a, b);
  double fa = g_function_func_base(a, args);
  double fb = g_function_func_base(b, args);
  printf("Evaluation at %g, %g, %g: %g, %g, %g\n", a, xmax, b, fa, fxmax, fb);
#endif

  *left = a;
  *right = b;
}
