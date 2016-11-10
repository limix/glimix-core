#ifndef UTIL_H
#define UTIL_H

#include <math.h>
#include <stdbool.h>
#include <string.h>

static inline void swap(double *a, double *b) {
  double c = *a;
  *a = *b;
  *b = c;
}

static inline bool streq(const char *a, const char *b) {
  return strcmp(a, b) == 0 ? true : false;
}

static inline bool almost_equal(double a, double b) {
  const double rtol = 1e-4;
  const double atol = 1e-4;
  return fabs(a - b) < rtol * fmax(fabs(a), fabs(b)) + atol;
}

static inline bool almost_equal_tol(double a, double b, double tol) {
  return fabs(a - b) < tol * fmax(fabs(a), fabs(b)) + tol;
}

#define LEN(foo) (sizeof(foo) / sizeof(foo[0]))

#endif /* end of include guard: UTIL_H */
