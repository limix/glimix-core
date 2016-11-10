#ifndef LOGADDEXP_H
#define LOGADDEXP_H

#include "compiler.h"
#include "constants.h"
#include <math.h>

/* Implements log(e^x + e^y).
 */
static inline double logaddexp(double x, double y) {
  double tmp = x - y;

  if (LIKNORM_UNLIKELY(x == y))
    return x + M_LN2;

  if (tmp > 0)
    return x + log1p(exp(-tmp));
  else if (tmp <= 0)
    return y + log1p(exp(tmp));

  return tmp;
}

static inline double logaddexp_array(double *x, int n, double xmax) {
  double total = 0;

  for (int i = 0; i < n; ++i)
    total += exp(x[i] - xmax);

  return xmax + log(total);
}

/* Implements log(sx * e^x + sy * e^y).
 *
 * It assumes that sx * e^x + sy * e^y > 0.
 */
inline static double logaddexps(double x, double y, double sx, double sy) {
  double tmp = x - y;

  double sxx = log(fabs(sx)) + x;
  double syy = log(fabs(sy)) + y;

  if (sxx == syy) {
    if (sx * sy > 0)
      return sxx + M_LN2;
    return -DBL_MAX;
  }

  if (sx > 0 && sy > 0) {
    if (tmp > 0)
      return sxx + log1p((sy / sx) * exp(-tmp));
    else if (tmp <= 0)
      return syy + log1p((sx / sy) * exp(tmp));
  } else if (sx > 0)
    return sxx + log1p((sy / sx) * exp(-tmp));
  else
    return syy + log1p((sx / sy) * exp(tmp));
  return tmp;
}

/* Returns log(|c|) and c/|c|, for c = sx * e^x + sy * e^y.
 */
inline static double logaddexpss(double x, double y, double sx, double sy,
                                 double *sign) {
  double sxx = log(fabs(sx)) + x;
  double syy = log(fabs(sy)) + y;

  if (sxx == syy) {
    if (sx * sy > 0) {
      if (sx > 0)
        *sign = +1.0;
      else
        *sign = -1.0;
      return sxx + M_LN2;
    } else {
      *sign = 1.0;
      return -DBL_MAX;
    }
  }

  if (sxx > syy) {
    if (sx >= 0.0)
      *sign = +1.0;
    else
      *sign = -1.0;
  } else {
    if (sy >= 0.0)
      *sign = +1.0;
    else
      *sign = -1.0;
  }

  sx *= *sign;
  sy *= *sign;
  return logaddexps(x, y, sx, sy);
}

#endif /* end of include guard: LOGADDEXP_H */
