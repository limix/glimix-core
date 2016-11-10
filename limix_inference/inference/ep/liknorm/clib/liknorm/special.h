#ifndef SPECIAL_H
#define SPECIAL_H

#include "constants.h"
#include <math.h>

/* Cumulative distribution function of the Normal distribution.
 */
double cdf(double x);

/* Log of the cumulative distribution function of the Normal distribution.
 */
double logcdf(double x);

/* Log of the probability distribution function of the Normal distribution.
 */
static inline double logpdf(double x) { return -(x * x) / 2 - LIK_LOG2PI_2; }

static inline double logbinom(double k, double n) {
  return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1);
}

static inline double logfactorial(double k) { return lgamma(k + 1); }

#endif /* end of include guard: SPECIAL_H */
