#ifndef LIKELIHOOD_H
#define LIKELIHOOD_H

#include "constants.h"
#include "liknorm.h"

/* -------------------- BINOMIAL -------------------- */
static inline double binomial_log_partition0(double theta) {
  return -theta < LIK_MAX_EXP ? theta + log1p(exp(-theta)) : 0;
}
static inline double binomial_log_partition1(double theta) {
  const double r = -theta < LIK_MAX_EXP ? -log1p(exp(-theta)) : theta;
  assert(isfinite(r) && r >= 0);
  return r;
}
static inline void binomial_log_partition(double theta, double *b0,
                                          double *logb1, double *logb2) {
  *b0 = binomial_log_partition0(theta);
  *logb1 = theta - *b0;
  *logb2 = theta - 2 * (*b0);
}
static inline double binomial_lower_bound(void) { return -DBL_MAX; }
static inline double binomial_upper_bound(void) { return +DBL_MAX; }

/* -------------------- BERNOULLI -------------------- */
static inline double bernoulli_log_partition0(double theta) {
  return binomial_log_partition0(theta);
}
static inline double bernoulli_log_partition1(double theta) {
  return binomial_log_partition1(theta);
}
static inline void bernoulli_log_partition(double theta, double *b0,
                                           double *logb1, double *logb2) {
  binomial_log_partition(theta, b0, logb1, logb2);
}
static inline double bernoulli_lower_bound(void) { return -DBL_MAX; }
static inline double bernoulli_upper_bound(void) { return +DBL_MAX; }

/* -------------------- POISSON -------------------- */
static inline double poisson_log_partition0(double theta) {
  return theta < LIK_MAX_EXP ? exp(theta) : exp(LIK_MAX_EXP);
}
static inline double poisson_log_partition1(double theta) {
  const double r = theta;
  assert(isfinite(r) && r >= 0);
  return r;
}
static inline void poisson_log_partition(double theta, double *b0,
                                         double *logb1, double *logb2) {
  *b0 = poisson_log_partition0(theta);
  *logb2 = *logb1 = theta;
}
static inline double poisson_lower_bound(void) { return -DBL_MAX; }
static inline double poisson_upper_bound(void) { return +DBL_MAX; }

/* -------------------- GAMMA -------------------- */
static inline double gamma_log_partition0(double theta) {
  assert(theta < 0);
  return -log(-theta);
}
static inline double gamma_log_partition1(double theta) {
  const double r = gamma_log_partition0(theta);
  assert(isfinite(r) && r >= 0);
  return r;
}
static inline void gamma_log_partition(double theta, double *b0, double *logb1,
                                       double *logb2) {
  *b0 = gamma_log_partition0(theta);
  *logb1 = *b0;
  *logb2 = 2 * (*b0);
}
static inline double gamma_lower_bound(void) { return -DBL_MAX; }
static inline double gamma_upper_bound(void) { return -DBL_EPSILON; }

/* -------------------- EXPONENTIAL -------------------- */
static inline double exponential_log_partition0(double theta) {
  return gamma_log_partition0(theta);
}
static inline double exponential_log_partition1(double theta) {
  return gamma_log_partition1(theta);
}
static inline void exponential_log_partition(double theta, double *b0,
                                             double *logb1, double *logb2) {
  return gamma_log_partition(theta, b0, logb1, logb2);
}
static inline double exponential_lower_bound(void) { return -DBL_MAX; }
static inline double exponential_upper_bound(void) { return -DBL_EPSILON; }

/* -------------------- GEOMETRIC -------------------- */
static inline double geometric_log_partition0(double theta) {
  return -log1p(-exp(theta));
}
static inline double geometric_log_partition1(double theta) {
  const double r = theta + geometric_log_partition0(theta);
  assert(isfinite(r) && r >= 0);
  return r;
}
static inline void geometric_log_partition(double theta, double *b0,
                                           double *logb1, double *logb2) {
  *b0 = geometric_log_partition0(theta);
  *logb1 = theta + *b0;
  *logb2 = theta + 2 * (*b0);
}
static inline double geometric_lower_bound(void) { return -DBL_MAX; }
static inline double geometric_upper_bound(void) { return -DBL_EPSILON; }

typedef enum {
  BINOMIAL,
  BERNOULLI,
  POISSON,
  GAMMA,
  EXPONENTIAL,
  GEOMETRIC,
} likelihood;

static liknorm_lp *const log_partitions[] = {
    &binomial_log_partition,    &bernoulli_log_partition,
    &poisson_log_partition,     &gamma_log_partition,
    &exponential_log_partition, &geometric_log_partition};

static liknorm_lp0 *const log_partitions0[] = {
    &binomial_log_partition0,    &bernoulli_log_partition0,
    &poisson_log_partition0,     &gamma_log_partition0,
    &exponential_log_partition0, &geometric_log_partition0};

static liknorm_lp1 *const log_partitions1[] = {
    &binomial_log_partition1,    &bernoulli_log_partition1,
    &poisson_log_partition1,     &gamma_log_partition1,
    &exponential_log_partition1, &geometric_log_partition1};

static inline liknorm_lp *get_log_partition(likelihood name) {
  return log_partitions[name];
}

static inline liknorm_lp0 *get_log_partition0(likelihood name) {
  return log_partitions0[name];
}

static inline liknorm_lp1 *get_log_partition1(likelihood name) {
  return log_partitions1[name];
}

const char *get_likelihood_string(likelihood name);

static inline void get_likelihood_interval(likelihood name, double *a,
                                           double *b) {
  double lower_bounds[] = {binomial_lower_bound(),    bernoulli_lower_bound(),
                           poisson_lower_bound(),     gamma_lower_bound(),
                           exponential_lower_bound(), geometric_lower_bound()};
  double upper_bounds[] = {binomial_upper_bound(),    bernoulli_upper_bound(),
                           poisson_upper_bound(),     gamma_upper_bound(),
                           exponential_upper_bound(), geometric_upper_bound()};
  *a = lower_bounds[name];
  *b = upper_bounds[name];
}

#endif /* end of include guard: LIKELIHOOD_H */
