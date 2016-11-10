#include "combine.h"
#include "expfam.h"
#include "integral.h"
#include "interval.h"
#include "likelihood.h"
#include "liknorm.h"
#include "machine.h"
#include "normal.h"
#include "special.h"
#include <float.h>
#include <stdio.h>

void liknorm_set_expfam(LikNormMachine *machine, double y, double aphi,
                        liknorm_lp *lp, liknorm_lp0 *lp0, liknorm_lp1 *lp1,
                        double lower_bound, double upper_bound) {
  machine->ef.y = y;
  machine->ef.aphi = aphi;
  machine->ef.lp = lp;
  machine->ef.lp0 = lp0;
  machine->ef.lp1 = lp1;
  machine->ef.lower_bound = lower_bound;
  machine->ef.upper_bound = upper_bound;
}

static inline void set_expfam(LikNormMachine *machine, likelihood name,
                              double y, double aphi, double c,
                              const char *string_name) {
  LikNormMachine *m = machine;
  m->ef.name = string_name;
  m->ef.y = y;
  m->ef.aphi = aphi;
  m->ef.log_aphi = log(aphi);
  m->ef.c = c;
  m->ef.lp = get_log_partition(name);
  m->ef.lp0 = get_log_partition0(name);
  m->ef.lp1 = get_log_partition1(name);
  get_likelihood_interval(name, &m->ef.lower_bound, &m->ef.upper_bound);

#ifndef NDEBUG
  printf("Likelihood range: [%g, %g].\n", m->ef.lower_bound, m->ef.upper_bound);
#endif
}

void liknorm_set_binomial(LikNormMachine *machine, double k, double n) {
  set_expfam(machine, BINOMIAL, k / n, 1 / n, logbinom(k, n),
             get_likelihood_string(BINOMIAL));
}
void liknorm_set_bernoulli(LikNormMachine *machine, double k) {
  set_expfam(machine, BERNOULLI, k, 1, 0, get_likelihood_string(BERNOULLI));
}
void liknorm_set_poisson(LikNormMachine *machine, double k) {
  set_expfam(machine, POISSON, k, 1, -logfactorial(k),
             get_likelihood_string(POISSON));
}
void liknorm_set_exponential(LikNormMachine *machine, double x) {
  set_expfam(machine, EXPONENTIAL, x, 1, 0, get_likelihood_string(EXPONENTIAL));
}
void liknorm_set_gamma(LikNormMachine *machine, double x, double a) {
  set_expfam(machine, GAMMA, x, 1 / a, 0, get_likelihood_string(GAMMA));
}
void liknorm_set_geometric(LikNormMachine *machine, double x) {
  set_expfam(machine, GEOMETRIC, x, 1, 0, get_likelihood_string(GEOMETRIC));
}

void liknorm_set_normal(LikNormMachine *machine, double tau, double eta) {
  assert(tau > 0);
  tau = fmax(tau, LIK_SQRT_EPSILON * 2);
  machine->normal.eta = eta;
  machine->normal.tau = tau;
  machine->normal.log_tau = log(tau);
}

LikNormMachine *liknorm_create_machine(int size) {
  LikNormMachine *machine = malloc(sizeof(LikNormMachine));

  machine->size = size;
  machine->log_zeroth = malloc(size * sizeof(double));
  machine->u = malloc(size * sizeof(double));
  machine->v = malloc(size * sizeof(double));
  machine->A0 = malloc(size * sizeof(double));
  machine->logA1 = malloc(size * sizeof(double));
  machine->logA2 = malloc(size * sizeof(double));
  machine->diff = malloc(size * sizeof(double));

  return machine;
}

void liknorm_destroy_machine(LikNormMachine *machine) {
  free(machine->log_zeroth);
  free(machine->u);
  free(machine->v);
  free(machine->A0);
  free(machine->logA1);
  free(machine->logA2);
  free(machine->diff);
  free(machine);
}

void liknorm_moments(LikNormMachine *machine, double *log_zeroth, double *mean,
                     double *variance) {
  double left, right;
  ExpFam *ef = &(machine->ef);
  Normal *normal = &(machine->normal);
  find_interval(ef, normal, &left, &right);
  assert(ef->lower_bound <= left && right <= ef->upper_bound);

  double step = (right - left) / machine->size;
  double *A0 = machine->A0;
  double *logA1 = machine->logA1;
  double *logA2 = machine->logA2;
  double *diff = machine->diff;

  for (int i = 0; i < machine->size; ++i)
    (*ef->lp)(left + step * i + step / 2, A0 + i, logA1 + i, logA2 + i);

  for (int i = 0; i < machine->size; ++i) {
    A0[i] /= ef->aphi;
    logA1[i] -= ef->log_aphi;
    logA2[i] -= ef->log_aphi;
    diff[i] = -exp(logA2[i] - logA1[i]);
  }

  double *u = machine->u;
  double *v = machine->v;
  double *mlog_zeroth = machine->log_zeroth;
  for (int i = 0; i < machine->size; ++i) {
    integrate_step(left + step * i, step, ef, normal, mlog_zeroth++, u++, v++,
                   A0++, logA1++, logA2++, diff++);
  }

  combine_steps(machine, log_zeroth, mean, variance);

  *log_zeroth += machine->ef.c;
  *log_zeroth -= log((2 * M_PI) / normal->tau) / 2;
  *log_zeroth -= (normal->eta * normal->eta) / (2 * normal->tau);
}
