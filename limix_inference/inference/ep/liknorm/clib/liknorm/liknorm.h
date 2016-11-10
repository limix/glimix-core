#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <assert.h>
#include <math.h>
#include <stdlib.h>

// computes log(b(x)), b'(x), b''(x)
typedef void liknorm_lp(double x, double *b0, double *logb1, double *logb2);

// returns log(b(x))
typedef double liknorm_lp0(double x);

// returns b'(x)
typedef double liknorm_lp1(double x);

typedef struct TLikNormMachine LikNormMachine;

void liknorm_set_expfam(LikNormMachine *machine, double y, double aphi,
                        liknorm_lp *lp, liknorm_lp0 *lp0, liknorm_lp1 *lp1,
                        double lower_bound, double upper_bound);
void liknorm_set_bernoulli(LikNormMachine *machine, double k);

/* Binomial likelihood.
   k: number of successes
   n: number of trials
*/
void liknorm_set_binomial(LikNormMachine *machine, double k, double n);
void liknorm_set_poisson(LikNormMachine *machine, double k);
void liknorm_set_exponential(LikNormMachine *machine, double x);
void liknorm_set_gamma(LikNormMachine *machine, double x, double a);
void liknorm_set_geometric(LikNormMachine *machine, double x);
void liknorm_set_normal(LikNormMachine *machine, double tau, double eta);
LikNormMachine *liknorm_create_machine(int size);
void liknorm_destroy_machine(LikNormMachine *machine);
void liknorm_moments(LikNormMachine *machine, double *log_zeroth, double *mean,
                     double *variance);

#endif /* end of include guard: DEFINITIONS_H */
