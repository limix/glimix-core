#ifndef INTEGRAL_H
#define INTEGRAL_H

#include "expfam.h"
#include "normal.h"

void integrate_step(double si, double step, ExpFam *ef, Normal *normal,
                    double *log_zeroth, double *u, double *v, double *A0,
                    double *logA1, double *logA2, double *diff);

#endif /* end of include guard: INTEGRAL_H */
