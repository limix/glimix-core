#ifndef SHRINK_H
#define SHRINK_H

#include "expfam.h"
#include "normal.h"

void shrink_interval(ExpFam *ef, Normal *normal, double *a, double xmax,
                     double *b, double fxmax);

#endif /* end of include guard: SHRINK_H */
