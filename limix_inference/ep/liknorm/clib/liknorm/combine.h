#ifndef COMBINE_H
#define COMBINE_H

#include "liknorm.h"

void combine_steps(LikNormMachine *machine, double *log_zeroth, double *mean,
                   double *variance);

#endif /* end of include guard: COMBINE_H */
