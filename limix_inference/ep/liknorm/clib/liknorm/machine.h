#ifndef MACHINE_H
#define MACHINE_H

#include "expfam.h"
#include "normal.h"

struct TLikNormMachine {
  double *log_zeroth; // log(mom0)
  double *u;          // mom1/mom0
  double *v;          // mom2/mom0
  double *A0;         // array for the log partition function
  double *logA1;      // array for log(A'(x))
  double *logA2;      // array for log(A''(x))
  double *diff;       // temporary array
  int size;           // size of the above arrays
  ExpFam ef;
  Normal normal;
};

#endif /* end of include guard: MACHINE_H */
