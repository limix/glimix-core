#ifndef EXPFAM_H
#define EXPFAM_H

#include "liknorm.h"
#include <stdio.h>

typedef struct {
  double y;
  double aphi;
  double log_aphi;
  double c;
  liknorm_lp *lp;
  liknorm_lp0 *lp0;
  liknorm_lp1 *lp1;
  double lower_bound;
  double upper_bound;
  const char *name;
} ExpFam;

void fprintf_expfam(FILE *stream, const ExpFam *ef);

#endif /* end of include guard: EXPFAM_H */
