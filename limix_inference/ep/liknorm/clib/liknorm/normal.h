#ifndef NORMAL_H
#define NORMAL_H

#include <stdio.h>

typedef struct {
  double eta;
  double tau;
  double log_tau;
} Normal;

void fprintf_normal(FILE *stream, const Normal *normal);

#endif /* end of include guard: NORMAL_H */
