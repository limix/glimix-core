#ifndef BRACKET_H
#define BRACKET_H

#include "func_base.h"
#include "util.h"
#include <assert.h>
#include <math.h>

void find_bracket(func_base *f, void *args, double a, double b, double lower,
                  double upper, double *left, double *right, double *fleft,
                  double *fright);

#endif /* end of include guard: BRACKET_H */
