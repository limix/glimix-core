#include "bracket.h"
#ifndef NDEBUG
#include <stdio.h>
#endif

void find_bracket(func_base *f, void *args, double a, double b, double lower,
                  double upper, double *left, double *right, double *fleft,
                  double *fright) {
  assert(a < b);
  double fa, fb, fc;
  double c = b;
  b = (a + c) / 2;

  fa = (*f)(a, args);
  fb = (*f)(b, args);
  fc = (*f)(c, args);

  double limit;
  int sign;

  if (fa > fc) {
    swap(&a, &c);
    swap(&fa, &fc);
    limit = lower;
    sign = -1;
  } else {
    limit = upper;
    sign = +1;
  }

  double step = c - a;

  while (fc >= fb && sign * (limit - c) > 0) {
    b = c;
    fb = fc;
    c += step;
    c = fmin(c, upper);
    c = fmax(c, lower);
    fc = (*f)(c, args);
    step *= 2;
  }

  if (a > c) {
    *left = c;
    *right = a;
    *fleft = fc;
    *fright = fa;
  } else {
    *left = a;
    *right = c;
    *fleft = fa;
    *fright = fc;
  }

  assert((fa <= fb && fb >= fc) || (a == lower || c == upper));
}
