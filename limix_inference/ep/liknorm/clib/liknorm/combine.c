#include "combine.h"
#include "logaddexp.h"
#include "machine.h"

void combine_steps(LikNormMachine *machine, double *log_zeroth, double *mean,
                   double *variance) {

  LikNormMachine *m = machine;

  double max_log_zeroth = m->log_zeroth[0];
  for (int i = 1; i < m->size; ++i)
    max_log_zeroth = fmax(m->log_zeroth[i], max_log_zeroth);

  (*log_zeroth) = logaddexp_array(m->log_zeroth, m->size, max_log_zeroth);

#ifndef NDEBUG
  for (int i = 0; i < m->size; ++i) {
    if (!isfinite(m->log_zeroth[i])) {
      printf("NOT FINITE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    }
  }
#endif

  for (int i = 0; i < m->size; ++i) {
    m->diff[i] = exp(m->log_zeroth[i] - *log_zeroth);
    if (!isfinite(m->diff[i])) {
      printf("m->log_zeroth[i] *log_zeroth: %g %g\n", m->log_zeroth[i],
             *log_zeroth);
    }
    assert(isfinite(m->diff[i]));
  }

  int left = -1;

  while (m->diff[++left] == 0)
    ;

  int right = m->size;

  while (m->diff[--right] == 0)
    ;
  ++right;

  assert(left < right);

  *mean = 0;
  *variance = 0;

  for (int i = left; i < right; ++i) {
    assert(isfinite(m->u[i]));
    assert(isfinite(m->v[i]));
    *mean += m->u[i] * m->diff[i];
    *variance += m->v[i] * m->diff[i];
  }

  *variance = *variance - (*mean) * (*mean);

  assert(isfinite(*variance));
  assert(isfinite(*mean));
}
