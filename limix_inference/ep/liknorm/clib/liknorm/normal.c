#include "normal.h"

void fprintf_normal(FILE *stream, const Normal *normal) {
  fprintf(stream, "Normal:\n");
  fprintf(stream, "  tau    : %g\n", normal->tau);
  fprintf(stream, "  eta    : %g\n", normal->eta);
  fprintf(stream, "  log_tau: %g\n", normal->log_tau);
}
