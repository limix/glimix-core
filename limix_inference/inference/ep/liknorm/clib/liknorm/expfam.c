#include "expfam.h"

void fprintf_expfam(FILE *stream, const ExpFam *ef) {
  fprintf(stream, "ExpFam:\n");
  fprintf(stream, "  name       : %s\n", ef->name);
  fprintf(stream, "  y          : %g\n", ef->y);
  fprintf(stream, "  aphi       : %g\n", ef->aphi);
  fprintf(stream, "  c          : %g\n", ef->c);
  fprintf(stream, "  log_aphi   : %g\n", ef->log_aphi);
  fprintf(stream, "  lower bound: %g\n", ef->lower_bound);
  fprintf(stream, "  upper bound: %g\n", ef->upper_bound);
}
