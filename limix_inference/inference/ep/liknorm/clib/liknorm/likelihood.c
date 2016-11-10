#include "likelihood.h"

const char *get_likelihood_string(likelihood name) {
  static const char *names[] = {"binomial", "bernoulli",   "poisson",
                                "gamma",    "exponential", "geometric"};
  return names[name];
}
