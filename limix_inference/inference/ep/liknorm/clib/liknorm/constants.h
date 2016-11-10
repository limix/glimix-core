#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <float.h>
#include <math.h>

#ifndef M_E
#define M_E 2.7182818284590452354 /* e */
#endif

#ifndef M_LOG2E
#define M_LOG2E 1.4426950408889634074 /* log_2 e */
#endif

#ifndef M_LOG10E
#define M_LOG10E 0.43429448190325182765 /* log_10 e */
#endif

#ifndef M_LN2
#define M_LN2 0.69314718055994530942 /* log_e 2 */
#endif

#ifndef M_LN10
#define M_LN10 2.30258509299404568402 /* log_e 10 */
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846 /* pi */
#endif

#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923 /* pi/2 */
#endif

#ifndef M_PI_4
#define M_PI_4 0.78539816339744830962 /* pi/4 */
#endif

#ifndef M_1_PI
#define M_1_PI 0.31830988618379067154 /* 1/pi */
#endif

#ifndef M_2_PI
#define M_2_PI 0.63661977236758134308 /* 2/pi */
#endif

#ifndef M_2_SQRTPI
#define M_2_SQRTPI 1.12837916709551257390 /* 2/sqrt(pi) */
#endif

#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880 /* sqrt(2) */
#endif

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440 /* 1/sqrt(2) */
#endif

/* log(pi)/2 */
#define LIK_LPI2 0.572364942924700081938738094323

/* log(2*PI)/2 */
#define LIK_LOG2PI_2 0.9189385332046726695409688545623794196

/* -log(DBL_TRUE_MIN) */
#define LIK_LOG_MAX 744.44007192138121808966388925909996032714843

#define LIK_SQRT_EPSILON (sqrt(DBL_EPSILON))
// #define LIK_SQRT_EPSILON 1.490116119384765625e-08

#define LIK_MAX_EXP (log(DBL_MAX))

#define SQRT32 (4.0 * M_SQRT2)

#ifndef M_1_SQRT2PI
#define M_1_SQRT2PI (M_2_SQRTPI * M_SQRT1_2 / 2.0)
#endif

#define GAUSS_EPSILON (DBL_EPSILON / 2)
#define GAUSS_XUPPER (8.572)
#define GAUSS_XLOWER (-37.519)
#define GAUSS_SCALE (16.0)

#endif /* end of include guard: CONSTANTS_H */
