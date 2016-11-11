#include "zero.h"
#include <math.h>

const double r8_epsilon = 2.220446049250313E-016;

double zero(double a, double b, double t, func_base *f, void *args)
// ****************************************************************************80
//
//  Purpose:
//
//    ZERO seeks the root of a function F(X) in an interval [A,B].
//
//  Discussion:
//
//    The interval [A,B] must be a change of sign interval for F.
//    That is, F(A) and F(B) must be of opposite signs.  Then
//    assuming that F is continuous implies the existence of at least
//    one value C between A and B for which F(C) = 0.
//
//    The location of the zero is determined to within an accuracy
//    of 6 * MACHEPS * fabs ( C ) + 2 * T.
//
//    Thanks to Thomas Secretin for pointing out a transcription error in the
//    setting of the value of P, 11 February 2013.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 February 2013
//
//  Author:
//
//    Original FORTRAN77 version by Richard Brent.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Richard Brent,
//    Algorithms for Minimization Without Derivatives,
//    Dover, 2002,
//    ISBN: 0-486-41998-3,
//    LC: QA402.5.B74.
//
//  Parameters:
//
//    Input, double A, B, the endpoints of the change of sign interval.
//
//    Input, double T, a positive error tolerance.
//
//    Input, func_base& F, the name of a user-supplied c++ functor
//    whose zero is being sought.  The input and output
//    of F() are of type double.
//
//    Output, double ZERO, the estimated value of a zero of
//    the function F.
//
{
  double c;
  double d;
  double e;
  double fa;
  double fb;
  double fc;
  double m;
  double macheps;
  double p;
  double q;
  double r;
  double s;
  double sa;
  double sb;
  double tol;

  //
  //  Make local copies of A and B.
  //
  sa = a;
  sb = b;
  fa = f(sa, args);
  fb = f(sb, args);

  c = sa;
  fc = fa;
  e = sb - sa;
  d = e;

  macheps = r8_epsilon;

  for (;;) {
    if (fabs(fc) < fabs(fb)) {
      sa = sb;
      sb = c;
      c = sa;
      fa = fb;
      fb = fc;
      fc = fa;
    }

    tol = 2.0 * macheps * fabs(sb) + t;
    m = 0.5 * (c - sb);

    if ((fabs(m) <= tol) || (fb == 0.0)) {
      break;
    }

    if ((fabs(e) < tol) || (fabs(fa) <= fabs(fb))) {
      e = m;
      d = e;
    } else {
      s = fb / fa;

      if (sa == c) {
        p = 2.0 * m * s;
        q = 1.0 - s;
      } else {
        q = fa / fc;
        r = fb / fc;
        p = s * (2.0 * m * q * (q - r) - (sb - sa) * (r - 1.0));
        q = (q - 1.0) * (r - 1.0) * (s - 1.0);
      }

      if (0.0 < p) {
        q = -q;
      } else {
        p = -p;
      }

      s = e;
      e = d;

      if ((2.0 * p < 3.0 * m * q - fabs(tol * q)) && (p < fabs(0.5 * s * q))) {
        d = p / q;
      } else {
        e = m;
        d = e;
      }
    }
    sa = sb;
    fa = fb;

    if (tol < fabs(d)) {
      sb = sb + d;
    } else if (0.0 < m) {
      sb = sb + tol;
    } else {
      sb = sb - tol;
    }

    fb = f(sb, args);

    if (((0.0 < fb) && (0.0 < fc)) || ((fb <= 0.0) && (fc <= 0.0))) {
      c = sa;
      fc = fa;
      e = sb - sa;
      d = e;
    }
  }
  return sb;
}
