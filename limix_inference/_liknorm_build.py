from cffi import FFI

ffibuilder = FFI()

ffibuilder.set_source("limix_inference._liknorm",
r"""
    #include "liknorm/liknorm.h"

    LikNormMachine* initialize(int n) { return liknorm_create_machine(n); }
    void destroy(LikNormMachine *machine) { liknorm_destroy_machine(machine); }

    enum Lik {
        BERNOULLI,
        BINOMIAL,
        POISSON,
        EXPONENTIAL,
        GAMMA,
        GEOMETRIC
    };

    typedef void lik1d(LikNormMachine*, double);

    void* set_lik[] = {liknorm_set_bernoulli,
                       liknorm_set_binomial,
                       liknorm_set_poisson,
                       liknorm_set_exponential,
                       liknorm_set_gamma,
                       liknorm_set_geometric};

    void apply1d(LikNormMachine *machine,
                 enum Lik lik, double *x, double *tau, double *eta,
                 size_t size, double *log_zeroth, double *mean,
                 double *variance)
    {
        size_t i;
        for (i = 0; i < size; ++i)
        {
            ((lik1d*) set_lik[lik])(machine, x[i]);
            liknorm_set_prior(machine, tau[i], eta[i]);
            liknorm_integrate(machine, log_zeroth+i, mean+i, variance+i);
        }
    }
""", libraries=['liknorm'])

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)

# ffi.cdef(r"""
#     typedef struct {
#         unsigned char r, g, b;
#     } pixel_t;
# """)
# image = ffi.new("pixel_t[]", 800*600)
#
# f = open('data', 'rb')     # binary mode -- important
# f.readinto(ffi.buffer(image))
# f.close()
#
# image[100].r = 255
# image[100].g = 192
# image[100].b = 128
#
# f = open('data', 'wb')
# f.write(ffi.buffer(image))
# f.close()
#
# # from __future__ import (division, absolute_import, print_function,
# #                         unicode_literals)
# #
# # import logging
# #
# # from glob import glob
# # from os.path import join
# #
# #
# # def make_sure_string(msg):
# #     import six
# #     if six.PY2:
# #         return bytes(msg)
# #     else:
# #         return u"%s" % __builtins__['str'](msg)
# #
# #
# # def _make():
# #     from cffi import FFI
# #
# #     logger = logging.getLogger()
# #
# #     logger.debug('CFFI make')
# #
# #     ffi = FFI()
# #
# #     rfolder = join('limix_inference', '_ep', 'liknorm', 'clib')
# #
# #     sources = glob(join(rfolder, 'liknorm', '*.c'))
# #     sources += [join(rfolder, 'liknorm.c')]
# #     sources = [make_sure_string(s) for s in sources]
# #
# #     hdrs = glob(join(rfolder, 'liknorm', '*.h'))
# #     hdrs += [join(rfolder, 'liknorm.h')]
# #     hdrs = [make_sure_string(h) for h in hdrs]
# #
# #     incls = [join(rfolder, 'liknorm')]
# #     incls = [make_sure_string(i) for i in incls]
# #     libraries = [make_sure_string('m')]
# #
# #     logger.debug("Sources: %s", str(sources))
# #     logger.debug('Headers: %s', str(hdrs))
# #     logger.debug('Incls: %s', str(incls))
# #     logger.debug('Libraries: %s', str(libraries))
# #
# #     ffi.set_source(
# #         'limix_inference._ep.liknorm._liknorm_ffi',
# #         '''#include "liknorm.h"''',
# #         include_dirs=incls,
# #         sources=sources,
# #         libraries=libraries,
# #         library_dirs=[],
# #         depends=sources + hdrs,
# #         extra_compile_args=['-std=c99'])
# #
# #     with open(join(rfolder, 'liknorm.h'), 'r') as f:
# #         ffi.cdef(f.read())
# #
# #     return ffi
# #
# #
# # liknorm = _make()
