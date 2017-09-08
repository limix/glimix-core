from __future__ import print_function

import sys


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def wprint(*args, **kwargs):
    print(*args, file=sys.stdout, **kwargs)
