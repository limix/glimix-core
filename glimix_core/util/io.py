from __future__ import print_function

import sys


def wprint(*args, **kwargs):
    print(*args, file=sys.stdout, **kwargs)
