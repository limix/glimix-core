from ._liknorm_ffi import lib

class LikNormMachine(object):
    def __init__(self, npoints=500):
        self._machine = lib.create_machine(npoints)

    def finish(self):
        lib.destroy_machine(self._machine)
