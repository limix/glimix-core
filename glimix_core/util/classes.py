class NamedClass(object):
    def __init__(self):
        self._name = None

    @property
    def name(self):
        r"""Get or set its name.

        Makes printting more user-friendly.
        """
        return self._name

    @name.setter
    def name(self, name):
        self._name = name
