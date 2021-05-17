def assert_interface(cls, callables, properties):
    from .cache import cached_property

    attrs = dir(cls)
    private = set(a for a in attrs if a.startswith("_"))
    public = set(attrs) - private

    public_methods = [a for a in public if callable(getattr(cls, a))]
    public_props = [
        a for a in public if isinstance(getattr(cls, a), (property, cached_property))
    ]

    _assert_callables(set(public_methods), callables)
    _assert_properties(set(public_props), properties)


def _assert_callables(public, required):
    missing = public - set(required)
    if missing:
        msg = "The following public methods exist but have not been asserted: "
        msg += "{}".format(", ".join(list(missing)))
        raise AssertionError(msg)

    missing = set(required) - public
    if missing:
        msg = "The following methods have not been found: "
        msg += "{}".format(", ".join(list(missing)))
        raise AssertionError(msg)


def _assert_properties(public, required):
    missing = public - set(required)
    if missing:
        msg = "The following public properties exist but have not been asserted: "
        msg += "{}".format(", ".join(list(missing)))
        raise AssertionError(msg)

    missing = set(required) - public
    if missing:
        msg = "The following properties have not been found: "
        msg += "{}".format(", ".join(list(missing)))
        raise AssertionError(msg)
