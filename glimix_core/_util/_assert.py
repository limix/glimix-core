def assert_interface(cls, req_attrs):
    attrs = dir(cls)
    private = set(a for a in attrs if a.startswith("_"))
    public = set(attrs) - private

    missing = public - set(req_attrs)
    if missing:
        msg = "The following public attributes exist but have not been asserted: "
        msg += "{}".format(", ".join(list(missing)))
        raise AssertionError(msg)

    missing = set(req_attrs) - public
    if missing:
        msg = "The following attributes have not been found: "
        msg += "{}".format(", ".join(list(missing)))
        raise AssertionError(msg)
