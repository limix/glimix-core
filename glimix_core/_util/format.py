def format_function(func, params, attrs=None):
    if attrs is None:
        attrs = []
    tname = type(func).__name__
    name = func.name
    kwargs_input = [f"{arg}={val}" for arg, val in params.items()]
    input = ", ".join(kwargs_input)
    msg = f"{tname}({input})"
    if name is not None:
        msg += f": {name}"

    msg += "\n"
    for a in attrs:
        msg += _format_named_arr(a[0], a[1])
    return msg


def _format_named_arr(name, arr):
    from textwrap import TextWrapper

    prefix = f"  {name}: "
    wrapper = TextWrapper(initial_indent=prefix)
    msg = wrapper.fill(str(arr).split("\n")[0]) + "\n"
    prefix = "     "
    wrapper = TextWrapper(initial_indent=prefix, subsequent_indent=prefix)
    for s in str(arr).split("\n")[1:]:
        msg += wrapper.fill(s) + "\n"
    return msg[:-1]
