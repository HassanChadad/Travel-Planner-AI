def num_or_inf(value):
    """Return the value, or infinity when it is None, so missing numbers sort last."""
    return value if value is not None else float("inf")