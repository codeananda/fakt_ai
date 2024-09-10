from time import time


def format_elapsed_time(start: float) -> str:
    """
    Convert the elapsed time since the provided start timestamp to a human-readable string.

    The resulting string is formatted as "Xd Xh Xm X.Xs", where X is the appropriate value
    for days, hours, minutes, and seconds. Each unit is included in the result only if its
    value is greater than zero.

    Parameters
    ----------
    start : float
        The start time, typically obtained from a call to `time()`.

    Returns
    -------
    str
        A human-readable representation of the elapsed time since the start timestamp.
    """

    elapsed_seconds = round(time() - start, 3)
    minutes, seconds = divmod(elapsed_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)

    time_parts = [
        (days, "d"),
        (int(hours), "h"),
        (int(minutes), "m"),
        (round(seconds, 1), "s"),
    ]

    result = "".join(f"{value}{unit}" for value, unit in time_parts if value)
    if result == "":
        result = "0.0s"

    return result
