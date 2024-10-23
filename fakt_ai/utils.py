from time import time
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from pydantic import validate_call


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


@validate_call
def _get_llm(name: Literal["openai", "anthropic", "groq"]):
    llm_kwargs = {
        "temperature": 0.5,  # TODO: play with temperature - what do we want?
        "timeout": None,
        "max_retries": 3,
    }
    match name:
        case "openai":
            return ChatOpenAI(model="gpt-4o", **llm_kwargs)
        case "anthropic":
            return ChatAnthropic(model="claude-3-5-sonnet-20240620", **llm_kwargs)
        case "groq":
            return ChatGroq(model="llama-3.1-70b-versatile", **llm_kwargs)
