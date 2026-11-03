"""Paths, secret resolution and spending limits."""

import getpass
import os
import subprocess
from functools import lru_cache
from pathlib import Path

STATE_DIR = Path(os.environ.get("FAKT_STATE_DIR", Path.cwd() / ".fakt"))
SPEND_CAP_USD = float(os.environ.get("FAKT_SPEND_CAP_USD", "40"))
DEEP_MODEL = os.environ.get("FAKT_DEEP_MODEL", "opus")
FAST_MODEL = os.environ.get("FAKT_FAST_MODEL", "sonnet")
JEV_MODEL = os.environ.get("FAKT_JEV_MODEL", "jev-1.14.0")
HIDE_FACT_CHECKS = os.environ.get("FAKT_HIDE_FACT_CHECKS") == "1"


@lru_cache
def secret(env_var: str, keychain_service: str) -> str | None:
    """
    Resolve a secret from the environment, then the macOS Keychain.

    Parameters
    ----------
    env_var : str
        Environment variable checked first.
    keychain_service : str
        Keychain service name checked second.

    Returns
    -------
    str | None
        The secret, or None when neither location holds it.
    """
    if value := os.environ.get(env_var):
        return value
    found = subprocess.run(
        ["security", "find-generic-password", "-a", getpass.getuser()]
        + ["-s", keychain_service, "-w"],
        capture_output=True,
        text=True,
    )
    return found.stdout.strip() or None
