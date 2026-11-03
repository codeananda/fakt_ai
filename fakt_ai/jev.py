"""Fast typed judgments from TypeSafe's Jev model."""

import json
from collections.abc import Mapping
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential
from typesafe_sdk import TypeSafeClient

from fakt_ai import config, ledger
from fakt_ai.cache import cached

_USD_PER_TOKEN = 0.042 / 1_000_000
_CHARS_PER_TOKEN = 4


def judge(state: Any, questions: Mapping[str, Any]) -> dict[str, dict]:
    """
    Ask Jev typed questions about one piece of state.

    Parameters
    ----------
    state : Any
        JSON-serialisable state the questions refer to.
    questions : Mapping[str, Any]
        Question id to `Noul`, `Choice` or `Score`. Ids reach only this code, so each
        question carries its full meaning in its instructions.

    Returns
    -------
    dict[str, dict]
        Question id to {"value", "confidence", "probabilities"}, where "value" is the
        Noul probability, Choice option or Score position.
    """
    specs = {name: repr(q) for name, q in questions.items()}
    return cached("jev", [config.JEV_MODEL, state, specs], lambda: _ask_jev(state, questions))


@retry(wait=wait_exponential(min=2, max=60), stop=stop_after_attempt(4), reraise=True)
def _ask_jev(state: Any, questions: Mapping[str, Any]) -> dict[str, dict]:
    size = len(json.dumps(state)) + sum(len(repr(q)) for q in questions.values())
    ledger.charge("jev", size / _CHARS_PER_TOKEN * _USD_PER_TOKEN)
    api_key = config.secret("TYPESAFE_API_KEY", "typesafe-api-key")
    with TypeSafeClient(api_key=api_key, model=config.JEV_MODEL) as client:
        response = client.system_one(state=state, questions=dict(questions))
    answers = {}
    for name, a in response.nouls.items():
        answers[name] = {"value": a.noul, "confidence": None, "probabilities": None}
    for name, a in response.choices.items():
        answers[name] = {
            "value": a.choice,
            "confidence": a.confidence,
            "probabilities": dict(a.probabilities),
        }
    for name, a in response.scores.items():
        answers[name] = {
            "value": a.score,
            "confidence": a.confidence,
            "probabilities": a.probabilities,
        }
    return answers
