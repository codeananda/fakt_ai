"""Data shared by every pipeline stage."""

from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field


def _short_id() -> str:
    return uuid4().hex[:8]


class Capture(BaseModel):
    """Whatever the user pointed at, reduced to text."""

    url: str | None = None
    platform: str = "text"
    author: str | None = None
    published: str | None = None
    text: str = ""
    transcript: str = ""
    image_text: str = ""

    @property
    def content(self) -> str:
        parts = [self.text, self.transcript, self.image_text]
        return "\n\n".join(p for p in parts if p)


class Claim(BaseModel):
    """One checkable statement."""

    id: str = Field(default_factory=_short_id)
    text: str = Field(
        description="Self-contained statement, with people, places, dates spelt out"
    )
    quote: str = Field("", description="Exact words from the content this comes from")
    speaker: str | None = None
    domain: str = Field("general", description="health, climate, politics, law, science, ...")
    time_sensitive: bool = Field(False, description="Depends on recent events or news")
    cited_works: list[str] = Field(
        default_factory=list, description="Papers, reports or studies the speaker relies on"
    )


class SourceKind(StrEnum):
    PAPER = "paper"
    DATASET = "dataset"
    OFFICIAL = "official"
    NEWS = "news"
    FACT_CHECK = "fact_check"
    WEB = "web"


class Source(BaseModel):
    """A paper, dataset or web page found as possible evidence."""

    id: str = Field(default_factory=_short_id)
    kind: SourceKind = SourceKind.WEB
    title: str
    url: str
    doi: str | None = None
    pmcid: str | None = None
    open_access_url: str | None = None
    pmcid: str | None = None
    open_access_url: str | None = None
    year: int | None = None
    venue: str | None = None
    authors: list[str] = Field(default_factory=list)
    citations: int | None = None
    abstract: str = ""
    full_text: str = ""
    retracted: bool = False
    publication_types: list[str] = Field(default_factory=list)
    found_by: str = ""

    @property
    def text(self) -> str:
        return self.full_text or self.abstract


class Stance(StrEnum):
    SUPPORTS = "supports"
    REFUTES = "refutes"
    MIXED = "mixed"
    IRRELEVANT = "irrelevant"


class Appraisal(BaseModel):
    """How one source bears on one claim, and how far to trust it."""

    source_id: str
    stance: Stance
    stance_probabilities: dict[str, float] = Field(default_factory=dict)
    relevance: float = Field(0.0, description="0 to 1")
    design: str = "unknown"
    quality: float = Field(0.0, description="0 to 1, from code-weighted signals")
    signals: dict[str, float] = Field(default_factory=dict)
    weight: float = 0.0
    key_finding: str = ""
    quote: str = Field("", description="Verbatim supporting span from the source")
    limitations: str = ""
    quote_check: str = Field(
        "", description="verified, contradicted, unsupported or not_found, set by code"
    )
    cited_check: str = Field(
        "", description="For a work the speaker cited: does it say what they claimed"
    )
    excluded: str = Field("", description="Why this source was left out of the verdict")


class VerdictLabel(StrEnum):
    TRUE = "True"
    MOSTLY_TRUE = "Mostly true"
    CONTESTED = "Contested"
    MISLEADING = "Misleading"
    MOSTLY_FALSE = "Mostly false"
    FALSE = "False"
    UNPROVEN = "Unproven"


class Verdict(BaseModel):
    """The judged outcome for one claim."""

    label: VerdictLabel
    probability_true: float = Field(
        description="0 to 1, how likely the claim is true as stated"
    )
    evidence_balance: float = Field(
        0.5, description="0 to 1, computed from the weighted appraisals alone"
    )
    consensus: str = Field("", description="Where experts stand, and how split they are")
    summary: str = Field(description="Two or three plain sentences")
    reasoning: str = Field(description="Full argument, citing sources as [id]")
    strongest_for: str = ""
    strongest_against: str = ""
    what_would_change_this: str = ""


class ClaimCheck(BaseModel):
    """A claim with everything found about it."""

    claim: Claim
    sources: list[Source] = Field(default_factory=list)
    appraisals: list[Appraisal] = Field(default_factory=list)
    verdict: Verdict | None = None
    failure: str = Field("", description="Why this claim could not be checked")


class PostProfile(BaseModel):
    """How the post as a whole behaves, separate from whether it is true."""

    signal_vs_hype: float = Field(0.5, description="0 hype, 1 substantive signal")
    fact_vs_fiction: float = Field(
        0.5, description="0 fiction, 1 fact, from the claim verdicts"
    )
    tactics: list[str] = Field(default_factory=list)


class Report(BaseModel):
    """The finished check."""

    id: str = Field(default_factory=_short_id)
    created: str = ""
    capture: Capture
    checks: list[ClaimCheck] = Field(default_factory=list)
    profile: PostProfile = Field(default_factory=PostProfile)
    one_page: str = Field("", description="Plain-English summary in Markdown")
    detail: str = Field("", description="Full report in Markdown")
