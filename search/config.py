"""Settings for the search package, read from ``config.yaml``.

Paths resolve against two roots, because the two kinds of file have different
lifetimes. Bulk data (downloads, Parquet, indexes) is gitignored and lives
under the *data root*. Tracked output (results, manifests) lives in the
repository the code runs from, so a result is committed on the branch that
produced it.

The data root is the repository root unless an argument or the
``MAKTSPRAK_DATA_ROOT`` environment variable says otherwise. In a git worktree,
point it at the main checkout, where the corpus is.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
CONFIG_FILE: Path = Path(__file__).resolve().with_name("config.yaml")
DATA_ROOT_ENV: str = "MAKTSPRAK_DATA_ROOT"


class _Frozen(BaseModel):
    """Base for settings: immutable, and unknown keys are an error."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Source(_Frozen):
    """Where the Riksdag's speech datasets are fetched from.

    Attributes:
        dataset_url: The dataset address, with a ``{session}`` placeholder.
        first_session: The earliest session fetched, for example ``2002/03``.
        last_session: The latest session fetched.
    """

    dataset_url: str
    first_session: str
    last_session: str


class CorpusSettings(_Frozen):
    """Which of the fetched sessions the speech corpus covers.

    Attributes:
        first_session: The earliest session the corpus keeps, for example
            ``2003/04``. A session fetched before it is read and counted, and
            left out. The corpus runs to ``source.last_session``.
    """

    first_session: str


class DataPaths(_Frozen):
    """Gitignored bulk data, resolved against the data root.

    Attributes:
        raw_anforanden: The Riksdag's speech datasets as downloaded.
        search: Parquet files, vectors and indexes built from them.
        legacy_corpus: The corpus parsed from the protocol PDFs, one record
            per speaker and protocol.
        party_classifier: The directory of the project's own classifier, whose
            tokenizer the token counts use.
    """

    raw_anforanden: Path
    search: Path
    legacy_corpus: Path
    party_classifier: Path


class TrackedPaths(_Frozen):
    """Tracked output, resolved against the repository root.

    Attributes:
        reports: Results and manifests as JSON.
        questions: The question sets of the evaluation, as JSON Lines.
    """

    reports: Path
    questions: Path


class PassageSettings(_Frozen):
    """How speeches are packed into passages.

    Attributes:
        target_words: The size a passage is packed towards, in words.
        trial_targets: Other sizes, for which the packing report counts the
            passages without writing them.
    """

    target_words: int = Field(gt=0)
    trial_targets: tuple[int, ...]

    @field_validator("trial_targets")
    @classmethod
    def _trial_targets_are_positive(cls, targets: tuple[int, ...]) -> tuple[int, ...]:
        if any(target < 1 for target in targets):
            raise ValueError("Every trial target must be at least one word.")
        return targets


class ReconcileSettings(_Frozen):
    """How the speech corpus is reconciled against the older corpus, and what the report lists.

    Attributes:
        coverage_threshold: The share of the older corpus's in-scope speeches
            that the new corpus must cover; the stop rule of phase 1.
        edge_tokens: How many tokens at the start and at the end of two texts
            are compared for agreement.
        min_sequence_share: The least share of the new speech's pairs of
            adjacent tokens that must be found in the older speech for the two
            to count as the same speech.
        min_shared_tokens: The least number of shared tokens for a pair.
        missing_tokens: The number of the new speech's tokens absent from the
            older speech above which a pair counts as missing words.
        surplus_tokens: The number of surplus tokens in an older speech above
            which a pair counts as carrying a surplus.
        listed_ids: How many protocol ids on one side only, and how many
            unmatched speeches, the report lists.
        listed_deviations: How many of the largest deviations among the pairs
            the report lists.
        listed_missing: How many of the pairs with most of the new speech's
            tokens missing the report lists.
        listed_tokens: How many of the most frequent differing tokens the
            report lists.
    """

    coverage_threshold: float = Field(gt=0, le=1)
    edge_tokens: int = Field(gt=0)
    min_sequence_share: float = Field(gt=0, le=1)
    min_shared_tokens: int = Field(gt=0)
    missing_tokens: int = Field(gt=0)
    surplus_tokens: int = Field(gt=0)
    listed_ids: int = Field(gt=0)
    listed_deviations: int = Field(gt=0)
    listed_missing: int = Field(gt=0)
    listed_tokens: int = Field(gt=0)


class LinkSettings(_Frozen):
    """Defaults for the deep-link check, which its command-line options override.

    Attributes:
        seed: Decides which speeches the check draws.
        max_requests: The most HTTP requests the check makes, retries included.
    """

    seed: int
    max_requests: int = Field(gt=0)


UnanswerableKind = Literal["off_topic", "unknown_speaker", "sd_before_2010", "after_index"]


class QuestionSettings(_Frozen):
    """How the question sets are drawn and written.

    Attributes:
        seed: Decides the draw of passages and the split.
        pilot_seed: Decides the pilot's passages, apart from the real draw.
        review_seed: Decides which questions the review draws.
        sizes: How many questions each generated set gets that is written from a passage.
        pilot_sizes: How many passages the pilot tries each prompt on.
        unanswerable: How many questions of each kind the unanswerable set keeps.
        unanswerable_asked: How many questions of each kind the model is asked for.
        pilot_unanswerable_asked: How many of each kind the pilot asks for.
        rare_share: A stem held by fewer than this share of the passages is rare.
        max_passages_per_slot: The most passages one slot tries.
        interpellation_sessions_per_period: How many sessions per period the
            interpellation set draws.
        pilot_budget_usd: The most the pilot may cost, all runs together.
    """

    seed: int
    pilot_seed: int
    review_seed: int
    sizes: dict[Literal["natural", "filtered", "stance"], int]
    pilot_sizes: dict[Literal["natural", "filtered", "stance"], int]
    unanswerable: dict[UnanswerableKind, int]
    unanswerable_asked: dict[UnanswerableKind, int]
    pilot_unanswerable_asked: int = Field(gt=0)
    rare_share: float = Field(gt=0, lt=1)
    max_passages_per_slot: int = Field(gt=0)
    interpellation_sessions_per_period: int = Field(gt=0)
    pilot_budget_usd: float = Field(gt=0)

    @field_validator("sizes", "pilot_sizes", "unanswerable", "unanswerable_asked")
    @classmethod
    def _counts_are_positive(cls, counts: dict[str, int]) -> dict[str, int]:
        if any(count < 1 for count in counts.values()):
            raise ValueError("Every count must be at least one.")
        return counts


class Prices(_Frozen):
    """List prices of one model, in US dollars per million tokens.

    Attributes:
        input: The price of an input token.
        output: The price of an output token.
    """

    input: float = Field(ge=0)
    output: float = Field(ge=0)


class GenerationSettings(_Frozen):
    """The language model that writes the questions, and what it costs.

    Attributes:
        model: The model id.
        effort: How much the model thinks: ``low``, ``medium``, ``high``,
            ``xhigh`` or ``max``.
        max_tokens: The most output tokens of one answer, thinking included.
        batch_discount: The share taken off the price in a message batch.
        prices: List prices by model id; the model in use must be listed.
    """

    model: str
    effort: Literal["low", "medium", "high", "xhigh", "max"]
    max_tokens: int = Field(gt=0)
    batch_discount: float = Field(ge=0, lt=1)
    prices: dict[str, Prices]

    @model_validator(mode="after")
    def _model_is_priced(self) -> GenerationSettings:
        if self.model not in self.prices:
            raise ValueError(f"The model {self.model} has no price in the settings.")
        return self


class JudgeSettings(_Frozen):
    """The judge and its calibration.

    Attributes:
        model: The model id; its price is read from the generation section.
        effort: How much the model thinks.
        max_tokens: The most output tokens of one answer, thinking included.
        seed: Decides the pilot questions of the calibration pairs and their order.
    """

    model: str
    effort: Literal["low", "medium", "high", "xhigh", "max"]
    max_tokens: int = Field(gt=0)
    seed: int


class BaselineSettings(_Frozen):
    """The BM25 baseline's tuning grid and bootstrap.

    Attributes:
        k1: The values of ``k1`` tried with each analyzer.
        b: The values of ``b`` tried with each analyzer.
        resamples: How many bootstrap resamples behind each interval.
        bootstrap_seed: Decides the resamples.
    """

    k1: tuple[float, ...]
    b: tuple[float, ...]
    resamples: int = Field(gt=0)
    bootstrap_seed: int

    @field_validator("k1")
    @classmethod
    def _k1_not_negative(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if not values or any(value < 0 for value in values):
            raise ValueError("k1 needs at least one value, none below 0.")
        return values

    @field_validator("b")
    @classmethod
    def _b_within_bounds(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if not values or any(not 0 <= value <= 1 for value in values):
            raise ValueError("b needs at least one value, each from 0 to 1.")
        return values


class AnswerSettings(_Frozen):
    """The answer step: the selector's model, the retrieval of its candidates, and the budgets.

    Attributes:
        model: The selector's model id; its price is read from the generation section.
        effort: How much the model thinks; a round may set another.
        max_tokens: The most output tokens of one answer, thinking included.
        analyzer: The analyzer of the BM25 index the candidates come from.
        k1: BM25's ``k1`` for the candidates.
        b: BM25's ``b`` for the candidates.
        monthly_budget_usd: What the page's language model may cost in a
            calendar month before the baseline answers.
        dev_budget_usd: What the rounds on the development split and the
            judge's grades of them may cost together.
        resamples: How many bootstrap resamples behind each interval.
        bootstrap_seed: Decides the resamples.
    """

    model: str
    effort: Literal["low", "medium", "high", "xhigh", "max"]
    max_tokens: int = Field(gt=0)
    analyzer: Literal["stem-stop", "stem-nostop", "nostem-stop", "nostem-nostop"]
    k1: float = Field(ge=0)
    b: float = Field(ge=0, le=1)
    monthly_budget_usd: float = Field(gt=0)
    dev_budget_usd: float = Field(gt=0)
    resamples: int = Field(gt=0)
    bootstrap_seed: int


EmbedderKind = Literal["sentence_transformers", "openrouter"]


class CandidateSettings(_Frozen):
    """One embedding model that the comparison of step 13 tries.

    Attributes:
        name: A short name for the candidate, used for its directories on disk.
        kind: How the model runs: on this machine through sentence-transformers,
            or through OpenRouter's API.
        model: The model's id on Hugging Face for a local model, or at
            OpenRouter for an API model.
        revision: The commit of the model's weights on Hugging Face.
        dimension: The size of the vectors the comparison uses.
        query_prompt: The text put before a question, as the model card gives it.
        document_prompt: The text put before a passage, as the model card gives it.
        max_tokens: The most tokens of a passage a local model reads.
        provider: The provider OpenRouter is told to use, for an API model.
        quantization: The precision that provider serves the model in.
        price_per_million_usd: The listed price of a million input tokens, for
            an API model.
        licensed: Whether the model's licence allows it to run on the site.
    """

    name: str = Field(pattern=r"^[a-z0-9][a-z0-9.-]*$")
    kind: EmbedderKind
    model: str
    revision: str
    dimension: int = Field(gt=0)
    query_prompt: str
    document_prompt: str
    max_tokens: int | None = Field(default=None, gt=0)
    provider: str | None = None
    quantization: str | None = None
    price_per_million_usd: float | None = Field(default=None, ge=0)
    licensed: bool

    @model_validator(mode="after")
    def _fields_fit_the_kind(self) -> CandidateSettings:
        api_fields = (self.provider, self.quantization, self.price_per_million_usd)
        if self.kind == "openrouter":
            if any(value is None for value in api_fields) or self.max_tokens is not None:
                raise ValueError(
                    f"{self.name}: an API model needs a provider, a quantization and a "
                    "price, and no token limit of its own."
                )
        elif self.max_tokens is None or any(value is not None for value in api_fields):
            raise ValueError(
                f"{self.name}: a local model needs a token limit, and no provider, "
                "quantization or price."
            )
        return self


class VectorSettings(_Frozen):
    """The vector leg: the candidates, the subset they are compared on, and the hybrid.

    Attributes:
        candidates: The embedding models, the lightest to operate first, which is
            the order a near tie goes to.
        tie_margin: A candidate this close to the best recall@10 counts as tied with it.
        api_budget_usd: The most that embedding the whole corpus may cost for an
            API model to win.
        run_cap_usd: A run through the API stops before it spends more than this,
            unless its command line raises the cap.
        subset_seed: Decides the speeches drawn into the subset.
        subset_size: The size the subset is filled up to, in passages.
        shard_rows: How many passages one shard of vectors holds.
        window: How many passages each of the two rankings of the hybrid holds.
        rrf_k: The values of the RRF constant tried in the tuning.
        rrf_k_preferred: A tie in the tuning goes to the value of ``k`` nearest this.
        api_batch_inputs: How many texts one request to the API carries.
        api_concurrency: How many requests to the API are in flight at once.
    """

    candidates: tuple[CandidateSettings, ...]
    tie_margin: float = Field(ge=0, lt=1)
    api_budget_usd: float = Field(gt=0)
    run_cap_usd: float = Field(gt=0)
    subset_seed: int
    subset_size: int = Field(gt=0)
    shard_rows: int = Field(gt=0)
    window: int = Field(gt=0)
    rrf_k: tuple[float, ...]
    rrf_k_preferred: float = Field(gt=0)
    api_batch_inputs: int = Field(gt=0)
    api_concurrency: int = Field(gt=0)

    @field_validator("rrf_k")
    @classmethod
    def _rrf_k_positive(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if not values or any(value <= 0 for value in values):
            raise ValueError("rrf_k needs at least one value, each above 0.")
        return values

    @model_validator(mode="after")
    def _candidates_named_once(self) -> VectorSettings:
        names = [candidate.name for candidate in self.candidates]
        if not names or len(set(names)) != len(names):
            raise ValueError("The candidates need at least one entry, each name once.")
        return self

    def candidate(self, name: str) -> CandidateSettings:
        """Return the candidate called *name*.

        Raises:
            KeyError: If no candidate has that name.
        """
        for candidate in self.candidates:
            if candidate.name == name:
                return candidate
        raise KeyError(f"No candidate is called {name!r}.")


class ServiceSettings(_Frozen):
    """The search service: its package, and what one caller may ask.

    Attributes:
        package: The name of the package step 08 builds.
        dataset: The dataset on Hugging Face the package is published as.
        vector_wait_s: How long a question waits for its vector before the
            service answers from BM25 alone.
        question_chars: The most characters a question holds once trimmed.
        rate_window_s: The window of the rate limit, in seconds.
        searches_per_window: How many searches one caller may ask in a window.
        answers_per_window: How many answers one caller may ask in a window.
        embedding_cap_usd: What the embedding API may cost one instance.
    """

    package: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    dataset: str = Field(pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    vector_wait_s: float = Field(gt=0)
    question_chars: int = Field(gt=0)
    rate_window_s: int = Field(gt=0)
    searches_per_window: int = Field(gt=0)
    answers_per_window: int = Field(gt=0)
    embedding_cap_usd: float = Field(gt=0)


class Settings(_Frozen):
    """Everything ``config.yaml`` holds, with paths made absolute.

    Attributes:
        source: Where the speech datasets are fetched from.
        corpus: Which of the fetched sessions the corpus covers.
        data: Gitignored bulk data, under the data root.
        tracked: Tracked output, under the repository root.
        passages: How speeches are packed into passages.
        reconcile: How the corpus is reconciled against the older corpus.
        links: Defaults for the deep-link check.
        questions: How the question sets are drawn and written.
        generation: The language model that writes the questions.
        judge: The judge and its calibration.
        baseline: The BM25 baseline's tuning grid and bootstrap.
        answer: The answer step.
        vectors: The vector leg and the hybrid.
        service: The search service.
    """

    source: Source
    corpus: CorpusSettings
    data: DataPaths
    tracked: TrackedPaths
    passages: PassageSettings
    reconcile: ReconcileSettings
    links: LinkSettings
    questions: QuestionSettings
    generation: GenerationSettings
    judge: JudgeSettings
    baseline: BaselineSettings
    answer: AnswerSettings
    vectors: VectorSettings
    service: ServiceSettings

    @model_validator(mode="after")
    def _judge_is_priced(self) -> Settings:
        if self.judge.model not in self.generation.prices:
            raise ValueError(f"The judge's model {self.judge.model} has no price in the settings.")
        return self

    @model_validator(mode="after")
    def _selector_is_priced(self) -> Settings:
        if self.answer.model not in self.generation.prices:
            raise ValueError(
                f"The selector's model {self.answer.model} has no price in the settings."
            )
        return self


def resolve_data_root(data_root: Path | None = None) -> Path:
    """Pick the data root: the argument, then the environment, then the repository.

    Args:
        data_root: An explicit data root, or ``None`` to fall back.

    Returns:
        The data root as an absolute path.

    Raises:
        FileNotFoundError: If the chosen root is not an existing directory.
    """
    if data_root is None:
        from_env = os.environ.get(DATA_ROOT_ENV)
        data_root = Path(from_env) if from_env else REPO_ROOT
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root {data_root} is not a directory.")
    return data_root.resolve()


def load_settings(config_file: Path = CONFIG_FILE, data_root: Path | None = None) -> Settings:
    """Read the settings file and resolve its paths.

    Args:
        config_file: The YAML file to read.
        data_root: An explicit data root, see :func:`resolve_data_root`.

    Returns:
        The validated, immutable settings.

    Raises:
        FileNotFoundError: If the data root does not exist.
        pydantic.ValidationError: If a key is missing, unknown or mistyped.
    """
    raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    root = resolve_data_root(data_root)
    resolved = dict(raw)
    resolved["data"] = {name: root / value for name, value in raw.get("data", {}).items()}
    resolved["tracked"] = {
        name: REPO_ROOT / value for name, value in raw.get("tracked", {}).items()
    }
    return Settings.model_validate(resolved)
