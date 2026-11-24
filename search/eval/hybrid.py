"""Scoring the vector leg and the hybrid: the comparison of step A, the tuning, and the test run.

The rules are those of the addendum of 2026-11-21 in
``docs/search/PREREGISTRATION.md``:

- **Step A.** Each candidate's vectors search the subset on their own. The
  highest recall@10 on the natural questions of the development split wins.
  Among the candidates within the tie margin of it, the first in the settings'
  order is chosen, the lightest to operate. A candidate whose licence does not
  allow it on the site cannot be chosen, and an API model cannot be chosen when
  embedding the whole corpus would cost more than the budget.
- **The tuning.** The hybrid is tried at every passage size and every RRF
  constant. The highest recall@10 on the natural questions wins; a tie goes to
  the higher MRR@10, then to the starting passage size, then to the constant
  nearest the preferred one, and between two equally near, to the smaller.
- **The test run.** Hybrid against BM25 on recall@10 for the natural
  questions decides what runs. Vectors alone against BM25, on the natural and
  the keyword questions, answers the question of the pre-registration and
  decides nothing.

A question is searched among the passages its filter allows, in both rankings.
"""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict

from search.config import REPO_ROOT
from search.corpus.table import PassageTable
from search.eval.assemble import MANIFEST_NAME, QuestionsManifest
from search.eval.baseline import FilterMasks, QuestionResult
from search.eval.metrics import (
    DEPTH,
    question_metrics,
    recall_at,
    reciprocal_rank,
    speech_ranking,
)
from search.eval.questions import EvalQuestion
from search.eval.stats import paired_bootstrap
from search.index.fuse import reciprocal_rank_fusion
from search.index.lexical import Bm25Params, LexicalIndex
from search.index.vector import VectorIndex

#: The set the comparison and the tuning are scored on.
DECIDING_SET = "natural"

#: The sets the second comparison of the test run reads.
VECTOR_COMPARISON_SETS = ("natural", "keyword")

#: The least difference in recall@10, hybrid minus BM25, for the hybrid to run.
HYBRID_MARGIN = 0.03

PREREGISTRATION = "docs/search/PREREGISTRATION.md"


def committed_bytes(relative: str) -> bytes | None:
    """A file as committed at ``HEAD``, or ``None`` when it is not there.

    Args:
        relative: The file's path within the repository, with forward slashes.
    """
    result = subprocess.run(
        ["git", "show", f"HEAD:{relative}"], cwd=REPO_ROOT, capture_output=True, check=False
    )
    return result.stdout if result.returncode == 0 else None


def text_sha256(path: Path) -> str:
    """The SHA-256 of a text file, with line endings made Unix first.

    A checkout that turns line feeds into CRLF therefore keeps the same hash.
    """
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def require_committed(path: Path) -> None:
    """Stop unless *path* is committed at ``HEAD`` as it stands on disk.

    Raises:
        SystemExit: If the file is missing, uncommitted or changed since.
    """
    if not path.is_file():
        raise SystemExit(f"{path} does not exist.")
    relative = path.resolve().relative_to(REPO_ROOT).as_posix()
    committed = committed_bytes(relative)
    on_disk = path.read_bytes().replace(b"\r\n", b"\n")
    if committed is None or committed.replace(b"\r\n", b"\n") != on_disk:
        raise SystemExit(f"{relative} must be committed as it stands before this step runs.")


def require_registered(texts: Sequence[str], what: str) -> None:
    """Stop unless every one of *texts* appears in the pre-registration as committed.

    Raises:
        SystemExit: If one does not.
    """
    registered = (committed_bytes(PREREGISTRATION) or b"").decode("utf-8")
    for text in texts:
        if text not in registered:
            raise SystemExit(
                f"{text} ({what}) is not in the committed {PREREGISTRATION}. Commit the "
                "addendum that records it first."
            )


def require_sets_registered(questions_dir: Path) -> dict[str, str]:
    """Stop unless every question set matches its hash and the hash is registered.

    Returns:
        Each set's hash, by name.
    """
    path = questions_dir / MANIFEST_NAME
    if not path.is_file():
        raise SystemExit(f"{path} does not exist. Assemble the question sets first.")
    manifest = QuestionsManifest.model_validate_json(path.read_text(encoding="utf-8"))
    for record in manifest.sets.values():
        actual = hashlib.sha256((questions_dir / record.file).read_bytes()).hexdigest()
        if actual != record.sha256:
            raise SystemExit(f"{record.file} no longer matches its hash in the manifest.")
    require_registered([record.sha256 for record in manifest.sets.values()], "a question set")
    return {name: record.sha256 for name, record in manifest.sets.items()}


class _Frozen(BaseModel):
    """Base for the records here: immutable, and unknown keys are an error."""

    model_config = ConfigDict(frozen=True, extra="forbid")


@dataclass(frozen=True)
class QueryVectors:
    """The questions' vectors of one run, by question id."""

    rows: Mapping[str, int]
    vectors: np.ndarray

    @classmethod
    def of(cls, question_ids: Sequence[str], vectors: np.ndarray) -> QueryVectors:
        """Pair question ids with their rows.

        Raises:
            ValueError: If the counts differ or an id repeats.
        """
        if len(question_ids) != len(vectors) or len(set(question_ids)) != len(question_ids):
            raise ValueError("Every question needs exactly one vector.")
        return cls(rows={qid: row for row, qid in enumerate(question_ids)}, vectors=vectors)

    def matrix(self, questions: Sequence[EvalQuestion]) -> np.ndarray:
        """The vectors of *questions*, in their order.

        Raises:
            KeyError: If a question has no vector.
        """
        return self.vectors[[self.rows[question.question_id] for question in questions]]


def combined_mask(masks: FilterMasks, question: EvalQuestion, within: np.ndarray | None):
    """The question's filter, narrowed to the passages *within* allows when given."""
    allowed = masks.of(question)
    if within is None:
        return allowed
    return within if allowed is None else allowed & within


def vector_hits(
    index: VectorIndex,
    masks: FilterMasks,
    questions: Sequence[EvalQuestion],
    vectors: QueryVectors,
    window: int,
    within: np.ndarray | None = None,
) -> list[list[int]]:
    """Each question's best passages by vector, best first, under its filter."""
    allowed = [combined_mask(masks, question, within) for question in questions]
    found = index.search_many(vectors.matrix(questions), window, allowed)
    return [[hit.passage for hit in hits] for hits in found]


def lexical_hits(
    index: LexicalIndex,
    params: Bm25Params,
    masks: FilterMasks,
    questions: Sequence[EvalQuestion],
    window: int,
    within: np.ndarray | None = None,
) -> list[list[int]]:
    """Each question's best passages by BM25, best first, under its filter."""
    return [
        [
            hit.passage
            for hit in index.search(
                question.text,
                params,
                limit=window,
                allowed=combined_mask(masks, question, within),
            )
        ]
        for question in questions
    ]


def fused_hits(
    lexical: Sequence[Sequence[int]], vector: Sequence[Sequence[int]], k: float, window: int
) -> list[list[int]]:
    """Each question's two rankings merged by reciprocal rank fusion."""
    return [
        [hit.passage for hit in reciprocal_rank_fusion((words, vectors), k, window)]
        for words, vectors in zip(lexical, vector, strict=True)
    ]


def speech_parties(table: PassageTable) -> dict[str, str]:
    """The party of every speech, from its passages."""
    parties: dict[str, str] = {}
    for speech, party in zip(table.anf_ids, table.parties.tolist(), strict=True):
        parties.setdefault(speech, str(party))
    return parties


def score_rankings(
    table: PassageTable,
    questions: Sequence[EvalQuestion],
    rankings: Sequence[Sequence[int]],
    years: Mapping[str, int],
    parties: Mapping[str, str],
) -> list[QuestionResult]:
    """Turn each question's passage ranking into speeches and score it.

    A question written from a passage is counted under the party of that
    passage's speech, which is its gold speech, so the passages may be of any size.

    Args:
        table: The passages the rankings point into.
        questions: The questions, each with at least one gold speech.
        rankings: Each question's passages, best first.
        years: The year of every speech.
        parties: The party of every speech.

    Returns:
        One result per question, in the order given.
    """
    results = []
    for question, ranking in zip(questions, rankings, strict=True):
        speeches = speech_ranking(ranking, table.anf_ids, DEPTH)
        party = parties[question.gold[0]] if question.passage_id is not None else None
        results.append(
            QuestionResult(
                question_id=question.question_id,
                period=table.period_of_year(years[question.gold[0]]),
                party=party,
                ranking=tuple(speeches),
                metrics=question_metrics(speeches, question.gold),
            )
        )
    return results


def mean_of(results: Sequence[QuestionResult], metric: str) -> float:
    """The mean of one metric over results, rounded as the reports round it."""
    return round(float(np.mean([result.metrics[metric] for result in results])), 6)


class CandidateScore(_Frozen):
    """One candidate's standing in the comparison of step A.

    Attributes:
        name: The candidate.
        recall_10: Its recall@10 on the natural questions of the development split.
        mrr_10: Its MRR@10 on them.
        licensed: Whether its licence allows it on the site.
        corpus_cost_usd: What embedding the whole corpus would cost, for an API model.
    """

    name: str
    recall_10: float
    mrr_10: float
    licensed: bool
    corpus_cost_usd: float | None


class CandidateChoice(_Frozen):
    """The outcome of the comparison of step A.

    Attributes:
        chosen: The candidate chosen.
        best: The candidate with the highest recall@10 among those that can be chosen.
        tied: The candidates within the tie margin of the best, in the settings' order.
        excluded: The candidates that could not be chosen, and why.
    """

    chosen: str
    best: str
    tied: tuple[str, ...]
    excluded: dict[str, str]


def choose_candidate(
    scores: Sequence[CandidateScore], margin: float, budget_usd: float
) -> CandidateChoice:
    """Choose the embedding model by the rule of step A.

    Args:
        scores: Every candidate's standing, in the settings' order, which is the
            order a near tie goes to.
        margin: Candidates within this much of the best recall@10 are tied.
        budget_usd: The most embedding the whole corpus may cost an API model.

    Returns:
        The choice.

    Raises:
        ValueError: If no candidate can be chosen.
    """
    excluded = {}
    eligible = []
    for score in scores:
        if not score.licensed:
            excluded[score.name] = "its licence does not allow it on the site"
        elif score.corpus_cost_usd is not None and score.corpus_cost_usd > budget_usd:
            excluded[score.name] = (
                f"embedding the whole corpus would cost {score.corpus_cost_usd:.2f} USD"
            )
        else:
            eligible.append(score)
    if not eligible:
        raise ValueError("No candidate can be chosen.")
    best = max(eligible, key=lambda score: score.recall_10)
    tied = [score for score in eligible if best.recall_10 - score.recall_10 <= margin + 1e-12]
    return CandidateChoice(
        chosen=tied[0].name,
        best=best.name,
        tied=tuple(score.name for score in tied),
        excluded=excluded,
    )


class HybridPoint(_Frozen):
    """One configuration tried in the tuning of the hybrid, and how it did.

    Attributes:
        target_words: The passage size.
        k: The RRF constant.
        recall_10: Mean recall@10 on the natural questions of the development split.
        mrr_10: Mean MRR@10 on them.
    """

    target_words: int
    k: float
    recall_10: float
    mrr_10: float


def choose_hybrid(grid: Sequence[HybridPoint], start_words: int, preferred_k: float) -> HybridPoint:
    """Pick the hybrid's configuration by the rule of the tuning.

    Args:
        grid: Every configuration tried.
        start_words: The starting passage size, which a tie after MRR@10 goes to.
        preferred_k: A tie after that goes to the constant nearest this.

    Returns:
        The chosen configuration.
    """

    def key(point: HybridPoint) -> tuple[float, float, int, float, float]:
        return (
            -point.recall_10,
            -point.mrr_10,
            point.target_words != start_words,
            abs(point.k - preferred_k),
            point.k,
        )

    return min(grid, key=key)


def tune_grid(
    table: PassageTable,
    questions: Sequence[EvalQuestion],
    lexical: Sequence[Sequence[int]],
    vector: Sequence[Sequence[int]],
    target_words: int,
    k_values: Sequence[float],
    window: int,
) -> list[HybridPoint]:
    """Score the hybrid at one passage size for every RRF constant.

    Args:
        table: The passages of that size.
        questions: The tuning questions.
        lexical: Each question's BM25 ranking at that size.
        vector: Each question's vector ranking at that size.
        target_words: The passage size.
        k_values: The constants to try.
        window: How many passages the fused ranking keeps.

    Returns:
        One point per constant.
    """
    points = []
    for k in k_values:
        recalls, reciprocal = [], []
        for question, ranking in zip(
            questions, fused_hits(lexical, vector, k, window), strict=True
        ):
            speeches = speech_ranking(ranking, table.anf_ids, 10)
            recalls.append(recall_at(speeches, question.gold, 10))
            reciprocal.append(reciprocal_rank(speeches, question.gold, 10))
        points.append(
            HybridPoint(
                target_words=target_words,
                k=k,
                recall_10=round(float(np.mean(recalls)), 6),
                mrr_10=round(float(np.mean(reciprocal)), 6),
            )
        )
    return points


class Comparison(_Frozen):
    """A pre-registered comparison of two systems on one set, from the test run.

    Attributes:
        question_set: The set.
        system: The system compared.
        against: The system it is compared with.
        metric: The metric.
        questions: How many questions.
        system_mean: The system's mean.
        against_mean: The other's mean.
        difference: The system's mean minus the other's.
        interval: The 95% paired bootstrap interval of the difference.
    """

    question_set: str
    system: str
    against: str
    metric: str
    questions: int
    system_mean: float
    against_mean: float
    difference: float
    interval: tuple[float, float]


def compare(
    question_set: str,
    system: str,
    system_results: Sequence[QuestionResult],
    against: str,
    against_results: Sequence[QuestionResult],
    seed: int,
    resamples: int,
    metric: str = "recall@10",
) -> Comparison:
    """Compare two systems question by question with the paired bootstrap.

    Raises:
        ValueError: If the two did not answer the same questions in the same order.
    """
    if [r.question_id for r in system_results] != [r.question_id for r in against_results]:
        raise ValueError("A paired comparison needs the same questions in the same order.")
    first = [result.metrics[metric] for result in system_results]
    second = [result.metrics[metric] for result in against_results]
    difference, low, high = paired_bootstrap(first, second, seed, resamples)
    return Comparison(
        question_set=question_set,
        system=system,
        against=against,
        metric=metric,
        questions=len(first),
        system_mean=round(float(np.mean(first)), 6),
        against_mean=round(float(np.mean(second)), 6),
        difference=round(difference, 6),
        interval=(round(low, 6), round(high, 6)),
    )


class Decision(_Frozen):
    """What the first comparison decides.

    Attributes:
        outcome: ``A`` when the hybrid runs, ``B`` when BM25 runs alone.
        rule: The rule, as the pre-registration states it.
        difference: Hybrid minus BM25 on recall@10 for the natural questions.
        lower_bound: The lower bound of its interval.
    """

    outcome: str
    rule: str
    difference: float
    lower_bound: float


def decide(comparison: Comparison) -> Decision:
    """Apply the rule for hybrid in production to the first comparison."""
    holds = comparison.difference >= HYBRID_MARGIN - 1e-12 and comparison.interval[0] > 0
    return Decision(
        outcome="A" if holds else "B",
        rule=(
            "recall@10 on the natural questions, test split: hybrid minus BM25 is at "
            "least 0.03 and the lower bound of the interval is above 0"
        ),
        difference=comparison.difference,
        lower_bound=comparison.interval[0],
    )


def deviating_parties(
    by_party: Mapping[str, float], overall: float, limit: float = 0.05
) -> dict[str, float]:
    """The parties whose recall@10 lies more than *limit* from the mean, with their deviation.

    Deviations are rounded to six decimals before they are compared, so a party
    exactly *limit* from the mean is not named because of rounding in the sum.
    """
    deviations = {party: round(value - overall, 6) for party, value in sorted(by_party.items())}
    return {party: value for party, value in deviations.items() if abs(value) > limit}
