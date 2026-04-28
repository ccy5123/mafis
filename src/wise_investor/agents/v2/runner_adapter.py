"""Adapter shims that fit the v2 prompt builders into the v1 runner contract.

`agents/runner.py` was built around v1 prompt builders with signatures
like `(symbol, value_chain, analyst_text, valuer_text)`. The v2
builders need extra inputs that aren't yet in the runner's variable
list at the call site:

  - Skeptic: needs `AttackPlan` (computed from Stage 3 passed axes).
  - Defender: needs `n_total_attacks` from the same plan.
  - Steward: needs `passed_axes`, `n_total_attacks`, AND the post-Skeptic
    + post-Defender audit summary (computed via `agents.v2.audit`).

This module closes over the Stage-3-derived state (passed axes →
AttackPlan) at adapter construction time, so each shim matches the
runner's positional signature while injecting v2 context internally.
The audit shim is the trickiest: it must produce its block at the
moment the steward prompt is built, which is exactly when both
skeptic_text and defender_text are already available — so we can
inline the audit call inside the steward prompt builder closure.

Why not modify the runner directly? The v1 runner has 1450+ lines of
careful keep_alive / model-swap logic for VRAM-constrained machines.
Touching it risks regressions across the full 6-agent pipeline. A
shim layer keeps the v2 path additive: enable via `run_crew.py --v2`,
fall back to v1 by omitting the flag.
"""

from __future__ import annotations

from collections.abc import Callable

from wise_investor.agents.v2.attack_distribution import (
    AttackPlan,
    distribute_skeptic_attacks,
    total_attacks,
)
from wise_investor.agents.v2.audit import audit_v2_attacks
from wise_investor.agents.v2.defender import (
    make_v2_defender_system_prompt,
    make_v2_defender_user_prompt,
)
from wise_investor.agents.v2.skeptic import (
    make_v2_skeptic_system_prompt,
    make_v2_skeptic_user_prompt,
)
from wise_investor.agents.v2.steward import (
    make_v2_steward_system_prompt,
    make_v2_steward_user_prompt,
)

# ---------------------------------------------------------------------------
# Public bundle — what run_crew.py asks for
# ---------------------------------------------------------------------------


def build_v2_prompt_bundle(passed_axes: list[str]) -> dict[str, object]:
    """Construct the prompt-source kwargs for `run_crew_synthesis`.

    Returns a dict with:
      - skeptic_system, skeptic_user_prompt_builder
      - defender_system, defender_user_prompt_builder
      - steward_system, steward_user_prompt_builder

    The Stage 3 `passed_axes` is the only context this needs. The
    AttackPlan is derived deterministically; the audit happens inside
    the Steward closure (once both Skeptic and Defender outputs exist).
    """
    plan = distribute_skeptic_attacks(passed_axes)
    n_total = total_attacks(plan)

    return {
        "skeptic_system": make_v2_skeptic_system_prompt(),
        "skeptic_user_prompt_builder": _make_skeptic_shim(plan),
        "defender_system": make_v2_defender_system_prompt(),
        "defender_user_prompt_builder": _make_defender_shim(n_total),
        "steward_system": make_v2_steward_system_prompt(),
        "steward_user_prompt_builder": _make_steward_shim(passed_axes, n_total),
    }


# ---------------------------------------------------------------------------
# Closure factories — one per agent role
# ---------------------------------------------------------------------------


def _make_skeptic_shim(
    plan: AttackPlan,
) -> Callable[[str, str, str, str], str]:
    """v1 signature: (symbol, value_chain, analyst_text, valuer_text)."""

    def _builder(
        symbol: str,
        value_chain_text: str,
        analyst_text: str,
        valuer_text: str,
    ) -> str:
        return make_v2_skeptic_user_prompt(
            symbol=symbol,
            plan=plan,
            value_chain_text=value_chain_text,
            analyst_output=analyst_text,
            valuer_output=valuer_text,
        )

    return _builder


def _make_defender_shim(
    n_total_attacks: int,
) -> Callable[[str, str, str, str], str]:
    """v1 signature for Defender (per agents/runner.py call):
    (symbol, analyst_text, valuer_text, skeptic_text).
    """

    def _builder(
        symbol: str,
        analyst_text: str,
        valuer_text: str,
        skeptic_text: str,
    ) -> str:
        return make_v2_defender_user_prompt(
            symbol=symbol,
            analyst_output=analyst_text,
            valuer_output=valuer_text,
            skeptic_output=skeptic_text,
            n_total_attacks=n_total_attacks,
        )

    return _builder


def _make_steward_shim(
    passed_axes: list[str],
    n_total_attacks: int,
) -> Callable[..., str]:
    """v1 signature for Steward (per agents/runner.py call):
    (symbol, value_chain, analyst_text, valuer_text, skeptic_text, defender_text).

    The runner forwards 6 positional args. We only consume what v2
    Steward needs; the value_chain / analyst / valuer args are
    accepted-and-ignored to preserve the signature contract.

    This is also the right place to compute the audit summary —
    skeptic_text and defender_text are both available here for the
    first time in the v2 pipeline.
    """

    def _builder(
        symbol: str,
        value_chain_text: str,  # noqa: ARG001 — preserved for runner contract
        analyst_text: str,      # noqa: ARG001
        valuer_text: str,       # noqa: ARG001
        skeptic_text: str,
        defender_text: str,
    ) -> str:
        audit = audit_v2_attacks(
            skeptic_text=skeptic_text,
            defender_text=defender_text,
            n_expected_attacks=n_total_attacks,
        )
        # AuditResult.summary_text is the canonical pre-formatted block
        # the Steward prompt embeds verbatim — built by audit.py to
        # match the §21 4-rule verdict template. Reusing it keeps any
        # future rule change in one place.
        return make_v2_steward_user_prompt(
            symbol=symbol,
            passed_axes_at_stage_3=passed_axes,
            n_total_attacks=n_total_attacks,
            skeptic_output=skeptic_text,
            defender_output=defender_text,
            audit_summary=audit.summary_text,
        )

    return _builder


__all__ = [
    "build_v2_prompt_bundle",
]
