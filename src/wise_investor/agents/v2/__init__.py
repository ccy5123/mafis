"""Constitution-aligned Stage 4 prompts and audit (v2.0).

Implements `docs/constitution.md` §19-21: rubric-aware Skeptic
attacks, strict-concede Defender, 4-rule binary Steward verdict.

This subpackage exists alongside the legacy `agents/` modules
rather than replacing them. The legacy prompts produce generic
adversarial review, which is what the system did before the
universe-driven discovery pipeline existed; the v2 prompts produce
*axis-aligned* adversarial review, which is what the constitution
v2.0 requires after Stage 3 has classified the candidate. The
runner picks v2 when Stage 3 has provided `passed_axes`, otherwise
falls back to legacy. Once the full v2 pipeline lands (Step 5+),
the legacy path can be retired.

Public surface:
  - attack_distribution.distribute_skeptic_attacks(passed_axes)
  - skeptic.make_v2_skeptic_user_prompt(...)
  - defender.make_v2_defender_user_prompt(...)
  - steward.make_v2_steward_user_prompt(...)
  - audit.audit_v2_steward_section(...)
"""

from wise_investor.agents.v2.attack_distribution import (
    ATTACK_AXIS_PRIORITY,
    AttackPlan,
    distribute_skeptic_attacks,
)
from wise_investor.agents.v2.audit import (
    AuditOutcome,
    AuditResult,
    audit_v2_attacks,
)
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

__all__ = [
    "ATTACK_AXIS_PRIORITY",
    "AttackPlan",
    "AuditOutcome",
    "AuditResult",
    "audit_v2_attacks",
    "distribute_skeptic_attacks",
    "make_v2_defender_system_prompt",
    "make_v2_defender_user_prompt",
    "make_v2_skeptic_system_prompt",
    "make_v2_skeptic_user_prompt",
    "make_v2_steward_system_prompt",
    "make_v2_steward_user_prompt",
]
