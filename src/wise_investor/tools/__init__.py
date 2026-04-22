"""Python calculation tools exposed to agents as CrewAI Tools.

Implements the "LLM is judgment, Python is calculation" principle (design-v2.2 §7).
Functions here are the single source of truth for all numerical values referenced
in reports. LLMs must not compute ratios or multiples themselves.

Phase 1A adds: calculate_per, calculate_ev_ebitda, get_peer_multiples,
reverse_dcf, verify_number.
"""
