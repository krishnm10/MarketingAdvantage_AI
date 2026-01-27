# core/policy/validator.py

from dataclasses import dataclass
from typing import List

from core.policy.schema import (
    StrategyDepthPolicy,
    StrategyLevelName,
    StrategyLevelConfig,
    EUAIActRiskClass,
)


# -------------------------------
# Validation Result Models
# -------------------------------

@dataclass(frozen=True)
class PolicyValidationResult:
    valid: bool
    errors: List[str]
    warnings: List[str]


class PolicyValidationError(RuntimeError):
    def __init__(self, errors: List[str]):
        message = "Policy validation failed:\n" + "\n".join(
            f"- {e}" for e in errors
        )
        super().__init__(message)
        self.errors = errors


# -------------------------------
# Validator Entry Point
# -------------------------------

def validate_policy(policy: StrategyDepthPolicy) -> PolicyValidationResult:
    """
    Validates the full strategy depth policy.
    Raises PolicyValidationError on hard failures.
    """

    errors: List[str] = []
    warnings: List[str] = []

    _validate_root(policy, errors)
    _validate_levels(policy, errors, warnings)
    _validate_eu_ai_act(policy, errors)

    if errors:
        raise PolicyValidationError(errors)

    return PolicyValidationResult(
        valid=True,
        errors=[],
        warnings=warnings,
    )


# -------------------------------
# Root-Level Validation
# -------------------------------

def _validate_root(policy: StrategyDepthPolicy, errors: List[str]) -> None:
    if "strategy_depths" not in policy:
        errors.append("Missing 'strategy_depths' section.")

    if "policy_version" not in policy:
        errors.append("Missing 'policy_version'.")

    if not isinstance(policy.get("policy_version"), str):
        errors.append("'policy_version' must be a string.")


# -------------------------------
# Per-Level Validation
# -------------------------------

def _validate_levels(
    policy: StrategyDepthPolicy,
    errors: List[str],
    warnings: List[str],
) -> None:

    depths = policy.get("strategy_depths", {})

    for level_name, level in depths.items():

        _validate_level_name(level_name, errors)
        _validate_level_core(level_name, level, errors)
        _validate_recommendation_rules(level_name, level, errors)
        _validate_trust_rules(level_name, level, errors, warnings)
        _validate_human_oversight(level_name, level, errors)
        _validate_operational_safety(level_name, level, warnings)
        _validate_forbidden_fields(level_name, level, errors)


def _validate_level_name(
    level_name: StrategyLevelName,
    errors: List[str],
) -> None:
    if not level_name.startswith("LEVEL_"):
        errors.append(
            f"Invalid strategy level name: {level_name}"
        )


def _validate_level_core(
    level_name: StrategyLevelName,
    level: StrategyLevelConfig,
    errors: List[str],
) -> None:
    if "allow_recommendations" not in level:
        errors.append(
            f"{level_name}: 'allow_recommendations' is required."
        )

    if "tap_required" not in level:
        errors.append(
            f"{level_name}: 'tap_required' is required."
        )


# -------------------------------
# Recommendation & TAP Rules
# -------------------------------

def _validate_recommendation_rules(
    level_name: StrategyLevelName,
    level: StrategyLevelConfig,
    errors: List[str],
) -> None:

    if level.get("allow_recommendations") is True:
        if level.get("tap_required") is not True:
            errors.append(
                f"{level_name}: Recommendations require tap_required=true."
            )


# -------------------------------
# Trust & Risk Rules
# -------------------------------

def _validate_trust_rules(
    level_name: StrategyLevelName,
    level: StrategyLevelConfig,
    errors: List[str],
    warnings: List[str],
) -> None:

    min_trust = level.get("min_trust_score")

    if min_trust is not None:
        if not (0.0 <= min_trust <= 1.0):
            errors.append(
                f"{level_name}: min_trust_score must be between 0 and 1."
            )

        if min_trust >= 0.95 and level.get("refusal_allowed") is not True:
            warnings.append(
                f"{level_name}: High min_trust_score without refusal_allowed "
                f"may cause frequent deadlocks."
            )


# -------------------------------
# Human-in-the-Loop Rules
# -------------------------------

def _validate_human_oversight(
    level_name: StrategyLevelName,
    level: StrategyLevelConfig,
    errors: List[str],
) -> None:

    if level.get("human_in_the_loop") == "required":
        if level.get("tap_required") is not True:
            errors.append(
                f"{level_name}: human_in_the_loop requires tap_required=true."
            )


# -------------------------------
# Operational Safety Rules
# -------------------------------

def _validate_operational_safety(
    level_name: StrategyLevelName,
    level: StrategyLevelConfig,
    warnings: List[str],
) -> None:

    cost_cap = level.get("max_api_cost_per_query")

    if cost_cap is not None:
        if cost_cap <= 0:
            warnings.append(
                f"{level_name}: max_api_cost_per_query should be positive."
            )

        if level_name != "LEVEL_4":
            warnings.append(
                f"{level_name}: Cost caps are typically applied only to LEVEL_4."
            )


# -------------------------------
# Forbidden / Dangerous Fields
# -------------------------------

def _validate_forbidden_fields(
    level_name: StrategyLevelName,
    level: StrategyLevelConfig,
    errors: List[str],
) -> None:

    if level.get("reasoning_trace_enabled") is True:
        errors.append(
            f"{level_name}: Chain-of-thought exposure is forbidden. "
            f"Use reasoning_summary instead."
        )


# -------------------------------
# EU AI Act Validation
# -------------------------------

def _validate_eu_ai_act(
    policy: StrategyDepthPolicy,
    errors: List[str],
) -> None:

    classifications = policy.get("eu_ai_act_risk_classification", {})

    for level, risk_class in classifications.items():

        if level not in policy.get("strategy_depths", {}):
            errors.append(
                f"EU AI Act classification provided for unknown level: {level}"
            )

        if level in ("LEVEL_3", "LEVEL_4"):
            if risk_class not in (
                "high_risk",
                "high_risk_with_human_oversight",
            ):
                errors.append(
                    f"{level}: Must be classified as high-risk under EU AI Act."
                )
