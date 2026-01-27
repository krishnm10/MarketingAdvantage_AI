# core/policy/registry.py

import json
import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from core.policy.schema import StrategyDepthPolicy
from core.policy.validator import PolicyValidationResult


# -------------------------------
# Immutable Policy Wrapper
# -------------------------------

@dataclass(frozen=True)
class RegisteredPolicy:
    """
    Immutable, runtime-safe policy object.
    TAP and other runtime components may only
    read from this structure.
    """

    policy: StrategyDepthPolicy
    policy_hash: str
    policy_version: str
    loaded_at: datetime


# -------------------------------
# Registry (Singleton per Process)
# -------------------------------

_REGISTERED_POLICY: Optional[RegisteredPolicy] = None


# -------------------------------
# Public API
# -------------------------------

def register_policy(
    policy: StrategyDepthPolicy,
    validation_result: PolicyValidationResult,
) -> RegisteredPolicy:
    """
    Registers a validated policy into the runtime registry.
    This may only be called once per process.
    """

    global _REGISTERED_POLICY

    if not validation_result.valid:
        raise RuntimeError(
            "Attempted to register an invalid policy."
        )

    if _REGISTERED_POLICY is not None:
        raise RuntimeError(
            "Policy already registered. Hot-swapping is forbidden."
        )

    canonical = _canonical_json(policy)
    policy_hash = _hash_policy(canonical)

    _REGISTERED_POLICY = RegisteredPolicy(
        policy=policy,
        policy_hash=policy_hash,
        policy_version=policy.get("policy_version"),
        loaded_at=datetime.utcnow(),
    )

    return _REGISTERED_POLICY


def get_registered_policy() -> RegisteredPolicy:
    """
    Returns the active registered policy.
    Runtime components (e.g., TAP) must use this.
    """

    if _REGISTERED_POLICY is None:
        raise RuntimeError(
            "No policy registered. System is not initialized correctly."
        )

    return _REGISTERED_POLICY


# -------------------------------
# Internal Helpers
# -------------------------------

def _canonical_json(policy: StrategyDepthPolicy) -> str:
    """
    Produce deterministic JSON for hashing.
    Order of keys must not affect the hash.
    """

    return json.dumps(
        policy,
        sort_keys=True,
        separators=(",", ":"),
    )


def _hash_policy(canonical_json: str) -> str:
    return hashlib.sha256(
        canonical_json.encode("utf-8")
    ).hexdigest()
