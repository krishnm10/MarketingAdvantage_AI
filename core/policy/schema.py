# core/policy/schema.py

from typing import TypedDict, Optional, Dict, Literal


# -------------------------------
# Enumerations (Strict)
# -------------------------------

StrategyLevelName = Literal[
    "LEVEL_0",
    "LEVEL_1",
    "LEVEL_2",
    "LEVEL_3",
    "LEVEL_4",
]

LanguageMode = Literal[
    "neutral",
    "hedged",
    "conservative",
]

HumanInTheLoopMode = Literal[
    "optional",
    "required",
]

EUAIActRiskClass = Literal[
    "minimal_risk",
    "limited_risk",
    "high_risk",
    "high_risk_with_human_oversight",
]


# -------------------------------
# Strategy Level Schema
# -------------------------------

class StrategyLevelConfig(TypedDict, total=False):
    """
    Configuration for a single strategy depth level.
    Fields are intentionally explicit to prevent
    silent misconfiguration.
    """

    # Core behavior
    allow_recommendations: bool
    tap_required: bool

    # Language & output control
    language_mode: LanguageMode

    # Trust & risk
    min_trust_score: float
    refusal_allowed: bool

    # Governance
    human_in_the_loop: HumanInTheLoopMode
    disclaimer_mode: Optional[str]

    # Operational controls
    max_api_cost_per_query: Optional[float]
    max_tokens: Optional[int]

    # Security & compliance
    pii_filter_active: Optional[bool]
    audit_trail: Optional[bool]
    citation_required: Optional[bool]

    # Identity / access
    identity_verification: Optional[Dict[str, object]]


# -------------------------------
# Top-Level Policy Schema
# -------------------------------

class StrategyDepthPolicy(TypedDict):
    """
    Root policy object.
    """

    strategy_depths: Dict[StrategyLevelName, StrategyLevelConfig]

    # EU AI Act classification (mandatory for high-risk)
    eu_ai_act_risk_classification: Dict[
        StrategyLevelName,
        EUAIActRiskClass
    ]

    # Optional role-based caps
    role_based_max_depth: Optional[
        Dict[str, StrategyLevelName]
    ]

    # Metadata
    policy_version: str
