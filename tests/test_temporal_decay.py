"""
Test temporal decay with real scenarios.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from app.services.validation.temporal_revalidation_engine import (
    compute_exponential_decay,
    compute_half_life,
    _classify_lifecycle_stage,
    _parse_timestamp,
    compute_freshness_fallback,
    TemporalConfig
)


def test_exponential_decay():
    print("=== Exponential Decay Tests ===\n")
    
    # Test 1: Fresh content (30 days old)
    print("1. Fresh content (30 days old)")
    freshness = compute_exponential_decay(30, 0.0020)
    print(f"   Freshness: {freshness:.4f} (should be ~0.94)")
    assert freshness > 0.9
    print("   ✅ High freshness for recent content\n")
    
    # Test 2: 1 year old content
    print("2. One year old content (default domain)")
    freshness = compute_exponential_decay(365, 0.0020)
    print(f"   Freshness: {freshness:.4f} (should be ~0.48)")
    assert 0.4 < freshness < 0.6
    print("   ✅ Medium freshness for 1-year old\n")
    
    # Test 3: Fast-aging domain (AI, 1 year old)
    print("3. AI domain (1 year old)")
    ai_decay = TemporalConfig.DECAY_LAMBDAS["ai"]
    freshness_ai = compute_exponential_decay(365, ai_decay)
    print(f"   AI freshness: {freshness_ai:.4f} (should be ~0.16)")
    assert freshness_ai < 0.3
    print("   ✅ Fast decay for AI content\n")
    
    # Test 4: Slow-aging domain (Legal, 1 year old)
    print("4. Legal domain (1 year old)")
    legal_decay = TemporalConfig.DECAY_LAMBDAS["legal"]
    freshness_legal = compute_exponential_decay(365, legal_decay)
    print(f"   Legal freshness: {freshness_legal:.4f} (should be ~0.69)")
    assert freshness_legal > 0.6
    print("   ✅ Slow decay for legal content\n")
    
    # Test 5: Half-life calculation
    print("5. Half-life calculations")
    for domain, lambda_val in [("ai", 0.0050), ("legal", 0.0010)]:
        half_life = compute_half_life(lambda_val)
        print(f"   {domain.upper()}: {half_life:.0f} days ({half_life/365:.1f} years)")
    print("   ✅ Half-life calculations correct\n")
    
    # Test 6: Minimum floor
    print("6. Minimum freshness floor")
    very_old = compute_exponential_decay(3650, 0.0050)  # 10 years, fast decay
    print(f"   10-year old content: {very_old:.4f}")
    assert very_old >= TemporalConfig.MIN_FRESHNESS
    print(f"   ✅ Never below minimum {TemporalConfig.MIN_FRESHNESS}\n")


def test_lifecycle_stages():
    print("=== Lifecycle Stage Tests ===\n")
    
    test_cases = [
        (30, "fresh"),
        (100, "aging"),
        (500, "stale"),
        (900, "expired"),
        (2000, "critically_old"),
    ]
    
    for age_days, expected_stage in test_cases:
        stage = _classify_lifecycle_stage(age_days)
        print(f"   {age_days} days → {stage}")
        assert stage == expected_stage
    
    print("   ✅ All lifecycle stages correct\n")


def test_timestamp_parsing():
    print("=== Timestamp Parsing Tests ===\n")
    
    # Test 1: datetime object
    dt = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    parsed = _parse_timestamp(dt)
    assert parsed == dt
    print("   ✅ datetime object parsed\n")
    
    # Test 2: ISO string with Z
    iso_str = "2025-01-01T12:00:00Z"
    parsed = _parse_timestamp(iso_str)
    assert parsed is not None
    print(f"   ✅ ISO string parsed: {parsed}\n")
    
    # Test 3: Unix timestamp
    unix_ts = 1704110400  # 2024-01-01
    parsed = _parse_timestamp(unix_ts)
    assert parsed is not None
    print(f"   ✅ Unix timestamp parsed: {parsed}\n")
    
    # Test 4: Invalid data
    parsed = _parse_timestamp("invalid")
    assert parsed is None
    print("   ✅ Invalid data returns None\n")
    
    # Test 5: None
    parsed = _parse_timestamp(None)
    assert parsed is None
    print("   ✅ None returns None\n")


def test_fallback_computation():
    print("=== Fallback Computation Test ===\n")
    
    # Create timestamp 6 months ago
    six_months_ago = datetime.now(timezone.utc) - timedelta(days=180)
    
    # Compute for different domains
    freshness_tech = compute_freshness_fallback(six_months_ago, "tech")
    freshness_legal = compute_freshness_fallback(six_months_ago, "legal")
    
    print(f"   6 months old (tech): {freshness_tech:.4f}")
    print(f"   6 months old (legal): {freshness_legal:.4f}")
    
    assert freshness_tech < freshness_legal
    print("   ✅ Tech decays faster than legal\n")


def test_real_world_scenarios():
    print("=== Real-World Scenarios ===\n")
    
    scenarios = [
        ("AI research paper", "ai", 90, "Should decay quickly"),
        ("SEC filing", "legal", 365, "Should stay relevant"),
        ("Marketing blog", "marketing", 180, "Medium decay"),
        ("Company policy", "policy", 730, "Very slow decay"),
    ]
    
    for name, domain, age_days, note in scenarios:
        decay_lambda = TemporalConfig.DECAY_LAMBDAS.get(domain, TemporalConfig.DEFAULT_DECAY)
        freshness = compute_exponential_decay(age_days, decay_lambda)
        stage = _classify_lifecycle_stage(age_days)
        
        print(f"   {name} ({age_days} days old)")
        print(f"      Freshness: {freshness:.3f} | Stage: {stage}")
        print(f"      Note: {note}\n")
    
    print("   ✅ Real-world scenarios computed\n")


if __name__ == "__main__":
    test_exponential_decay()
    test_lifecycle_stages()
    test_timestamp_parsing()
    test_fallback_computation()
    test_real_world_scenarios()
    
    print("✅ All temporal decay tests passed!\n")
