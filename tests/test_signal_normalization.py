"""
Test signal normalization with all edge cases.
"""

from app.retrieval.repository import (
    _normalize_signal,
    _extract_float,
    _validate_snapshot
)


def test_normalize_signal():
    print("=== Signal Normalization Tests ===\n")
    
    # Test 1: None
    print("1. None input")
    result = _normalize_signal(None)
    assert result == {}, f"Expected empty dict, got {result}"
    print("   ✅ Returns empty dict\n")
    
    # Test 2: Single dict (legacy)
    print("2. Single dict (legacy format)")
    legacy = {"tap_trust_score": 0.75, "validation_version": "1.0"}
    result = _normalize_signal(legacy)
    assert result["tap_trust_score"] == 0.75
    print(f"   ✅ Extracted: {result}\n")
    
    # Test 3: Array of dicts (current)
    print("3. Array of dicts (current format)")
    array = [
        {"tap_trust_score": 0.5, "validated_at": "2026-01-20"},
        {"tap_trust_score": 0.8, "validated_at": "2026-01-24"},  # Latest
    ]
    result = _normalize_signal(array)
    assert result["tap_trust_score"] == 0.8, "Should return LATEST"
    print(f"   ✅ Extracted latest: {result['tap_trust_score']}\n")
    
    # Test 4: Empty array
    print("4. Empty array")
    result = _normalize_signal([])
    assert result == {}
    print("   ✅ Returns empty dict\n")
    
    # Test 5: Mixed array (defensive)
    print("5. Mixed array with invalid items")
    mixed = [
        "invalid",
        None,
        {"tap_trust_score": 0.9},  # Only valid item
        123,
    ]
    result = _normalize_signal(mixed)
    assert result["tap_trust_score"] == 0.9
    print(f"   ✅ Filtered to valid dict: {result}\n")
    
    # Test 6: Extract float with clamping
    print("6. Float extraction with clamping")
    data = {"score": 1.5}  # Out of range
    result = _extract_float(data, "score")
    assert result == 1.0, "Should clamp to 1.0"
    print(f"   ✅ Clamped 1.5 → {result}\n")
    
    data2 = {"score": -0.5}  # Below range
    result2 = _extract_float(data2, "score")
    assert result2 == 0.0, "Should clamp to 0.0"
    print(f"   ✅ Clamped -0.5 → {result2}\n")
    
    # Test 7: Invalid types
    print("7. Invalid type handling")
    data = {"score": "invalid"}
    result = _extract_float(data, "score", default=0.5)
    assert result == 0.5, "Should use default"
    print(f"   ✅ Invalid type → default {result}\n")
    
    print("✅ All tests passed!\n")


if __name__ == "__main__":
    test_normalize_signal()
