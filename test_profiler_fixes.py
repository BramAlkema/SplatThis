#!/usr/bin/env python3
"""Test PerformanceProfiler timing accumulation fix."""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.splat_this.utils.profiler import PerformanceProfiler

def test_profiler_timing_accumulation():
    """Test that PerformanceProfiler accumulates timing from multiple calls."""
    print("Testing PerformanceProfiler timing accumulation...")

    profiler = PerformanceProfiler()

    @profiler.profile_function("test_function")
    def slow_function(duration: float):
        """A function that takes some time."""
        time.sleep(duration)
        return f"slept for {duration}s"

    # Call the function multiple times
    durations = [0.01, 0.02, 0.01]  # Small durations to keep test fast

    for i, duration in enumerate(durations):
        result = slow_function(duration)
        print(f"Call {i+1}: {result}")

    # Get the metrics
    summary = profiler.get_summary()
    function_metrics = summary['by_function']['test_function']

    print(f"\nMetrics after {len(durations)} calls:")
    print(f"Last call duration: {function_metrics['duration']:.3f}s")
    print(f"Total duration: {function_metrics['total_duration']:.3f}s")
    print(f"Call count: {function_metrics['calls']}")

    # Check that we have the correct call count
    if function_metrics['calls'] == len(durations):
        print(f"✅ Correct call count: {function_metrics['calls']}")
    else:
        print(f"❌ Wrong call count: expected {len(durations)}, got {function_metrics['calls']}")
        return False

    # Check that total duration is greater than any individual duration
    if function_metrics['total_duration'] > function_metrics['duration']:
        print(f"✅ Total duration ({function_metrics['total_duration']:.3f}s) > last call duration ({function_metrics['duration']:.3f}s)")
    else:
        print(f"❌ Total duration should be greater than last call duration")
        return False

    # Check that total duration is approximately the sum of all durations
    expected_total = sum(durations)
    actual_total = function_metrics['total_duration']
    tolerance = 0.05  # 50ms tolerance for timing overhead

    if abs(actual_total - expected_total) < tolerance:
        print(f"✅ Total duration ({actual_total:.3f}s) reasonably matches expected sum ({expected_total:.3f}s)")
    else:
        # More lenient check - just ensure it's in the right ballpark
        if actual_total >= expected_total * 0.8 and actual_total <= expected_total * 3.0:
            print(f"✅ Total duration ({actual_total:.3f}s) is reasonable for expected sum ({expected_total:.3f}s)")
        else:
            print(f"❌ Total duration ({actual_total:.3f}s) is way off from expected sum ({expected_total:.3f}s)")
            return False

    # Check that the last call duration matches the last individual duration
    expected_last = durations[-1]
    actual_last = function_metrics['duration']

    if abs(actual_last - expected_last) < tolerance:
        print(f"✅ Last call duration ({actual_last:.3f}s) matches expected ({expected_last:.3f}s)")
    else:
        print(f"❌ Last call duration ({actual_last:.3f}s) doesn't match expected ({expected_last:.3f}s)")
        return False

    return True

def test_profiler_multiple_functions():
    """Test that profiler handles multiple functions correctly."""
    print("\nTesting PerformanceProfiler with multiple functions...")

    profiler = PerformanceProfiler()

    @profiler.profile_function("function_a")
    def function_a():
        time.sleep(0.01)
        return "a"

    @profiler.profile_function("function_b")
    def function_b():
        time.sleep(0.02)
        return "b"

    # Call functions in sequence
    function_a()
    function_b()
    function_a()  # Call function_a again

    summary = profiler.get_summary()

    # Check function_a metrics
    if 'function_a' in summary['by_function']:
        metrics_a = summary['by_function']['function_a']
        if metrics_a['calls'] == 2:
            print(f"✅ function_a called {metrics_a['calls']} times")
        else:
            print(f"❌ function_a call count wrong: expected 2, got {metrics_a['calls']}")
            return False
    else:
        print("❌ function_a not found in metrics")
        return False

    # Check function_b metrics
    if 'function_b' in summary['by_function']:
        metrics_b = summary['by_function']['function_b']
        if metrics_b['calls'] == 1:
            print(f"✅ function_b called {metrics_b['calls']} time")
        else:
            print(f"❌ function_b call count wrong: expected 1, got {metrics_b['calls']}")
            return False
    else:
        print("❌ function_b not found in metrics")
        return False

    # Check total times are reasonable
    total_a = metrics_a['total_duration']
    total_b = metrics_b['total_duration']

    if total_a > 0.015:  # Should be ~0.02s for 2 calls
        print(f"✅ function_a total time reasonable: {total_a:.3f}s")
    else:
        print(f"❌ function_a total time too low: {total_a:.3f}s")
        return False

    if total_b > 0.015:  # Should be ~0.02s for 1 call
        print(f"✅ function_b total time reasonable: {total_b:.3f}s")
    else:
        print(f"❌ function_b total time too low: {total_b:.3f}s")
        return False

    return True

if __name__ == "__main__":
    print("🔧 Testing PerformanceProfiler Fixes")
    print("=" * 45)

    test1 = test_profiler_timing_accumulation()
    test2 = test_profiler_multiple_functions()

    if all([test1, test2]):
        print("\n🎉 ALL PROFILER TESTS PASSED!")
        print("\nSummary of fixes:")
        print("1. ✅ PerformanceProfiler accumulates timing across multiple calls")
        print("2. ✅ PerformanceProfiler tracks both last call and total duration")
        print("3. ✅ PerformanceProfiler handles multiple functions correctly")
    else:
        print("\n❌ SOME PROFILER TESTS FAILED!")
        sys.exit(1)