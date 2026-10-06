import math
import pytest

from flight.parafoil import ParafoilConfig
from guidance.terminal_guidance import (
    GuidanceMode,
    TerminalDecision,
    evaluate_terminal_guidance,
    evaluate_terminal_guidance_from_config,
    should_enter_terminal_guidance,
)


def test_insufficient_glide_range():
    """When Rg < Dt + M, DIRECT_GUIDANCE should continue."""
    # altitude = 200 m, descent_rate = 2.5 m/s -> t = 80 s
    # horizontal_speed = 8.0 m/s -> Rg = 640 m
    # target_distance = 1000 m, margin = 100 m -> threshold = 1100 m
    # Rg (640) < threshold (1100) -> DIRECT_GUIDANCE
    decision = evaluate_terminal_guidance(
        altitude=200.0,
        horizontal_speed=8.0,
        descent_rate=2.5,
        target_distance=1000.0,
        terminal_margin=100.0,
    )

    assert decision.mode == GuidanceMode.DIRECT_GUIDANCE
    assert not decision.should_enter_terminal
    assert math.isclose(decision.glide_range, 640.0)
    assert math.isclose(decision.target_distance, 1000.0)
    assert math.isclose(decision.margin, 100.0)
    assert math.isclose(decision.excess_range, 640.0 - 1100.0)  # -460.0
    assert not should_enter_terminal_guidance(
        altitude=200.0,
        horizontal_speed=8.0,
        descent_rate=2.5,
        target_distance=1000.0,
        terminal_margin=100.0,
    )


def test_exact_boundary_condition():
    """At the exact boundary Rg == Dt + M, TERMINAL_GUIDANCE should be selected."""
    # altitude = 600 m, descent_rate = 2.5 m/s -> t = 240 s
    # horizontal_speed = 8.0 m/s -> Rg = 1920 m
    # Dt = 1820 m, margin = 100 m -> Dt + M = 1920 m
    # Rg == Dt + M -> TERMINAL_GUIDANCE
    decision = evaluate_terminal_guidance(
        altitude=600.0,
        horizontal_speed=8.0,
        descent_rate=2.5,
        target_distance=1820.0,
        terminal_margin=100.0,
    )

    assert decision.mode == GuidanceMode.TERMINAL_GUIDANCE
    assert decision.should_enter_terminal
    assert math.isclose(decision.glide_range, 1920.0)
    assert math.isclose(decision.excess_range, 0.0)
    assert should_enter_terminal_guidance(
        altitude=600.0,
        horizontal_speed=8.0,
        descent_rate=2.5,
        target_distance=1820.0,
        terminal_margin=100.0,
    )


def test_excess_glide_range():
    """When Rg > Dt + M, TERMINAL_GUIDANCE should be selected."""
    # altitude = 600 m, descent_rate = 2.5 m/s -> Rg = 1920 m
    # target_distance = 1200 m, margin = 100 m -> Dt + M = 1300 m
    # Rg (1920) > 1300 -> excess = 620 m -> TERMINAL_GUIDANCE
    decision = evaluate_terminal_guidance(
        altitude=600.0,
        horizontal_speed=8.0,
        descent_rate=2.5,
        target_distance=1200.0,
        terminal_margin=100.0,
    )

    assert decision.mode == GuidanceMode.TERMINAL_GUIDANCE
    assert decision.should_enter_terminal
    assert math.isclose(decision.glide_range, 1920.0)
    assert math.isclose(decision.target_distance, 1200.0)
    assert math.isclose(decision.margin, 100.0)
    assert math.isclose(decision.excess_range, 620.0)
    assert should_enter_terminal_guidance(
        altitude=600.0,
        horizontal_speed=8.0,
        descent_rate=2.5,
        target_distance=1200.0,
        terminal_margin=100.0,
    )


def test_increasing_margin_delays_terminal_guidance():
    """A larger margin requires more excess energy, postponing terminal transition."""
    # Rg = 1920 m, target_distance = 1850 m
    # With margin = 50 m: Dt + M = 1900 m -> Rg >= 1900 -> TERMINAL_GUIDANCE
    decision_low_margin = evaluate_terminal_guidance(
        altitude=600.0,
        horizontal_speed=8.0,
        descent_rate=2.5,
        target_distance=1850.0,
        terminal_margin=50.0,
    )
    assert decision_low_margin.mode == GuidanceMode.TERMINAL_GUIDANCE
    assert decision_low_margin.should_enter_terminal

    # With margin = 100 m: Dt + M = 1950 m -> Rg < 1950 -> DIRECT_GUIDANCE
    decision_high_margin = evaluate_terminal_guidance(
        altitude=600.0,
        horizontal_speed=8.0,
        descent_rate=2.5,
        target_distance=1850.0,
        terminal_margin=100.0,
    )
    assert decision_high_margin.mode == GuidanceMode.DIRECT_GUIDANCE
    assert not decision_high_margin.should_enter_terminal


def test_zero_altitude():
    """At 0 m altitude, Rg = 0 m."""
    # Rg = 0 m. If target_distance = 100 m, margin = 0 -> threshold = 100 m -> DIRECT_GUIDANCE
    d1 = evaluate_terminal_guidance(
        altitude=0.0,
        horizontal_speed=8.0,
        descent_rate=2.5,
        target_distance=100.0,
        terminal_margin=0.0,
    )
    assert d1.mode == GuidanceMode.DIRECT_GUIDANCE
    assert not d1.should_enter_terminal
    assert math.isclose(d1.glide_range, 0.0)

    # If target_distance = 0 m, margin = 0 -> Rg (0) == threshold (0) -> TERMINAL_GUIDANCE
    d2 = evaluate_terminal_guidance(
        altitude=0.0,
        horizontal_speed=8.0,
        descent_rate=2.5,
        target_distance=0.0,
        terminal_margin=0.0,
    )
    assert d2.mode == GuidanceMode.TERMINAL_GUIDANCE
    assert d2.should_enter_terminal
    assert math.isclose(d2.glide_range, 0.0)


def test_zero_target_distance():
    """When vehicle is directly above target (Dt = 0), any altitude with margin 0 triggers terminal."""
    # Rg = 1920 m, target_distance = 0 m, margin = 0 m -> Rg >= 0 -> TERMINAL_GUIDANCE
    decision = evaluate_terminal_guidance(
        altitude=600.0,
        horizontal_speed=8.0,
        descent_rate=2.5,
        target_distance=0.0,
        terminal_margin=0.0,
    )
    assert decision.mode == GuidanceMode.TERMINAL_GUIDANCE
    assert decision.should_enter_terminal
    assert math.isclose(decision.excess_range, 1920.0)


def test_invalid_inputs_raise():
    """All invalid input parameters must raise ValueError."""
    # Negative altitude
    with pytest.raises(ValueError, match="altitude cannot be negative"):
        evaluate_terminal_guidance(-10.0, 8.0, 2.5, 500.0, 50.0)

    # Negative horizontal speed
    with pytest.raises(ValueError, match="horizontal_speed cannot be negative"):
        evaluate_terminal_guidance(600.0, -8.0, 2.5, 500.0, 50.0)

    # Zero descent rate
    with pytest.raises(ValueError, match="descent_rate must be greater than zero"):
        evaluate_terminal_guidance(600.0, 8.0, 0.0, 500.0, 50.0)

    # Negative descent rate
    with pytest.raises(ValueError, match="descent_rate must be greater than zero"):
        evaluate_terminal_guidance(600.0, 8.0, -2.5, 500.0, 50.0)

    # Negative target distance
    with pytest.raises(ValueError, match="target_distance cannot be negative"):
        evaluate_terminal_guidance(600.0, 8.0, 2.5, -500.0, 50.0)

    # Negative margin
    with pytest.raises(ValueError, match="terminal_margin cannot be negative"):
        evaluate_terminal_guidance(600.0, 8.0, 2.5, 500.0, -50.0)


def test_evaluate_from_config():
    """Verify convenience wrapper using ParafoilConfig."""
    config = ParafoilConfig(forward_speed=8.0, descent_rate=2.5)
    decision = evaluate_terminal_guidance_from_config(
        altitude=600.0,
        target_distance=1200.0,
        config=config,
        terminal_margin=100.0,
    )
    assert isinstance(decision, TerminalDecision)
    assert decision.mode == GuidanceMode.TERMINAL_GUIDANCE
    assert decision.should_enter_terminal
    assert math.isclose(decision.glide_range, 1920.0)
    assert math.isclose(decision.excess_range, 620.0)


def test_spatial_gate_prevents_entry_when_outside_capture_radius():
    """Rg is sufficient (1920m >= 1300m), but Dt (1200m) > R_capture (200m) -> remain DIRECT_GUIDANCE."""
    decision = evaluate_terminal_guidance(
        altitude=600.0,
        horizontal_speed=8.0,
        descent_rate=2.5,
        target_distance=1200.0,
        terminal_margin=100.0,
        terminal_capture_radius=200.0,
    )
    assert decision.mode == GuidanceMode.DIRECT_GUIDANCE
    assert not decision.should_enter_terminal
    assert decision.energy_condition
    assert not decision.spatial_capture
    assert decision.capture_radius == 200.0
    assert not should_enter_terminal_guidance(
        altitude=600.0,
        horizontal_speed=8.0,
        descent_rate=2.5,
        target_distance=1200.0,
        terminal_margin=100.0,
        terminal_capture_radius=200.0,
    )


def test_spatial_gate_allows_entry_when_within_capture_radius_and_energy_sufficient():
    """Dt (150m) <= R_capture (200m) AND Rg (960m) >= Dt + M (250m) -> enter TERMINAL_GUIDANCE."""
    decision = evaluate_terminal_guidance(
        altitude=300.0,
        horizontal_speed=8.0,
        descent_rate=2.5,
        target_distance=150.0,
        terminal_margin=100.0,
        terminal_capture_radius=200.0,
    )
    assert decision.mode == GuidanceMode.TERMINAL_GUIDANCE
    assert decision.should_enter_terminal
    assert decision.energy_condition
    assert decision.spatial_capture
    assert should_enter_terminal_guidance(
        altitude=300.0,
        horizontal_speed=8.0,
        descent_rate=2.5,
        target_distance=150.0,
        terminal_margin=100.0,
        terminal_capture_radius=200.0,
    )


def test_insufficient_energy_prevents_entry_even_within_capture_radius():
    """Dt (180m) <= R_capture (200m), but Rg (160m) < Dt + M (230m) -> remain DIRECT_GUIDANCE."""
    decision = evaluate_terminal_guidance(
        altitude=50.0,
        horizontal_speed=8.0,
        descent_rate=2.5,
        target_distance=180.0,
        terminal_margin=50.0,
        terminal_capture_radius=200.0,
    )
    assert decision.mode == GuidanceMode.DIRECT_GUIDANCE
    assert not decision.should_enter_terminal
    assert not decision.energy_condition
    assert decision.spatial_capture
    assert not should_enter_terminal_guidance(
        altitude=50.0,
        horizontal_speed=8.0,
        descent_rate=2.5,
        target_distance=180.0,
        terminal_margin=50.0,
        terminal_capture_radius=200.0,
    )


def test_exact_capture_radius_boundary():
    """Exact boundary Dt == R_capture satisfies spatial gate."""
    # Dt = 200.0 == R_capture = 200.0
    # Rg = 1920.0 >= 300.0
    decision = evaluate_terminal_guidance(
        altitude=600.0,
        horizontal_speed=8.0,
        descent_rate=2.5,
        target_distance=200.0,
        terminal_margin=100.0,
        terminal_capture_radius=200.0,
    )
    assert decision.spatial_capture
    assert decision.energy_condition
    assert decision.should_enter_terminal
    assert decision.mode == GuidanceMode.TERMINAL_GUIDANCE


def test_configured_capture_radius_is_used():
    """Varying R_capture switches between DIRECT_GUIDANCE and TERMINAL_GUIDANCE."""
    # Dt = 250m, Rg = 1920m >= 350m
    d_tight = evaluate_terminal_guidance(
        altitude=600.0,
        horizontal_speed=8.0,
        descent_rate=2.5,
        target_distance=250.0,
        terminal_margin=100.0,
        terminal_capture_radius=200.0,
    )
    assert d_tight.mode == GuidanceMode.DIRECT_GUIDANCE
    assert not d_tight.should_enter_terminal

    d_wide = evaluate_terminal_guidance(
        altitude=600.0,
        horizontal_speed=8.0,
        descent_rate=2.5,
        target_distance=250.0,
        terminal_margin=100.0,
        terminal_capture_radius=300.0,
    )
    assert d_wide.mode == GuidanceMode.TERMINAL_GUIDANCE
    assert d_wide.should_enter_terminal


def test_invalid_capture_radius_raises():
    """Zero or negative capture radius must raise ValueError."""
    with pytest.raises(ValueError, match="terminal_capture_radius must be greater than zero"):
        evaluate_terminal_guidance(600.0, 8.0, 2.5, 100.0, 50.0, terminal_capture_radius=0.0)

    with pytest.raises(ValueError, match="terminal_capture_radius must be greater than zero"):
        evaluate_terminal_guidance(600.0, 8.0, 2.5, 100.0, 50.0, terminal_capture_radius=-50.0)
