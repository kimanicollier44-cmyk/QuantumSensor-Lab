"""Tests for QuantumSensor-Lab: every number checked against the model."""
import math

import pytest

import main as m


# ---- Ramsey interferometry: SQL vs Heisenberg, asserted numerically
def test_sql_single_probe():
    assert m.sql_uncertainty(1) == 1.0


def test_sql_hundred_probes():
    assert abs(m.sql_uncertainty(100) - 0.1) < 1e-12


def test_heisenberg_hundred_probes():
    assert abs(m.heisenberg_uncertainty(100) - 0.01) < 1e-12


def test_heisenberg_is_sql_squared():
    for n in (4, 100, 10000):
        assert abs(m.heisenberg_uncertainty(n) - m.sql_uncertainty(n) ** 2) < 1e-12


def test_entanglement_gain_is_sqrt_n():
    for n in (100, 10000):
        ratio = m.sql_uncertainty(n) / m.heisenberg_uncertainty(n)
        assert abs(ratio - math.sqrt(n)) < 1e-9


def test_heisenberg_beats_sql_for_n_gt_1():
    for n in (2, 10, 1000):
        assert m.heisenberg_uncertainty(n) < m.sql_uncertainty(n)


def test_ramsey_invalid_probes_raise():
    with pytest.raises(ValueError):
        m.sql_uncertainty(0)
    with pytest.raises(ValueError):
        m.heisenberg_uncertainty(-5)


def test_ramsey_table_shape():
    rows = m.ramsey_table()
    assert rows[0]["probes"] == 1 and rows[-1]["probes"] == 10000
    assert abs(rows[2]["gain_x"] - 10.0) < 1e-9


# ---- NV-center magnetometry
def test_nv_worked_example_value():
    eta = m.nv_sensitivity_t_per_sqrt_hz(m.NV_EXAMPLE_SPINS, m.NV_EXAMPLE_T2STAR_S)
    assert abs(eta - 5.679e-13) / 5.679e-13 < 0.01


def test_nv_spins_sqrt_scaling():
    base = m.nv_sensitivity_t_per_sqrt_hz(1e12, 1e-6)
    assert abs(m.nv_sensitivity_t_per_sqrt_hz(4e12, 1e-6) - base / 2) / base < 1e-9


def test_nv_t2star_sqrt_scaling():
    base = m.nv_sensitivity_t_per_sqrt_hz(1e12, 1e-6)
    assert abs(m.nv_sensitivity_t_per_sqrt_hz(1e12, 4e-6) - base / 2) / base < 1e-9


def test_nv_efficiency_linear():
    base = m.nv_sensitivity_t_per_sqrt_hz(1e12, 1e-6, efficiency=0.01)
    assert abs(m.nv_sensitivity_t_per_sqrt_hz(1e12, 1e-6, efficiency=0.02)
               - base / 2) / base < 1e-9


def test_nv_invalid_inputs_raise():
    with pytest.raises(ValueError):
        m.nv_sensitivity_t_per_sqrt_hz(0, 1e-6)
    with pytest.raises(ValueError):
        m.nv_sensitivity_t_per_sqrt_hz(1e12, 1e-6, efficiency=1.5)


def test_nv_report_consistency():
    rep = m.nv_report()
    assert abs(rep["field_after_100s_t"] - rep["field_after_1s_t"] / 10) \
        / rep["field_after_1s_t"] < 1e-12
    assert rep["sensitivity_pt_per_sqrt_hz"] > 0.5


# ---- atom-interferometer gravimeter
def test_gravimeter_phase_formula():
    expected = m.K_EFF_RAD_PER_M * m.G_EARTH * 0.01
    assert abs(m.gravimeter_phase_rad(m.G_EARTH, 0.1) - expected) < 1e-6


def test_gravimeter_phase_t_squared():
    p1 = m.gravimeter_phase_rad(m.G_EARTH, 0.1)
    p2 = m.gravimeter_phase_rad(m.G_EARTH, 0.2)
    assert abs(p2 - 4 * p1) / p1 < 1e-12


def test_gravimeter_per_shot_value():
    s = m.gravimeter_sensitivity_per_shot()
    assert abs(s - 6.207e-9) / 6.207e-9 < 0.01


def test_gravimeter_averaging_sqrt_law():
    per_shot = m.gravimeter_sensitivity_per_shot()
    assert abs(m.gravimeter_averaged(100) - per_shot / 10) / per_shot < 1e-12


def test_gravimeter_longer_drop_wins():
    assert m.gravimeter_sensitivity_per_shot(interrogation_s=0.2) \
        < m.gravimeter_sensitivity_per_shot(interrogation_s=0.1) / 3.9


# ---- atomic clocks: Allan deviation, labelled table
def test_clock_table_all_labelled():
    rep = m.clock_report()
    assert all(row["labelled_published_class"] is True for row in rep.values())
    assert len(rep) == 3


def test_allan_deviation_sqrt_tau():
    assert abs(m.allan_deviation(2.0e-16, 100) - 2.0e-17) / 2.0e-17 < 1e-9
    assert abs(m.allan_deviation(3.0e-13, 1) - 3.0e-13) < 1e-27


def test_clock_ordering_strontium_best():
    rep = m.clock_report()
    assert rep["strontium-lattice"]["sigma_by_tau"]["10000"] \
        < rep["cesium-fountain"]["sigma_by_tau"]["10000"] \
        < rep["rubidium-oscillator"]["sigma_by_tau"]["10000"]


# ---- spin squeezing
def test_squeezing_10db_factor_and_gain():
    assert abs(m.squeezing_variance_factor(10) - 10.0) < 1e-12
    assert abs(m.squeezing_gain(10) - 3.16228) < 1e-4


def test_squeezing_zero_db_is_unity():
    assert m.squeezing_variance_factor(0) == 1.0
    assert m.squeezing_gain(0) == 1.0


def test_squeezing_20db_factor_100():
    assert abs(m.squeezing_variance_factor(20) - 100.0) < 1e-9


def test_squeezed_uncertainty_consistent():
    assert abs(m.squeezed_uncertainty(100, 10)
               - m.sql_uncertainty(100) / m.squeezing_gain(10)) < 1e-15


# ---- integration-time solver
def test_solver_no_floor_formula():
    assert abs(m.solve_integration_time(1e-12, 1e-13) - 100.0) < 1e-9


def test_solver_reaches_target():
    t = m.solve_integration_time(1e-12, 1e-13)
    assert abs(m.noise_after_time(1e-12, t) - 1e-13) / 1e-13 < 1e-9


def test_solver_floor_unreachable():
    assert m.solve_integration_time(1e-12, 1e-14, floor=1e-13) is None
    assert m.solve_integration_time(1e-12, 1e-13, floor=1e-13) is None


def test_solver_with_floor_reaches_target():
    t = m.solve_integration_time(1e-12, 2e-13, floor=1e-13)
    assert t is not None
    assert abs(m.noise_after_time(1e-12, t, floor=1e-13) - 2e-13) / 2e-13 < 1e-9


def test_noise_decreases_with_time():
    assert m.noise_after_time(1e-12, 100.0) < m.noise_after_time(1e-12, 1.0)


def test_solver_invalid_target_raises():
    with pytest.raises(ValueError):
        m.solve_integration_time(1e-12, 0.0)


# ---- kNN detectability classifier: labels from the model itself
def test_model_snr_formula():
    assert abs(m.model_snr(1e-12, 1e-12, 25.0) - 5.0) < 1e-9


def test_classifier_scenarios():
    rep = m.regime_report()
    assert rep["strong_field_100s"] == "detectable"
    assert rep["pt_field_1s"] == "marginal"
    assert rep["faint_field_short"] == "undetectable"
    assert rep["nv_example_10000s"] == "detectable"


def test_classifier_matches_model_on_scenarios():
    for (s, n, t) in [(1e-9, 1e-12, 100.0), (5e-12, 1e-12, 1.0),
                      (1e-13, 1e-12, 0.01), (1e-12, 5.679e-13, 1.0),
                      (1e-12, 5.679e-13, 10000.0)]:
        assert m.classify_detectability(s, n, t) == m.model_label(s, n, t)


def test_classifier_training_labels_match_model():
    for row in m._CLASSIFIER_TRAIN:
        s, n, t = 10 ** row[0], 10 ** row[1], 10 ** row[2]
        assert m.model_label(s, n, t) == row[3], (s, n, t, row[3])


def test_classifier_invalid_inputs_raise():
    with pytest.raises(ValueError):
        m.classify_detectability(0.0, 1e-12, 1.0)


# ---- gates + seats
def test_hold_gates_all_present():
    assert len(m.HOLD_GATES) == 14
    assert all(v.startswith("HOLD") for v in m.HOLD_GATES.values())


def test_claim_gates_hold():
    for action in ("real_hardware_operation", "live_measurement_claim",
                   "navigation_guidance_claim", "weapons_guidance_claim",
                   "gps_replacement_claim", "medical_diagnosis_claim",
                   "commercial_service_claim"):
        assert m.gate(action) == "HOLD"


def test_sandbox_proceeds():
    assert m.gate("run_simulation") == "PROCEED"


def test_seats_never_override():
    b = m.brains_report()
    assert b["decider"]["seat"] == "Gemini Argon 4 Extended Thinking"
    assert b["decider"]["can_override_hold"] is False
    assert b["advisor"]["can_override_hold"] is False
    assert b["flow"]["rendered"] is False and b["flow"]["live"] == 0


def test_demo_checks_all_pass():
    checks = m.run_checks()
    assert all(ok for _, ok in checks), [n for n, ok in checks if not ok]


def test_demo_deterministic():
    import io
    import contextlib
    out1, out2 = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out1):
        m.demo()
    with contextlib.redirect_stdout(out2):
        m.demo()
    assert out1.getvalue() == out2.getvalue()
