import math

import pytest

from packaging2d import model as m
from packaging2d import study


def test_clean_limits_match_geometry_note():
    # jtc_geometry_constraints.md: F=256, N=3 -> M_clean_max = 64
    assert m.clean_input_limit(256, 3) == 64
    mx, my = m.geometry_2d(100, 3)
    assert my == 98
    assert mx == m.clean_input_limit(100, 3)
    assert m.macs_per_shot_1d(256, 3) == (64 - 3 + 1) * 3
    assert m.macs_per_shot_2d(100, 3) == (mx - 2) * (my - 2) * 9


def test_stated_optics_set_the_via_geometry():
    p = m.StackParams()
    assert (p.gap_um, p.optics.pixel_pitch_um) == (500.0, 5.0)
    assert m.via_geometry_um(p.gap_um, p.via) == (50.0, 100.0)
    assert m.fourier_fill_factor(p.optics) == 1.0
    r = m.evaluate("C", p)
    assert r["aperture_mm"] == pytest.approx(0.5)
    assert "ok_optics" not in r


def test_tgv_parasitics_are_tem_consistent():
    via = m.ViaTech(kind="TGV", returns_per_signal=1)
    par = m.via_parasitics(500.0, via)
    # L C of a two-wire line equals mu eps l^2 regardless of geometry
    expected = m.MU0 * m.EPS0 * via.eps_r_glass * (500e-6) ** 2
    assert par["l_h"] * par["c_f"] == pytest.approx(expected, rel=1e-9)
    tsv = m.via_parasitics(500.0, m.ViaTech(kind="TSV"))
    assert tsv["c_f"] > 10 * par["c_f"]


def test_link_bandwidth_ordering():
    link = m.LinkParams()
    tgv = m.link_metrics(500.0, m.ViaTech(kind="TGV"), link)
    tsv = m.link_metrics(500.0, m.ViaTech(kind="TSV"), link)
    thick = m.link_metrics(2000.0, m.ViaTech(kind="TGV"), link)
    assert tgv["analog_gsps"] > 5.0 > tsv["analog_gsps"] > 0.5
    assert thick["f3db_ghz"] < tgv["f3db_ghz"]
    assert tgv["lanes_per_mm2"] == pytest.approx(50.0)


def test_thermal_network_conserves_energy():
    arch = m.ARCHITECTURES["C"]
    powers = {"top": 10.0, "eic": 5.0, "bottom": 20.0}
    t = m.stack_temperatures(arch, powers, 0.5, 0.2, 500.0, m.ThermalParams())
    assert t["heat_to_sink_w"] + t["heat_to_board_w"] == pytest.approx(35.0)
    assert t["t_bottom_c"] > t["t_top_c"] > m.ThermalParams().t_ambient_c


@pytest.mark.parametrize("f_mod", [1e9, 1e10])
def test_architecture_traffic_contracts(f_mod):
    p = m.with_overrides(m.StackParams(), gap_um=500.0, f_mod_hz=f_mod)
    a, b, c = (m.evaluate(k, p) for k in "ABC")
    assert a["analog_lanes"] > 0 and b["analog_lanes"] > 0
    assert c["analog_lanes"] == 0 and c["digital_lanes"] > 0
    assert b["power_vias"] == 0 and a["power_vias"] > 0
    assert c["f_shot_ghz"] == pytest.approx(f_mod / 1e9)
    assert a["f_shot_ghz"] <= f_mod / 2e9 + 1e-12
    assert c["speedup_workload"] >= 10.0


def test_with_overrides_nested():
    p = m.with_overrides(m.StackParams(), n=64, via__kind="TSV", power__vdd=0.8)
    assert (p.n, p.via.kind, p.power.vdd) == (64, "TSV", 0.8)
    assert p.optics == m.OpticsParams()


def test_study_report_builds():
    report, baseline, records = study.build_report()
    assert "## 5. Feasible design points" in report
    assert "f_eff" not in report
    assert len(baseline) == 12
    assert any(r["feasible"] and r["arch"] == "C" and r["n"] == 100 for r in records)
    assert all(math.isfinite(r["f_shot_ghz"]) for r in records)
