"""Layer 1 — prediction_service 임계값 단위 테스트.

실행:
    uv run pytest tests/test_prediction_service.py -v

API 키: 불필요 (완전 결정론적)
대상:   manufacturing_agent/services/prediction_service.py → compute_partial_risks()
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest
from manufacturing_agent.services.prediction_service import compute_partial_risks, _level


# ── _level() 기준값 ────────────────────────────────────────────────────────────

class TestLevel:
    def test_high(self):
        assert _level(0.66) == "high"
        assert _level(1.0) == "high"

    def test_medium(self):
        assert _level(0.33) == "medium"
        assert _level(0.65) == "medium"

    def test_low(self):
        assert _level(0.0) == "low"
        assert _level(0.32) == "low"


# ── TWF (공구 마모) ─────────────────────────────────────────────────────────────

class TestTWF:
    def _twf_level(self, tool_wear: float) -> str:
        risks = compute_partial_risks({"tool_wear": tool_wear})
        twf = next(r for r in risks if r.failure_type == "TWF")
        return twf.level

    def test_below_low(self):
        assert self._twf_level(179) == "low"

    def test_medium_boundary(self):
        assert self._twf_level(180) == "medium"

    def test_medium(self):
        assert self._twf_level(195) == "medium"

    def test_high_boundary(self):
        assert self._twf_level(200) == "high"

    def test_high(self):
        assert self._twf_level(215) == "high"

    def test_always_calculated(self):
        """tool_wear만 있어도 TWF는 계산된다."""
        risks = compute_partial_risks({"tool_wear": 200})
        assert any(r.failure_type == "TWF" for r in risks)


# ── OSF (과부하) ───────────────────────────────────────────────────────────────

class TestOSF:
    def _osf_level(self, tool_wear: float, torque: float, machine_type: str) -> str:
        risks = compute_partial_risks({"tool_wear": tool_wear, "torque": torque, "type": machine_type})
        osf = next(r for r in risks if r.failure_type == "OSF")
        return osf.level

    def test_L_below(self):
        # 100×100=10,000 < 11,000 → ratio=0.909 → osf_score=0.5 → medium
        level = self._osf_level(100, 100, "L")
        assert level == "medium"

    def test_L_over(self):
        # 150×100=15,000 > 11,000 → osf_score=0.8 → high
        assert self._osf_level(150, 100, "L") == "high"

    def test_M_boundary(self):
        # 200×60=12,000 = 12,000 → ratio=1.0 → osf_score=0.8 → high
        assert self._osf_level(200, 60, "M") == "high"

    def test_M_below(self):
        # 100×100=10,000 < 12,000 → ratio=0.833 < 0.9 → min(0.3, 0.833×0.3)=0.25 → low
        assert self._osf_level(100, 100, "M") == "low"

    def test_H_over(self):
        # 200×70=14,000 > 13,000 → osf_score=0.8 → high
        assert self._osf_level(200, 70, "H") == "high"

    def test_H_below(self):
        # 100×100=10,000 < 13,000 → ratio=0.769 → osf_score=0.3 → low
        assert self._osf_level(100, 100, "H") == "low"

    def test_unknown_type_skipped(self):
        """알 수 없는 type이면 OSF 계산 자체를 건너뛴다."""
        risks = compute_partial_risks({"tool_wear": 200, "torque": 60, "type": "X"})
        assert not any(r.failure_type == "OSF" for r in risks)

    def test_missing_torque_skipped(self):
        """torque가 없으면 OSF 계산을 건너뛴다."""
        risks = compute_partial_risks({"tool_wear": 200, "type": "M"})
        assert not any(r.failure_type == "OSF" for r in risks)


# ── HDF (열/냉각) ──────────────────────────────────────────────────────────────

class TestHDF:
    def _hdf_level(self, air: float, process: float, rpm: float) -> str:
        risks = compute_partial_risks({
            "air_temperature": air,
            "process_temperature": process,
            "rotational_speed": rpm,
        })
        hdf = next(r for r in risks if r.failure_type == "HDF")
        return hdf.level

    def test_safe(self):
        # 온도차 11K(≥8.6), rpm 1400(≥1380) → score=0.0 → low
        assert self._hdf_level(298, 309, 1400) == "low"

    def test_temp_only_risk(self):
        # 온도차 6K(<8.6), rpm 1400(ok) → score=0.5 → medium
        assert self._hdf_level(300, 306, 1400) == "medium"

    def test_rpm_only_risk(self):
        # 온도차 11K(ok), rpm 1200(<1380) → score=0.5 → medium
        assert self._hdf_level(298, 309, 1200) == "medium"

    def test_both_risk(self):
        # 온도차 5K(<8.6), rpm 1200(<1380) → score=1.0 → high
        assert self._hdf_level(300, 305, 1200) == "high"

    def test_boundary_temp(self):
        # 온도차 8.6K(경계, ≥8.6), rpm 1400 → score=0.0 → low
        assert self._hdf_level(300, 308.6, 1400) == "low"

    def test_boundary_rpm(self):
        # 온도차 11K, rpm 1380(경계, ≥1380) → score=0.0 → low
        assert self._hdf_level(298, 309, 1380) == "low"

    def test_missing_fields_skipped(self):
        """필요 feature 부족 시 HDF 건너뜀."""
        risks = compute_partial_risks({"air_temperature": 300, "process_temperature": 305})
        assert not any(r.failure_type == "HDF" for r in risks)


# ── PWF (전원/구동) ────────────────────────────────────────────────────────────

class TestPWF:
    def _pwf_level(self, rpm: float, torque: float) -> str:
        risks = compute_partial_risks({"rotational_speed": rpm, "torque": torque})
        pwf = next(r for r in risks if r.failure_type == "PWF")
        return pwf.level

    def test_in_range(self):
        # power = 40 × 1500 × 2π/60 ≈ 6,283W → 정상 → score=0.1 → low
        assert self._pwf_level(1500, 40) == "low"

    def test_below_range(self):
        # power = 10 × 100 × 2π/60 ≈ 105W → 범위 밖 → score=0.7 ≥ 0.66 → high
        assert self._pwf_level(100, 10) == "high"

    def test_above_range(self):
        # power = 100 × 2000 × 2π/60 ≈ 20,944W → 범위 밖 → score=0.7 ≥ 0.66 → high
        assert self._pwf_level(2000, 100) == "high"

    def test_missing_torque_skipped(self):
        risks = compute_partial_risks({"rotational_speed": 1500})
        assert not any(r.failure_type == "PWF" for r in risks)


# ── 복합 및 누락값 처리 ────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_feats_no_risks(self):
        assert compute_partial_risks({}) == []

    def test_combined_osf_twf_hdf(self):
        feats = {
            "tool_wear": 215, "torque": 62, "type": "M",
            "rotational_speed": 1200,
            "air_temperature": 300, "process_temperature": 305,
        }
        risks = compute_partial_risks(feats)
        types = {r.failure_type for r in risks}
        assert "OSF" in types
        assert "TWF" in types
        assert "HDF" in types

    def test_sorted_by_score_desc(self):
        """위험이 높은 고장 유형이 앞에 온다."""
        feats = {"tool_wear": 215, "torque": 62, "type": "M",
                 "rotational_speed": 1500, "air_temperature": 298, "process_temperature": 309}
        risks = compute_partial_risks(feats)
        scores = [r.score for r in risks]
        assert scores == sorted(scores, reverse=True)

    def test_non_numeric_value_skipped(self):
        """숫자로 해석 불가한 값은 해당 위험 계산에서 건너뜀."""
        risks = compute_partial_risks({"tool_wear": "unknown", "torque": 60, "type": "M"})
        assert not any(r.failure_type == "OSF" for r in risks)
        assert not any(r.failure_type == "TWF" for r in risks)

    def test_partial_feats_partial_risks(self):
        """torque만 있으면 TWF/OSF/HDF 계산 불가 — 빈 리스트 반환."""
        risks = compute_partial_risks({"torque": 60})
        assert risks == []

    def test_risk_has_formula_and_rule(self):
        """각 FailureRisk에 formula, rule 필드가 채워져야 한다."""
        risks = compute_partial_risks({"tool_wear": 200})
        twf = next(r for r in risks if r.failure_type == "TWF")
        assert twf.formula
        assert twf.rule
