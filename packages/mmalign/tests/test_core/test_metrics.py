"""Tests for mmalign.core.metrics — MetricSpec and METRIC_SPECS."""

import pytest
from mmalign.core.metrics import METRIC_SPECS, MetricSpec


class TestMetricSpec:
    def test_gte_met(self):
        spec = MetricSpec(
            id="x",
            name="x",
            description="",
            unit="%",
            direction="gte",
            target_mid=85.0,
            target_final=90.0,
        )
        assert spec.is_met(85.0) is True
        assert spec.is_met(84.9) is False
        assert spec.is_met(90.0, checkpoint="final") is True
        assert spec.is_met(89.9, checkpoint="final") is False

    def test_lte_met(self):
        spec = MetricSpec(
            id="y",
            name="y",
            description="",
            unit="%",
            direction="lte",
            target_mid=5.0,
            target_final=2.0,
        )
        assert spec.is_met(5.0) is True
        assert spec.is_met(5.1) is False
        assert spec.is_met(2.0, checkpoint="final") is True
        assert spec.is_met(2.1, checkpoint="final") is False


class TestMetricSpecCatalog:
    def test_all_required_keys_present(self):
        required = {
            "1.1", "1.2", "1.3",
            "1.4_algorithms", "1.4_stars",
            "1.5_scenarios", "1.5_samples", "1.5_models",
            "1.6", "1.7", "1.8", "1.9",
        }
        assert required.issubset(set(METRIC_SPECS.keys()))

    def test_all_metrics_have_source_components(self):
        # Every metric (except GitHub Stars) should have at least one component
        for key, spec in METRIC_SPECS.items():
            if key == "1.4_stars":
                continue
            assert len(spec.source_components) > 0, f"{key} missing source_components"

    def test_metric_1_1_targets(self):
        m = METRIC_SPECS["1.1"]
        assert m.direction == "gte"
        assert m.target_mid == pytest.approx(85.0)
        assert m.target_final == pytest.approx(90.0)

    def test_metric_1_2_lte(self):
        m = METRIC_SPECS["1.2"]
        assert m.direction == "lte"
        assert m.target_mid == pytest.approx(5.0)

    def test_serialisation_round_trip(self):
        for spec in METRIC_SPECS.values():
            restored = MetricSpec.model_validate(spec.model_dump())
            assert restored.id == spec.id
