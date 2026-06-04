"""Tests for mmalign.core.registry — Registry[T] with built-in protection."""

import pytest
from mmalign.core.registry import BuiltinOverrideError, NotRegisteredError, Registry


class TestRegistry:
    def _make_reg(self) -> Registry[str]:
        return Registry[str]("test")

    def test_register_and_get(self):
        reg = self._make_reg()
        reg.register("foo", "bar")
        assert reg.get("foo") == "bar"

    def test_get_missing_raises(self):
        reg = self._make_reg()
        with pytest.raises(NotRegisteredError, match="not registered"):
            reg.get("missing")

    def test_get_or_none_missing(self):
        reg = self._make_reg()
        assert reg.get_or_none("x") is None

    def test_contains(self):
        reg = self._make_reg()
        reg.register("a", "1")
        assert "a" in reg
        assert "b" not in reg

    def test_list_sorted(self):
        reg = self._make_reg()
        reg.register("b", "2")
        reg.register("a", "1")
        assert reg.list() == ["a", "b"]

    def test_count(self):
        reg = self._make_reg()
        assert reg.count() == 0
        reg.register("x", "1")
        assert reg.count() == 1

    def test_iter(self):
        reg = self._make_reg()
        reg.register("k", "v")
        pairs = list(reg)
        assert ("k", "v") in pairs


class TestBuiltinProtection:
    def test_builtin_cannot_be_overwritten(self):
        reg: Registry[str] = Registry("methods")
        reg.register("align3", "impl_v1", builtin=True)
        with pytest.raises(BuiltinOverrideError, match="built-in"):
            reg.register("align3", "impl_v2")

    def test_non_builtin_can_be_overwritten(self):
        reg: Registry[str] = Registry("methods")
        reg.register("custom", "v1")
        reg.register("custom", "v2")   # should not raise
        assert reg.get("custom") == "v2"

    def test_builtin_appears_in_list_builtins(self):
        reg: Registry[str] = Registry("methods")
        reg.register("vanilla", "impl", builtin=True)
        reg.register("custom", "impl2")
        assert "vanilla" in reg.list_builtins()
        assert "custom" not in reg.list_builtins()

    def test_builtin_cannot_be_unregistered(self):
        reg: Registry[str] = Registry("methods")
        reg.register("raw", "impl", builtin=True)
        with pytest.raises(BuiltinOverrideError):
            reg.unregister("raw")

    def test_non_builtin_can_be_unregistered(self):
        reg: Registry[str] = Registry("methods")
        reg.register("temp", "v")
        reg.unregister("temp")
        assert "temp" not in reg

    def test_plugin_namespace_does_not_conflict_with_builtin(self):
        reg: Registry[str] = Registry("methods")
        reg.register("align3", "native_impl", builtin=True)
        # plugin uses namespace prefix, so this must succeed
        reg.register("specbench.align3", "specbench_impl")
        assert reg.get("specbench.align3") == "specbench_impl"
        assert reg.get("align3") == "native_impl"
