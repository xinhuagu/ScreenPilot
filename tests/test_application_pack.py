"""Tests for ApplicationPack, ModelRegistry, and AppRouter."""

import tempfile
from pathlib import Path

import yaml

from gazefy.core.app_router import AppRouter
from gazefy.core.application_pack import ApplicationPack, PackMetadata
from gazefy.core.model_registry import ModelRegistry


def _create_pack_dir(tmp: Path, name: str, window_match: list[str] | None = None) -> Path:
    """Helper: create a minimal pack directory with pack.yaml."""
    pack_dir = tmp / name
    pack_dir.mkdir()
    meta = {
        "name": name,
        "version": "1.0.0",
        "labels": ["button", "menu_item", "input_field"],
        "window_match": window_match or [],
        "input_size": 1024,
    }
    with open(pack_dir / "pack.yaml", "w") as f:
        yaml.dump(meta, f)
    return pack_dir


# --- PackMetadata ---


def test_pack_metadata_from_dict():
    m = PackMetadata.from_dict({"name": "test", "version": "2.0", "labels": ["a", "b"]})
    assert m.name == "test"
    assert m.version == "2.0"
    assert m.labels == ["a", "b"]
    assert m.conf_threshold == 0.5  # default


def test_pack_metadata_ignores_unknown_fields():
    m = PackMetadata.from_dict({"name": "x", "unknown_field": 42})
    assert m.name == "x"


# --- ApplicationPack ---


def test_load_pack():
    with tempfile.TemporaryDirectory() as tmp:
        pack_dir = _create_pack_dir(Path(tmp), "my_app")
        pack = ApplicationPack.load(pack_dir)
        assert pack.metadata.name == "my_app"
        assert pack.metadata.labels == ["button", "menu_item", "input_field"]
        assert pack.label_map == {0: "button", 1: "menu_item", 2: "input_field"}


def test_pack_no_manifest_raises():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            ApplicationPack.load(tmp)
            assert False, "Should have raised"
        except FileNotFoundError:
            pass


def test_pack_matches_window():
    with tempfile.TemporaryDirectory() as tmp:
        pack_dir = _create_pack_dir(Path(tmp), "erp", window_match=["SAP", "ERP Client"])
        pack = ApplicationPack.load(pack_dir)
        assert pack.matches_window("SAP Logon 770")
        assert pack.matches_window("My ERP Client v2")
        assert not pack.matches_window("Notepad")


def test_pack_has_model_false_when_no_file():
    with tempfile.TemporaryDirectory() as tmp:
        pack_dir = _create_pack_dir(Path(tmp), "test")
        pack = ApplicationPack.load(pack_dir)
        assert not pack.has_model


# --- ModelRegistry ---


def test_registry_scan_empty_dir():
    with tempfile.TemporaryDirectory() as tmp:
        reg = ModelRegistry(tmp)
        assert reg.scan() == 0


def test_registry_scan_finds_packs():
    with tempfile.TemporaryDirectory() as tmp:
        _create_pack_dir(Path(tmp), "app_a", window_match=["AppA"])
        _create_pack_dir(Path(tmp), "app_b", window_match=["AppB"])
        reg = ModelRegistry(tmp)
        assert reg.scan() == 2
        assert reg.get("app_a") is not None
        assert reg.get("app_b") is not None
        assert reg.get("nonexistent") is None


def test_registry_find_for_window():
    with tempfile.TemporaryDirectory() as tmp:
        _create_pack_dir(Path(tmp), "erp", window_match=["SAP"])
        _create_pack_dir(Path(tmp), "crm", window_match=["Salesforce"])
        reg = ModelRegistry(tmp)
        reg.scan()
        assert reg.find_for_window("SAP Logon").metadata.name == "erp"
        assert reg.find_for_window("Salesforce CRM").metadata.name == "crm"
        assert reg.find_for_window("Notepad") is None


# --- AppRouter ---


def test_router_routes_to_pack():
    with tempfile.TemporaryDirectory() as tmp:
        _create_pack_dir(Path(tmp), "erp", window_match=["SAP"])
        reg = ModelRegistry(tmp)
        reg.scan()
        router = AppRouter(reg)

        pack = router.route("SAP Logon 770")
        assert pack is not None
        assert pack.metadata.name == "erp"
        assert router.active_pack is pack


def test_router_deactivates_on_no_match():
    with tempfile.TemporaryDirectory() as tmp:
        _create_pack_dir(Path(tmp), "erp", window_match=["SAP"])
        reg = ModelRegistry(tmp)
        reg.scan()
        router = AppRouter(reg)

        router.route("SAP Logon")
        assert router.active_pack is not None

        router.route("Unknown App")
        assert router.active_pack is None


def test_router_force_pack():
    with tempfile.TemporaryDirectory() as tmp:
        _create_pack_dir(Path(tmp), "erp", window_match=["SAP"])
        reg = ModelRegistry(tmp)
        reg.scan()
        router = AppRouter(reg)

        pack = router.force_pack("erp")
        assert pack is not None
        assert router.active_pack.metadata.name == "erp"

        assert router.force_pack("nonexistent") is None
