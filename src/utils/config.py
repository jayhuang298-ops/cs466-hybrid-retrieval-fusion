"""Centralized config loader — reads base.yaml, datasets.yaml, fusion.yaml."""
import os
from pathlib import Path
import yaml


def _find_project_root() -> Path:
    """Walk up from __file__ until we find configs/base.yaml."""
    current = Path(__file__).resolve()
    for parent in [current, *current.parents]:
        if (parent / "configs" / "base.yaml").exists():
            return parent
    raise FileNotFoundError("Cannot locate project root (no configs/base.yaml found).")


PROJECT_ROOT = _find_project_root()


def load_config() -> dict:
    """Return merged dict of base + datasets + fusion configs."""
    cfg = {}
    for name in ("base", "datasets", "fusion"):
        path = PROJECT_ROOT / "configs" / f"{name}.yaml"
        with open(path) as f:
            cfg[name] = yaml.safe_load(f)
    # Resolve all relative paths to absolute
    base = cfg["base"]
    for key, rel in base["paths"].items():
        base["paths"][key] = str(PROJECT_ROOT / rel)
    base["project_root"] = str(PROJECT_ROOT)
    return cfg


def get_dataset_path(dataset_name: str, cfg: dict | None = None) -> Path:
    if cfg is None:
        cfg = load_config()
    return Path(cfg["base"]["paths"]["data"]) / dataset_name


def get_index_path(retriever: str, dataset_name: str, cfg: dict | None = None) -> Path:
    if cfg is None:
        cfg = load_config()
    key = "indexes_bm25" if retriever == "bm25" else "indexes_dense"
    return Path(cfg["base"]["paths"][key]) / dataset_name


def get_run_path(retriever: str, dataset_name: str, cfg: dict | None = None) -> Path:
    if cfg is None:
        cfg = load_config()
    key = "runs_bm25" if retriever == "bm25" else "runs_dense"
    return Path(cfg["base"]["paths"][key]) / f"{dataset_name}.trec"


def get_fused_run_path(strategy: str, dataset_name: str, cfg: dict | None = None) -> Path:
    if cfg is None:
        cfg = load_config()
    return Path(cfg["base"]["paths"]["fused_runs"]) / strategy / f"{dataset_name}.trec"
