"""YAML-driven research pipeline orchestration with run manifests.

A *pipeline* is a directed sequence of stages (download → quality →
factors → riskmodel → backtest → train → monitoring).  Each stage:

* is a plain function ``fn(ctx, params) -> list[Path]`` registered in
  :data:`STAGES`;
* declares the stage names it ``requires``;
* reports its output files, which are hashed into a **run manifest**.

Re-running a pipeline **resumes**: a stage is skipped when its manifest
entry matches (same params, same input hashes, outputs still on disk);
``force=True`` re-runs everything.  The manifest is the audit trail:
which code version, which inputs, which outputs, how long.

Example::

    # configs/production.yaml
    universe: sp100
    factors: {horizon: 5}
    backtest: {rebalance: MS, side: long, top_n: 15}
    train: {models: lightgbm,ridge}

    # run it
    alphaforge pipeline run configs/production.yaml
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from alphaforge.config import get_config


# --------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------- #
@dataclass
class StageRecord:
    """One stage's entry in the run manifest."""

    name: str
    status: str  # 'ok' | 'skipped' | 'failed'
    seconds: float = 0.0
    params: dict = field(default_factory=dict)
    input_hash: str = ""
    outputs: list[str] = field(default_factory=list)
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "seconds": round(self.seconds, 2),
            "params": self.params,
            "input_hash": self.input_hash,
            "outputs": self.outputs,
            "error": self.error,
        }


def _hash_obj(obj: Any) -> str:
    """Stable hash of a JSON-able object or file path."""
    h = hashlib.sha256()
    if isinstance(obj, Path):
        h.update(obj.read_bytes() if obj.exists() else b"missing")
    else:
        h.update(json.dumps(obj, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


# --------------------------------------------------------------------- #
# Stage implementations (thin wrappers around the library)
# --------------------------------------------------------------------- #
def _stage_download(ctx: dict, params: dict) -> list[Path]:
    from alphaforge.data.downloader import PriceDownloader
    from alphaforge.data.universe import Universe

    uni = Universe.sp500() if params.get("universe") == "sp500" else Universe.sp100()
    dl = PriceDownloader()
    panel = dl.download(
        tickers=list(uni.tickers),
        start=params.get("start", "2015-01-01"),
        benchmark=params.get("benchmark", "SPY"),
    )
    from alphaforge.data.cache import DataRepository

    repo = DataRepository()
    out = repo.save_panel(panel, name=params.get("name", "prices"))
    if dl.report:
        ctx.setdefault("reports", []).append(repr(dl.report))
    return [out]


def _stage_factors(ctx: dict, params: dict) -> list[Path]:
    from alphaforge.data.cache import DataRepository
    from alphaforge.factors import FactorEvaluator
    from alphaforge.factors.library import FactorLibrary

    panel = DataRepository().load_panel(ctx.get("panel_name", "prices"))
    lib = FactorLibrary(panel)
    lib.compute_all()
    stats = FactorEvaluator(panel, horizon=int(params.get("horizon", 5))).evaluate_all(
        lib.compute_all()
    )
    out = get_config().results_dir / "factor_evaluation.csv"
    stats.to_csv(out, index=False)
    return [out]


def _stage_riskmodel(ctx: dict, params: dict) -> list[Path]:
    from alphaforge.data.cache import DataRepository
    from alphaforge.riskmodel import StyleRiskModel

    panel = DataRepository().load_panel(ctx.get("panel_name", "prices"))
    model = StyleRiskModel(panel)
    fr = model.factor_returns()
    out = get_config().results_dir / "riskmodel_factor_returns.csv"
    fr.to_csv(out)
    summary = get_config().results_dir / "riskmodel_summary.json"
    summary.write_text(json.dumps(model.summary(), indent=2, default=str))
    ctx["riskmodel"] = model
    return [out, summary]


def _stage_backtest(ctx: dict, params: dict) -> list[Path]:
    from alphaforge.data.cache import DataRepository
    from alphaforge.engine import run_backtest
    from alphaforge.factors import FactorEvaluator
    from alphaforge.factors.library import FactorLibrary

    panel = DataRepository().load_panel(ctx.get("panel_name", "prices"))
    lib = FactorLibrary(panel)
    lib.compute_all()
    stats = FactorEvaluator(panel, horizon=int(params.get("horizon", 5))).evaluate_all(
        lib.compute_all()
    )
    stats = stats.sort_values("ic_ir", key=lambda c: c.abs(), ascending=False)
    top = stats["name"].head(5).tolist()
    direction = {
        n: (1.0 if s > 0 else -1.0) for n, s in stats.set_index("name").loc[top, "ic_mean"].items()
    }
    signal = lib.composite(names=top, direction=direction)
    result = run_backtest(
        panel,
        signal,
        signal_name="composite_top5",
        rebalance=params.get("rebalance", "MS"),
        side=params.get("side", "long"),
        top_n=int(params.get("top_n", 15)),
        cost_bps=float(params.get("cost_bps", 10.0)),
    )
    out_dir = get_config().results_dir / "backtest_composite"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(result.summary(), indent=2, default=str))
    return [out_dir / "metrics.json"]


def _stage_train(ctx: dict, params: dict) -> list[Path]:
    from alphaforge.data.cache import DataRepository
    from alphaforge.ml.models import LightGBMAlphaModel, RidgeAlphaModel
    from alphaforge.ml.pipeline import AlphaTrainingPipeline
    from alphaforge.ml.validation import PurgedWalkForwardCV

    panel = DataRepository().load_panel(ctx.get("panel_name", "prices"))
    registry = {
        "lightgbm": lambda: LightGBMAlphaModel(),
        "ridge": lambda: RidgeAlphaModel(),
    }
    names = [m.strip() for m in str(params.get("models", "lightgbm")).split(",") if m.strip()]
    models = [registry[n]() for n in names if n in registry]
    if not models:
        raise ValueError(f"no known models in {names} (have {list(registry)})")
    pipe = AlphaTrainingPipeline(
        panel, models=models, cv=PurgedWalkForwardCV(n_splits=int(params.get("splits", 6)))
    )
    result = pipe.run(verbose=bool(params.get("verbose", False)))
    out = result.save(get_config().results_dir / "ml_run")
    return [out / "oos_predictions.parquet", out / "ic_summary.csv"]


def _stage_monitoring(ctx: dict, params: dict) -> list[Path]:
    from alphaforge.ml.monitoring import monitoring_report

    ml_dir = get_config().results_dir / "ml_run"
    preds = pd.read_parquet(ml_dir / "oos_predictions.parquet")
    preds.get("label")
    reports = {}
    for col in preds.columns:
        if col in ("label", "ensemble_causal"):
            continue
        rep = monitoring_report(preds[col], model=col)
        reports[col] = rep.as_dict()
    if "ensemble_causal" in preds.columns:
        rep = monitoring_report(preds["ensemble_causal"], model="ensemble_causal")
        reports["ensemble_causal"] = rep.as_dict()
    out = get_config().results_dir / "monitoring" / "report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, indent=2, default=str))
    return [out]


STAGES: dict[str, dict] = {
    "download": {"fn": _stage_download, "requires": []},
    "factors": {"fn": _stage_factors, "requires": ["download"]},
    "riskmodel": {"fn": _stage_riskmodel, "requires": ["download"]},
    "backtest": {"fn": _stage_backtest, "requires": ["factors"]},
    "train": {"fn": _stage_train, "requires": ["download"]},
    "monitoring": {"fn": _stage_monitoring, "requires": ["train"]},
}


class PipelineRunner:
    """Executes (or resumes) a pipeline described by a YAML/dict config."""

    def __init__(self, config: dict | str | Path, run_dir: str | Path | None = None) -> None:
        if isinstance(config, (str, Path)) and Path(config).suffix in (".yaml", ".yml"):
            import yaml

            config = yaml.safe_load(Path(config).read_text())
        if not isinstance(config, dict):
            raise TypeError("config must be a dict or a YAML file path")
        self.config: dict = config
        base = get_config().results_dir.parent
        self.run_dir = Path(run_dir) if run_dir else base / ".runs" / _run_id()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.run_dir / "manifest.json"
        self.ctx: dict = {"panel_name": config.get("panel_name", "prices")}

    # ------------------------------------------------------------------ #
    def run(self, force: bool = False) -> dict:
        """Execute all stages in dependency order, resuming where possible."""
        previous = self._load_previous() if not force else {}
        records: dict[str, dict] = {}
        for name in self._order():
            spec = STAGES[name]
            params = self.config.get(name, {})
            if params is False:  # `stage: false` disables a stage
                continue
            missing = [r for r in spec["requires"] if r not in records]
            if missing:
                raise RuntimeError(f"stage {name!r} requires {missing} but they were disabled")
            inputs = {r: _hash_obj(records[r]["outputs"]) for r in spec["requires"]}
            input_hash = _hash_obj({"params": params, "inputs": inputs, "stage": name})

            cached = previous.get(name)
            if (
                cached
                and cached.get("input_hash") == input_hash
                and all(Path(f).exists() for f in cached.get("outputs", []))
            ):
                records[name] = {**cached, "status": "skipped"}
                print(f"  [skip] {name} (manifest match)")
                continue

            t0 = time.time()
            try:
                outputs = [str(p) for p in spec["fn"](self.ctx, params)]
                rec = StageRecord(
                    name=name,
                    status="ok",
                    seconds=time.time() - t0,
                    params=params,
                    input_hash=input_hash,
                    outputs=outputs,
                )
                records[name] = rec.as_dict()
                print(f"  [ok]   {name} ({rec.seconds:.1f}s)")
            except Exception as exc:  # noqa: BLE001
                rec = StageRecord(
                    name=name,
                    status="failed",
                    seconds=time.time() - t0,
                    params=params,
                    input_hash=input_hash,
                    error=f"{type(exc).__name__}: {exc}",
                )
                records[name] = rec.as_dict()
                self._write_manifest(records)
                raise
            self._write_manifest(records)
        return records

    # ------------------------------------------------------------------ #
    def _order(self) -> list[str]:
        """Topological order of the enabled stages."""
        enabled: list[str] = []
        for name in STAGES:
            if (name == "download" or name in self.config) and self.config.get(name) is not False:
                enabled.append(name)
        seen: list[str] = []
        pending = list(enabled)
        while pending:
            progressed = False
            for name in list(pending):
                if all(r in seen for r in STAGES[name]["requires"]):
                    seen.append(name)
                    pending.remove(name)
                    progressed = True
            if not progressed:
                raise RuntimeError(f"circular or missing dependencies among {pending}")
        return seen

    def _load_previous(self) -> dict:
        if self.manifest_path.exists():
            try:
                return json.loads(self.manifest_path.read_text()).get("stages", {})
            except json.JSONDecodeError:
                return {}
        return {}

    def _write_manifest(self, records: dict) -> None:
        self.manifest_path.write_text(
            json.dumps(
                {
                    "config": self.config,
                    "stages": records,
                    "n_stages": len(records),
                    "written": pd.Timestamp.utcnow().isoformat(),
                },
                indent=2,
                default=str,
            )
        )


def _run_id() -> str:
    return pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
