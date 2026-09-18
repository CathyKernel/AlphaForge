"""AlphaForge command-line interface.

Examples
--------
    alphaforge download --universe sp100 --start 2015-01-01
    alphaforge universe --refresh
    alphaforge fundamentals
    alphaforge factors --horizon 5
    alphaforge backtest --composite top5 --rebalance MS --side long
    alphaforge train --models lightgbm,lstm,ridge --splits 6
    alphaforge pipeline run configs/production.yaml
    alphaforge dashboard
"""

from __future__ import annotations

import argparse
import sys


def _cmd_download(args: argparse.Namespace) -> int:
    from alphaforge.data.cache import DataRepository
    from alphaforge.data.quality import DataQualityChecker
    from alphaforge.data.universe import Universe

    universe = Universe.sp500(refresh=args.refresh_universe)
    if args.universe == "sp100":
        universe = Universe.sp100()
    print(f"universe: {len(universe)} tickers (source: {universe.source})")
    repo = DataRepository()
    panel, report = repo.fetch_and_cache(
        universe, start=args.start, end=args.end, name=args.name, benchmark=args.benchmark or None
    )
    print(
        f"downloaded {report.succeeded}/{report.requested} tickers, "
        f"{report.rows:,} rows ({report.start} .. {report.end})"
    )
    if report.failed_tickers:
        print(f"failed: {report.failed_tickers}")
    if args.quality:
        qr = DataQualityChecker().run(panel)
        print(qr.summary().to_string(index=False))
    return 0


def _cmd_universe(args: argparse.Namespace) -> int:
    from alphaforge.data.universe import Universe

    u = Universe.sp500(refresh=args.refresh)
    print(f"S&P 500 universe: {len(u)} tickers (snapshot {u.snapshot_date}, source {u.source})")
    print(u.sector_counts().to_string())
    sp100 = Universe.sp100()
    print(f"\nS&P 100 subset: {len(sp100)} tickers")
    return 0


def _cmd_fundamentals(args: argparse.Namespace) -> int:
    from alphaforge.data.cache import DataRepository
    from alphaforge.data.fundamentals import FundamentalsRepository
    from alphaforge.data.universe import Universe

    uni = Universe.sp500() if args.universe == "sp500" else Universe.sp100()
    # restrict to tickers present in the bundled price panel when available
    try:
        panel = DataRepository().load_panel(args.panel)
        names = [t for t in uni.tickers if t in panel.tickers]
    except FileNotFoundError:
        names = list(uni.tickers)
    print(f"fetching quarterly fundamentals for {len(names)} tickers ...")
    repo = FundamentalsRepository()
    df, path = repo.refresh(names, max_workers=args.workers)
    print(f"snapshot: {len(df)} tickers x {df.shape[1]} metrics -> {path}")
    print(f"date: {repo.snapshot_date()} (static cross-section; look-ahead disclosed)")
    if df.empty:
        print("no fundamentals fetched — check network access")
        return 1
    print(df.head(8).to_string())
    return 0


def _cmd_pipeline(args: argparse.Namespace) -> int:
    from alphaforge.pipeline import PipelineRunner

    runner = PipelineRunner(args.config)
    print(f"pipeline run dir: {runner.run_dir}")
    records = runner.run(force=args.force)
    ok = sum(1 for r in records.values() if r["status"] in ("ok", "skipped"))
    print(f"\nstages: {ok}/{len(records)} ok; manifest at {runner.manifest_path}")
    return 0


def _cmd_factors(args: argparse.Namespace) -> int:
    from alphaforge.config import get_config
    from alphaforge.data.cache import DataRepository
    from alphaforge.factors import FactorEvaluator
    from alphaforge.factors.library import FactorLibrary

    panel = DataRepository().load_panel()
    lib = FactorLibrary(panel)
    evaluator = FactorEvaluator(panel, horizon=args.horizon)
    stats = evaluator.evaluate_all(lib.compute_all())
    print(stats.round(4).to_string(index=False))
    out = get_config().results_dir / "factor_evaluation.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(out, index=False)
    print(f"\nsaved: {out}")
    return 0


def _cmd_backtest(args: argparse.Namespace) -> int:
    from alphaforge.data.cache import DataRepository
    from alphaforge.engine import run_backtest
    from alphaforge.factors import FactorEvaluator
    from alphaforge.factors.library import FactorLibrary
    from alphaforge.reporting.tearsheet import tearsheet

    panel = DataRepository().load_panel()
    lib = FactorLibrary(panel)

    if args.factor:
        signal = lib.compute(args.factor)
        name = args.factor
    elif args.composite == "top5":
        stats = FactorEvaluator(panel, horizon=args.horizon).evaluate_all(lib.compute_all())
        ranked = stats.sort_values("ic_ir", key=lambda c: c.abs(), ascending=False)
        top = ranked["name"].head(5).tolist()
        direction = {
            n: (1.0 if s.ic_mean > 0 else -1.0)
            for n, s in ranked.set_index("name").loc[top].iterrows()
        }
        signal = lib.composite(names=top, direction=direction)
        name = "composite_top5"
    else:
        signal = lib.composite()
        name = "composite_all"

    result = run_backtest(
        panel,
        signal,
        signal_name=name,
        rebalance=args.rebalance,
        top_n=args.top_n,
        side=args.side,
        weighting=args.weighting,
        execution_lag=args.lag,
        cost_bps=args.cost_bps,
        vol_target=args.vol_target,
    )
    out_dir = _results_path() / f"backtest_{name}"
    metrics = tearsheet(result, out_dir, name=name)
    s = metrics["performance"]
    print(
        f"\n{name}: CAGR {s['cagr']:+.2%}  Sharpe {s['sharpe']:.2f}  "
        f"MaxDD {s['max_drawdown']:.1%}  IR "
        f"{s.get('information_ratio', float('nan')):+.2f}"
    )
    print(f"report written to {out_dir}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from alphaforge.data.cache import DataRepository
    from alphaforge.factors.library import FactorLibrary
    from alphaforge.ml import (
        AlphaTrainingPipeline,
        FeatureConfig,
        FeaturePanelBuilder,
        LightGBMAlphaModel,
        PurgedWalkForwardCV,
        lstm_available,
    )

    panel = DataRepository().load_panel()
    lib = FactorLibrary(panel)
    models = []
    model_names = [m.strip().lower() for m in args.models.split(",")]
    if "lightgbm" in model_names:
        models.append(LightGBMAlphaModel())
    if "ridge" in model_names:
        from alphaforge.ml import RidgeAlphaModel

        models.append(RidgeAlphaModel())
    if "lstm" in model_names:
        if not lstm_available():
            print("PyTorch not installed -- skipping LSTM (pip install alphaforge[dl])")
        else:
            from alphaforge.ml import LSTMAlphaModel

            models.append(LSTMAlphaModel(max_epochs=args.lstm_epochs))
    if not models:
        print("no models selected")  # noqa: T201
        return 1

    cv = PurgedWalkForwardCV(n_splits=args.splits, purge=args.horizon + 5, embargo=5)
    fb = FeaturePanelBuilder(panel, config=FeatureConfig(label_horizon=args.horizon))
    pipeline = AlphaTrainingPipeline(panel, models, cv=cv, feature_builder=fb)
    print(f"training {len(models)} model(s) over {args.splits} folds ...")
    result = pipeline.run(factor_values=lib.compute_all())
    out = result.save(_results_path() / "ml_run")
    print()
    print(result.ic_summary.round(4).to_string())
    print(f"\nartifacts saved to {out}")
    return 0


def _cmd_dashboard(args: argparse.Namespace) -> int:
    import subprocess

    from alphaforge.config import get_config

    root = get_config().data_dir.parent
    server = root / "app" / "server.py"
    if not server.exists():
        print(f"dashboard server not found at {server}")  # noqa: T201
        return 1
    port = getattr(args, "port", None) or 8501
    subprocess.run(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.server:app",
            "--app-dir",
            str(root),
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
        ]
    )
    return 0


def _results_path():
    from alphaforge.config import get_config

    return get_config().results_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="alphaforge", description="Quantitative research platform for US equities"
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="download and cache price data")
    p.add_argument("--universe", choices=["sp100", "sp500"], default="sp100")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--name", default="prices")
    p.add_argument("--benchmark", default="SPY")
    p.add_argument("--refresh-universe", action="store_true")
    p.add_argument("--quality", action="store_true", default=True)
    p.set_defaults(func=_cmd_download)

    p = sub.add_parser("universe", help="show universe info")
    p.add_argument("--refresh", action="store_true")
    p.set_defaults(func=_cmd_universe)

    p = sub.add_parser("factors", help="evaluate the factor library")
    p.add_argument("--horizon", type=int, default=5)
    p.set_defaults(func=_cmd_factors)

    p = sub.add_parser("backtest", help="backtest a factor or composite")
    p.add_argument("--factor", default=None, help="single factor name")
    p.add_argument("--composite", choices=["all", "top5"], default="top5")
    p.add_argument("--rebalance", default="W-FRI")
    p.add_argument("--top-n", dest="top_n", type=int, default=15)
    p.add_argument("--side", choices=["long", "short", "long_short"], default="long_short")
    p.add_argument("--weighting", default="equal")
    p.add_argument("--lag", dest="lag", type=int, default=1)
    p.add_argument("--cost-bps", dest="cost_bps", type=float, default=10.0)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--vol-target", dest="vol_target", type=float, default=None)
    p.set_defaults(func=_cmd_backtest)

    p = sub.add_parser("train", help="train ML alpha models (walk-forward)")
    p.add_argument("--models", default="lightgbm", help="comma-separated: lightgbm,lstm,ridge")
    p.add_argument("--splits", type=int, default=6)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--lstm-epochs", dest="lstm_epochs", type=int, default=25)
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser(
        "fundamentals", help="fetch the quarterly-fundamentals snapshot (value factors)"
    )
    p.add_argument("--universe", choices=["sp100", "sp500"], default="sp100")
    p.add_argument("--panel", default="prices", help="restrict to tickers in this cached panel")
    p.add_argument("--workers", type=int, default=8, help="parallel fetch workers")
    p.set_defaults(func=_cmd_fundamentals)

    p = sub.add_parser("pipeline", help="run a YAML research pipeline (resumable)")
    sub_pipe = p.add_subparsers(dest="pipeline_cmd", required=True)
    run = sub_pipe.add_parser("run", help="execute (or resume) a pipeline config")
    run.add_argument("config", help="path to a YAML pipeline config")
    run.add_argument("--force", action="store_true", help="re-run every stage")
    run.set_defaults(func=_cmd_pipeline)

    p = sub.add_parser("dashboard", help="launch the web dashboard (FastAPI + uvicorn)")
    p.add_argument("--port", type=int, default=8501, help="port to serve on (default 8501)")
    p.set_defaults(func=_cmd_dashboard)

    args = ap.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
