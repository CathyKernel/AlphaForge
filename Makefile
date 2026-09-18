.PHONY: install dev lint format typecheck test coverage dashboard download backtest train clean

install:            ## install the package
        pip install -e .

dev:               ## install with dev + app + dl extras
        pip install -e .[dev,app,dl]

lint:              ## ruff lint
        ruff check src tests scripts app

format:            ## ruff autoformat
        ruff format src tests scripts app
        ruff check --fix src tests scripts app

typecheck:         ## mypy (advisory)
        mypy

test:              ## run the test suite
        pytest -q

coverage:          ## tests with coverage report
        pytest --cov=alphaforge --cov-report=term-missing -q

download:          ## refresh the bundled S&P 100 dataset
        python scripts/download_data.py --universe sp100 --start 2015-01-01

backtest:          ## run the flagship composite backtest
        python scripts/run_backtest.py --composite top5 --rebalance MS

train:             ## train ML models walk-forward (lightgbm + lstm)
        python scripts/train_models.py --models lightgbm,lstm

dashboard:         ## launch the web dashboard (FastAPI + uvicorn)
        uvicorn app.server:app --host 0.0.0.0 --port 8501

clean:             ## remove caches and build artifacts
        rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml \
                htmlcov build dist *.egg-info
        find . -name __pycache__ -type d -exec rm -rf {} +
