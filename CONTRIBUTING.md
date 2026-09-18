# Contributing to AlphaForge

Thanks for your interest! This project welcomes contributions of all sizes.

## Development setup

```bash
git clone https://github.com/CathyKernel/alphaforge.git
cd alphaforge
pip install -e .[dev,app,dl]   # dev tools + dashboard + PyTorch
make test                       # 61 tests, ~12 s
make lint                       # ruff check + format
```

## Ground rules

1. **No look-ahead bias.** Any signal computed at date T may only use data through T's close; execution assumptions belong to the engine, not to factors. New factors are expected to pass the no-lookahead test pattern used in `tests/test_factors.py`.
2. **No silent data mutation.** Quality checks report; they do not "fix" data. Filters belong to the strategy layer and must be documented.
3. **Vectorise.** Anything on the per-date hot path should be a pandas/NumPy axis operation, not a Python loop. If you must loop, say why in a comment.
4. **Numbers are net.** Any reported backtest statistic should be net of the documented cost model, or explicitly labelled gross.
5. **Tests for mechanics.** Anything that touches money math (turnover, costs, drift, timing) needs a unit test with a hand-computable expected value.

## Adding a factor

```python
from alphaforge.factors import Factor, register


@register
class MyFactor(Factor):
    name = "my_factor"
    category = "momentum"  # or a new category
    description = "one-line intuition"

    def compute(self, panel):
        return panel.close / panel.close.rolling(63).mean() - 1
```

Run `pytest tests/test_factors.py` and add a correctness + no-lookahead test for your factor.

## Pull requests

- Keep PRs focused; one feature or fix per PR.
- Run `make lint test` before submitting (CI runs the same).
- New public APIs need docstrings and, where money math is involved, tests.

## Reporting issues

Include: OS, Python version, package version (`python -c "import alphaforge; print(alphaforge.__version__)"`), a minimal reproduction snippet, and (for data issues) the dataset manifest from `results/`.
