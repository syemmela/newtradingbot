"""Validation framework: parameter-robustness sweeps, out-of-sample testing,
and walk-forward analysis. Built to make it structurally harder to mistake
an in-sample-only backtest result for a validated one.

This module never edits strategy source files. It works by temporarily
overriding a strategy module's constants (e.g. trend_following.FAST_PERIOD)
or a symbol's config.INSTRUMENTS params (e.g. z_entry, trend_adx_min) via a
context manager, running the backtest simulator on the affected data, then
restoring the original value. This formalizes the same monkey-patch sweep
pattern used ad-hoc while tuning trend_following's EMA periods and ADX
floors (see git history) into something reusable instead of throwaway
scripts.

Core discipline enforced here (the actual point of this module):
- A parameter is "robust" if most of its neighborhood also performs, not
  just the single best value (a spike surrounded by losers is fragile).
- Parameters are only ever selected using in-sample data. Out-of-sample
  data is never touched until AFTER a choice is locked in, and is the
  only number that counts as a real performance estimate.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

import pandas as pd

import config
from bot.backtest import (
    MIN_TRADES_FOR_VALIDATION,
    STRATEGY_MODULES,
    BacktestResult,
    _compute_stats,
    _empty_result,
    _fetch_prepared_bars,
    _run_backtest_from_bars,
)
from bot.types import TradeRecord


@contextmanager
def override_module_attr(module, **attrs):
    """Temporarily set module-level constants (e.g. trend_following.FAST_PERIOD),
    restoring the originals on exit even if the block raises."""
    originals = {name: getattr(module, name) for name in attrs}
    for name, value in attrs.items():
        setattr(module, name, value)
    try:
        yield
    finally:
        for name, value in originals.items():
            setattr(module, name, value)


@contextmanager
def override_symbol_param(symbol: str, **params):
    """Temporarily set config.INSTRUMENTS[symbol]["params"] entries (e.g.
    z_entry, trend_adx_min), restoring the originals on exit."""
    target = config.INSTRUMENTS[symbol]["params"]
    originals = {name: target.get(name) for name in params}
    target.update(params)
    try:
        yield
    finally:
        for name, value in originals.items():
            if value is None:
                target.pop(name, None)
            else:
                target[name] = value


@contextmanager
def override_symbol_strategy(symbol: str, strategy_name: str):
    """Temporarily reassign config.INSTRUMENTS[symbol]["strategy"] -- for
    backtesting a CANDIDATE strategy against a symbol that's still
    actively assigned to a different one in the live config.

    This is required, not optional, for every entry point below:
    _simulate() (called by _run_backtest_from_bars, which every sweep/
    out-of-sample/walk-forward helper in this module goes through) looks
    up config.INSTRUMENTS[symbol]["strategy"] ITSELF rather than trusting
    the strategy_name argument passed into run_backtest() -- so without
    this override, evaluate() from the symbol's currently-CONFIGURED
    strategy runs against bars prepared for a different one entirely
    (e.g. mean_reversion.evaluate() crashing on a KeyError for "zscore"
    against bars that trend_pullback.compute_indicators() produced,
    which don't have that column). All the sweep/apply helpers below
    wrap their _run_backtest_from_bars calls with this internally, via
    _run_backtest_for_strategy(), specifically so callers can't forget.
    Restores the original assignment on exit even if the block raises."""
    original = config.INSTRUMENTS[symbol]["strategy"]
    config.INSTRUMENTS[symbol]["strategy"] = strategy_name
    try:
        yield
    finally:
        config.INSTRUMENTS[symbol]["strategy"] = original


async def _run_backtest_for_strategy(bars: pd.DataFrame, strategy_name: str, symbol: str) -> BacktestResult:
    """The one place every sweep/out-of-sample/walk-forward helper below
    calls into _run_backtest_from_bars -- always under
    override_symbol_strategy, so this module works correctly for a
    candidate strategy regardless of what the symbol is currently
    configured to run live."""
    with override_symbol_strategy(symbol, strategy_name):
        return await _run_backtest_from_bars(bars, strategy_name, symbol)


def split_out_of_sample(bars: pd.DataFrame, in_sample_frac: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split: everything before the split point is
    'in-sample' (fair game for choosing parameters); everything after is
    'out-of-sample' and must never be touched while choosing parameters —
    only used afterward to report how the chosen parameters actually
    perform on data they've never seen."""
    split_idx = int(len(bars) * in_sample_frac)
    return bars.iloc[:split_idx], bars.iloc[split_idx:]


def _raw_ohlcv(bars: pd.DataFrame) -> pd.DataFrame:
    """compute_indicators() always adds columns without dropping the
    original OHLCV ones, so this recovers the pre-indicator data needed to
    recompute indicators fresh under a different period/window parameter."""
    return bars[["open", "high", "low", "close", "volume"]]


@dataclass
class SweepPoint:
    params: dict
    result: BacktestResult


@dataclass
class RobustnessReport:
    param_name: str = ""
    points: list[SweepPoint] = field(default_factory=list)

    @property
    def positive_fraction(self) -> float:
        """Fraction of tested values with positive Sharpe. The core
        robustness signal: a setting with one great value and mostly-bad
        neighbors is fragile, not validated, even if that one value backtests
        beautifully."""
        if not self.points:
            return 0.0
        return sum(1 for p in self.points if p.result.sharpe > 0) / len(self.points)

    @property
    def best(self) -> SweepPoint | None:
        return max(self.points, key=lambda p: p.result.sharpe) if self.points else None

    @property
    def worst(self) -> SweepPoint | None:
        return min(self.points, key=lambda p: p.result.sharpe) if self.points else None

    def is_robust(self, min_positive_fraction: float = 0.6) -> bool:
        return self.positive_fraction >= min_positive_fraction

    def print_table(self) -> None:
        print(f"  {self.param_name:>16s}   trades  win%    PF  sharpe   maxDD  return")
        for p in self.points:
            r = p.result
            value = next(iter(p.params.values()))
            pf = "inf" if r.profit_factor == float("inf") else f"{r.profit_factor:.2f}"
            print(f"  {value!r:>16}   {r.trades:6d}  {r.win_rate:4.0f}%  {pf:>4s}  {r.sharpe:6.2f}  {r.max_dd:5.1f}%  {r.total_return_pct:+6.1f}%")
        marker = "ROBUST" if self.is_robust() else "NOT ROBUST"
        print(f"  -> {self.positive_fraction*100:.0f}% of values positive Sharpe -- {marker}")


async def sweep_module_param(
    bars: pd.DataFrame, strategy_name: str, symbol: str, module_attr: str, values: list
) -> RobustnessReport:
    """Sweep a strategy module constant that affects compute_indicators()
    (e.g. FAST_PERIOD, PERIOD, VOLUME_CONFIRM_MULT) -- indicators are
    recomputed fresh for each value since they depend on it."""
    module = STRATEGY_MODULES[strategy_name]
    report = RobustnessReport(param_name=module_attr)
    if bars.empty:
        return report
    raw = _raw_ohlcv(bars)
    for value in values:
        with override_module_attr(module, **{module_attr: value}):
            recomputed = module.compute_indicators(raw, symbol)
            result = await _run_backtest_for_strategy(recomputed, strategy_name, symbol)
        report.points.append(SweepPoint(params={module_attr: value}, result=result))
    return report


async def sweep_symbol_param(
    bars: pd.DataFrame, strategy_name: str, symbol: str, param_name: str, values: list
) -> RobustnessReport:
    """Sweep a per-symbol config param read at evaluate()-time (e.g.
    z_entry, trend_adx_min) -- no indicator recomputation needed since
    these don't affect compute_indicators(), only evaluate()'s thresholds."""
    report = RobustnessReport(param_name=param_name)
    if bars.empty:
        return report
    for value in values:
        with override_symbol_param(symbol, **{param_name: value}):
            result = await _run_backtest_for_strategy(bars, strategy_name, symbol)
        report.points.append(SweepPoint(params={param_name: value}, result=result))
    return report


async def sweep_config_attr(
    bars: pd.DataFrame, strategy_name: str, symbol: str, config_attr: str, values: list
) -> RobustnessReport:
    """Sweep a GLOBAL config constant shared across every symbol using a
    strategy (e.g. MEAN_REVERSION_ADX_MAX, MOMENTUM_BREAKOUT_MIN_VOL_RATIO)
    -- distinct from sweep_symbol_param, which only covers per-symbol
    config.INSTRUMENTS[symbol]["params"] entries like z_entry or
    trend_adx_min. No indicator recomputation needed; these thresholds are
    only read at evaluate()-time."""
    report = RobustnessReport(param_name=config_attr)
    if bars.empty:
        return report
    for value in values:
        with override_module_attr(config, **{config_attr: value}):
            result = await _run_backtest_for_strategy(bars, strategy_name, symbol)
        report.points.append(SweepPoint(params={config_attr: value}, result=result))
    return report


async def _apply_chosen_params(
    bars: pd.DataFrame, strategy_name: str, symbol: str, chosen: dict, is_module_attr: bool
) -> BacktestResult:
    """Run one backtest with a specific parameter choice applied."""
    if bars.empty:
        return _empty_result(symbol, strategy_name, 100_000.0)
    if is_module_attr:
        module = STRATEGY_MODULES[strategy_name]
        with override_module_attr(module, **chosen):
            recomputed = module.compute_indicators(_raw_ohlcv(bars), symbol)
            return await _run_backtest_for_strategy(recomputed, strategy_name, symbol)
    with override_symbol_param(symbol, **chosen):
        return await _run_backtest_for_strategy(bars, strategy_name, symbol)


@dataclass
class OutOfSampleResult:
    symbol: str
    strategy: str
    param_name: str
    chosen_params: dict
    in_sample: BacktestResult
    out_of_sample: BacktestResult
    robustness: RobustnessReport


async def run_out_of_sample_test(
    broker,
    strategy_name: str,
    symbol: str,
    months: int,
    param_name: str,
    values: list,
    in_sample_frac: float = 0.7,
    is_module_attr: bool = False,
) -> OutOfSampleResult:
    """Split history chronologically. Sweep `values` on the in-sample
    slice ONLY, pick whichever is most robust (see RobustnessReport), then
    run that single choice on the out-of-sample slice, never touched
    during selection. The out-of-sample number is the honest performance
    estimate; the in-sample number is not (it's what was optimized for)."""
    full_bars = await _fetch_prepared_bars(broker, strategy_name, symbol, months)
    in_sample_bars, out_sample_bars = split_out_of_sample(full_bars, in_sample_frac)

    sweep_fn = sweep_module_param if is_module_attr else sweep_symbol_param
    report = await sweep_fn(in_sample_bars, strategy_name, symbol, param_name, values)
    best = report.best
    chosen = dict(best.params) if best else {}

    in_sample_result = await _apply_chosen_params(in_sample_bars, strategy_name, symbol, chosen, is_module_attr)
    out_sample_result = await _apply_chosen_params(out_sample_bars, strategy_name, symbol, chosen, is_module_attr)

    return OutOfSampleResult(
        symbol=symbol,
        strategy=strategy_name,
        param_name=param_name,
        chosen_params=chosen,
        in_sample=in_sample_result,
        out_of_sample=out_sample_result,
        robustness=report,
    )


def print_out_of_sample_report(result: OutOfSampleResult) -> None:
    print(f"=== Out-of-sample test: {result.symbol} / {result.strategy} / {result.param_name} ===")
    result.robustness.print_table()
    print(f"  chosen: {result.chosen_params}")
    is_pf = "inf" if result.in_sample.profit_factor == float("inf") else f"{result.in_sample.profit_factor:.2f}"
    oos_pf = "inf" if result.out_of_sample.profit_factor == float("inf") else f"{result.out_of_sample.profit_factor:.2f}"
    print(
        f"  in-sample:     trades={result.in_sample.trades:3d}  win%={result.in_sample.win_rate:5.1f}  "
        f"PF={is_pf:>5s}  sharpe={result.in_sample.sharpe:6.2f}  return={result.in_sample.total_return_pct:+6.1f}%"
    )
    print(
        f"  out-of-sample: trades={result.out_of_sample.trades:3d}  win%={result.out_of_sample.win_rate:5.1f}  "
        f"PF={oos_pf:>5s}  sharpe={result.out_of_sample.sharpe:6.2f}  return={result.out_of_sample.total_return_pct:+6.1f}%"
    )
    gap = result.in_sample.sharpe - result.out_of_sample.sharpe
    if gap > 0.3:
        print(f"  -> WARNING: in-sample Sharpe is {gap:.2f} higher than out-of-sample -- likely overfit to the in-sample period.")
    else:
        print("  -> out-of-sample performance is consistent with in-sample -- not an obvious overfit.")


def _chain_equity_curves(curves: list[list[float]], initial_equity: float = 100_000.0) -> list[float]:
    """Chain per-fold equity curves (each starting fresh at initial_equity)
    into one continuous curve, compounding each fold's growth onto the
    previous fold's ending equity. This is what "out-of-sample performance
    across the whole walk-forward run" means -- not just averaging folds."""
    chained = [initial_equity]
    running = initial_equity
    for curve in curves:
        if not curve:
            continue
        base = curve[0] if curve[0] else initial_equity
        for value in curve[1:]:
            growth = value / base
            chained.append(running * growth)
        if len(curve) > 1:
            running = chained[-1]
    return chained


@dataclass
class WalkForwardFold:
    window_start: object
    window_end: object
    chosen_params: dict
    in_sample: BacktestResult
    out_of_sample: BacktestResult


@dataclass
class WalkForwardResult:
    symbol: str
    strategy: str
    param_name: str
    folds: list[WalkForwardFold] = field(default_factory=list)
    pooled_out_of_sample: dict = field(default_factory=dict)
    chained_equity_curve: list[float] = field(default_factory=list)


async def run_walk_forward(
    broker,
    strategy_name: str,
    symbol: str,
    param_name: str,
    values: list,
    total_months: int,
    num_folds: int = 4,
    in_sample_frac: float = 0.7,
    is_module_attr: bool = False,
) -> WalkForwardResult:
    """Divide the full available history into `num_folds` consecutive,
    non-overlapping chunks. Within each fold: sweep `values` on that
    fold's in-sample slice, pick the most robust, test on that fold's
    out-of-sample slice, then move to the next fold. No fold's parameter
    choice is ever picked using data from a later fold -- the whole point
    is that a parameter chosen this way can't have been fit to data it's
    later reported against (#2/#12).
    """
    full_bars = await _fetch_prepared_bars(broker, strategy_name, symbol, total_months)
    result = WalkForwardResult(symbol=symbol, strategy=strategy_name, param_name=param_name)
    if full_bars.empty or len(full_bars) < num_folds * 20:
        return result

    fold_size = len(full_bars) // num_folds
    for i in range(num_folds):
        start_idx = i * fold_size
        end_idx = len(full_bars) if i == num_folds - 1 else (i + 1) * fold_size
        fold_bars = full_bars.iloc[start_idx:end_idx]
        if len(fold_bars) < 20:
            continue

        in_sample_bars, out_sample_bars = split_out_of_sample(fold_bars, in_sample_frac)
        if in_sample_bars.empty or out_sample_bars.empty:
            continue

        sweep_fn = sweep_module_param if is_module_attr else sweep_symbol_param
        report = await sweep_fn(in_sample_bars, strategy_name, symbol, param_name, values)
        best = report.best
        chosen = dict(best.params) if best else {}

        in_sample_result = await _apply_chosen_params(in_sample_bars, strategy_name, symbol, chosen, is_module_attr)
        out_sample_result = await _apply_chosen_params(out_sample_bars, strategy_name, symbol, chosen, is_module_attr)

        result.folds.append(
            WalkForwardFold(
                window_start=fold_bars.index[0],
                window_end=fold_bars.index[-1],
                chosen_params=chosen,
                in_sample=in_sample_result,
                out_of_sample=out_sample_result,
            )
        )

    pooled_trades: list[TradeRecord] = [t for fold in result.folds for t in fold.out_of_sample.trade_log]
    result.chained_equity_curve = _chain_equity_curves([fold.out_of_sample.equity_curve for fold in result.folds])
    if pooled_trades:
        result.pooled_out_of_sample = _compute_stats(pooled_trades, result.chained_equity_curve, 100_000.0)
    return result


def print_walk_forward_report(result: WalkForwardResult) -> None:
    print(f"=== Walk-forward: {result.symbol} / {result.strategy} / {result.param_name} ({len(result.folds)} folds) ===")
    for i, fold in enumerate(result.folds):
        gap = fold.in_sample.sharpe - fold.out_of_sample.sharpe
        flag = " <- overfit gap" if gap > 0.3 else ""
        print(
            f"  fold {i+1} [{fold.window_start.date()} -> {fold.window_end.date()}]  chosen={fold.chosen_params}  "
            f"in-sample sharpe={fold.in_sample.sharpe:6.2f}  out-of-sample sharpe={fold.out_of_sample.sharpe:6.2f}{flag}"
        )
    if result.pooled_out_of_sample:
        s = result.pooled_out_of_sample
        pf = "inf" if s["profit_factor"] == float("inf") else f"{s['profit_factor']:.2f}"
        print(
            f"\n  POOLED OUT-OF-SAMPLE (the honest estimate -- only data never used for parameter choice):\n"
            f"    trades={s['trades']}  win%={s['win_rate']:.1f}  PF={pf}  sharpe={s['sharpe']:.2f}  "
            f"sortino={s['sortino']:.2f}  calmar={s['calmar']:.2f}  max_dd={s['max_dd']:.1f}%  return={s['total_return_pct']:+.1f}%"
        )
        if s["trades"] < MIN_TRADES_FOR_VALIDATION:
            print(f"    -> WARNING: only {s['trades']} pooled trades (below {MIN_TRADES_FOR_VALIDATION}) -- not enough to trust this number either way.")
    else:
        print("  no folds produced enough data to report")


if __name__ == "__main__":
    import asyncio

    from bot.broker import Broker
    from bot.logs import setup_logging

    setup_logging()

    async def _main():
        broker = Broker()
        result = await run_walk_forward(
            broker,
            "trend_following",
            "GLD",
            param_name="FAST_PERIOD",
            values=[10, 20, 30, 50],
            total_months=config.BACKTEST_MONTHS,
            num_folds=4,
            is_module_attr=True,
        )
        print_walk_forward_report(result)

    asyncio.run(_main())
