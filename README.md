"""
TJR-style strategy bot + backtester (rule-based approximation).

Logic (5-minute bars, times in UTC):
  1. Key liquidity levels: previous-day high/low and Asian-session (00-07) high/low.
  2. Liquidity sweep: price wicks beyond a level and closes back inside it.
  3. Market-structure shift (MSS): a close beyond the recent swing that led into the sweep.
  4. Entry: limit order at the midpoint of the Fair Value Gap (FVG) left by the displacement.
  5. Stop: beyond the sweep extreme (+ buffer). Target: fixed R multiple.
  6. Risk: fixed % of equity per trade, max trades per day, flat at end of session.

Usage:
  python tjr_bot.py                       # synthetic data demo
  python tjr_bot.py --csv EURUSD_5m.csv   # real data: columns datetime,open,high,low,close (UTC)

Fill model is deliberately conservative: if SL and TP are both touched in one bar, SL is assumed first.
"""
import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Params:
    capital: float = 10_000.0
    risk_pct: float = 0.01          # equity risked per trade
    rr: float = 2.0                 # take-profit in R
    session_start: int = 7          # UTC hour (London open)
    session_end: int = 16           # UTC hour (flat after this)
    mss_lookback: int = 12          # bars before sweep used to define structure level
    setup_expiry: int = 36          # bars to get MSS after sweep
    order_expiry: int = 24          # bars the FVG limit order stays live
    stop_buffer: float = 0.00005    # price units beyond sweep extreme
    cost: float = 0.00010           # round-trip spread+slippage in price units
    max_trades_day: int = 2
    min_stop: float = 0.00020       # ignore tiny stops (cost would dominate)


# ---------------------------------------------------------------- data
def load_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    tcol = "datetime" if "datetime" in df.columns else df.columns[0]
    df[tcol] = pd.to_datetime(df[tcol], utc=True).dt.tz_localize(None)
    return df.set_index(tcol)[["open", "high", "low", "close"]].sort_index()


def synthetic(days=260, seed=7, start_price=1.10):
    """Random-walk 5m FX-like bars. Contains NO built-in edge."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=days * 288, freq="5min")
    idx = idx[idx.dayofweek < 5]
    hr = idx.hour.values
    mult = np.where((hr >= 7) & (hr < 16), 1.6, np.where(hr < 7, 0.7, 0.9))
    ret = rng.normal(0, 0.00018, len(idx)) * mult
    close = start_price * np.exp(np.cumsum(ret))
    open_ = np.r_[start_price, close[:-1]]
    wick = np.abs(rng.normal(0, 0.00010, (2, len(idx)))) * mult
    high = np.maximum(open_, close) + wick[0]
    low = np.minimum(open_, close) - wick[1]
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


# ---------------------------------------------------------------- strategy / backtest
def backtest(df, p: Params):
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    t = df.index
    dates = t.date
    hrs = t.hour.values
    n = len(df)

    equity = p.capital
    trades = []
    eq_curve = np.full(n, np.nan)

    prev_day = {}                       # date -> (H, L) of completed day
    day_hi = day_lo = None
    cur_date = None
    last_date = None
    asia_hi = asia_lo = None
    used = set()
    n_today = 0
    setup = None                        # active sweep/MSS tracker
    order = None                        # pending limit order
    pos = None                          # open position
    pd_hi = pd_lo = None

    for i in range(n):
        d = dates[i]
        if d != cur_date:               # new day: roll levels, reset state
            if cur_date is not None:
                pd_hi, pd_lo = day_hi, day_lo
            cur_date, day_hi, day_lo = d, h[i], l[i]
            asia_hi = asia_lo = None
            used, n_today, setup, order = set(), 0, None, None
        else:
            day_hi, day_lo = max(day_hi, h[i]), min(day_lo, l[i])
        if hrs[i] < p.session_start:
            asia_hi = h[i] if asia_hi is None else max(asia_hi, h[i])
            asia_lo = l[i] if asia_lo is None else min(asia_lo, l[i])

        in_session = p.session_start <= hrs[i] < p.session_end

        # ---- manage open position
        if pos is not None:
            side, entry, sl, tp, risk, t_in = pos
            exit_px = None
            if side == 1:
                if l[i] <= sl: exit_px = sl
                elif h[i] >= tp: exit_px = tp
            else:
                if h[i] >= sl: exit_px = sl
                elif l[i] <= tp: exit_px = tp
            if exit_px is None and not in_session:
                exit_px = c[i]
            if exit_px is not None:
                r = (side * (exit_px - entry) - p.cost) / risk
                equity += equity * p.risk_pct * r
                trades.append(dict(entry_time=t_in, exit_time=t[i], side="long" if side == 1 else "short",
                                   entry=entry, exit=exit_px, stop=sl, target=tp, R=r, equity=equity))
                pos = None
            eq_curve[i] = equity
            continue

        # ---- pending limit order
        if order is not None:
            side, entry, sl, tp, exp_i = order
            if i > exp_i or not in_session:
                order = None
            else:
                invalid = (side == 1 and l[i] <= sl) or (side == -1 and h[i] >= sl)
                touched = (side == 1 and l[i] <= entry) or (side == -1 and h[i] >= entry)
                if invalid and not touched:
                    order = None
                elif touched:
                    fill = min(o[i], entry) if side == 1 else max(o[i], entry)
                    risk = abs(fill - sl)
                    order = None
                    if risk >= p.min_stop:
                        pos = (side, fill, sl, tp, risk, t[i])
                        n_today += 1
                        # same-bar stop check (conservative)
                        if (side == 1 and l[i] <= sl) or (side == -1 and h[i] >= sl):
                            r = (-risk - p.cost) / risk
                            equity += equity * p.risk_pct * r
                            trades.append(dict(entry_time=t[i], exit_time=t[i], side="long" if side == 1 else "short",
                                               entry=fill, exit=sl, stop=sl, target=tp, R=r, equity=equity))
                            pos = None
            eq_curve[i] = equity
            continue

        eq_curve[i] = equity
        if not in_session or n_today >= p.max_trades_day or i < p.mss_lookback + 3:
            setup = None if not in_session else setup
            continue

        # ---- step 2: detect liquidity sweep
        if setup is None:
            levels_lo = [("PDL", pd_lo), ("ASL", asia_lo)]
            levels_hi = [("PDH", pd_hi), ("ASH", asia_hi)]
            for name, lv in levels_lo:
                if lv is not None and name not in used and l[i] < lv < c[i]:
                    setup = dict(side=1, level=name, ext=l[i], sweep_i=i,
                                 mss=h[i - p.mss_lookback:i + 1].max())
                    used.add(name); break
            if setup is None:
                for name, lv in levels_hi:
                    if lv is not None and name not in used and h[i] > lv > c[i]:
                        setup = dict(side=-1, level=name, ext=h[i], sweep_i=i,
                                     mss=l[i - p.mss_lookback:i + 1].min())
                        used.add(name); break
            continue

        # ---- step 3: MSS after sweep
        s = setup
        side = s["side"]
        if i - s["sweep_i"] > p.setup_expiry:
            setup = None; continue
        # sweep extends -> move extreme / structure
        if side == 1 and l[i] < s["ext"]:
            s["ext"], s["sweep_i"] = l[i], i
            s["mss"] = h[i - p.mss_lookback:i + 1].max(); continue
        if side == -1 and h[i] > s["ext"]:
            s["ext"], s["sweep_i"] = h[i], i
            s["mss"] = l[i - p.mss_lookback:i + 1].min(); continue

        broke = (side == 1 and c[i] > s["mss"]) or (side == -1 and c[i] < s["mss"])
        if not broke:
            continue

        # ---- step 4: find latest FVG from sweep to now
        fvg = None
        for k in range(i, max(s["sweep_i"], 2) - 1, -1):
            if side == 1 and l[k] > h[k - 2]:
                fvg = (h[k - 2], l[k]); break          # bullish gap (low, high)
            if side == -1 and h[k] < l[k - 2]:
                fvg = (h[k], l[k - 2]); break          # bearish gap (low, high)
        setup = None
        if fvg is None:
            continue
        entry = (fvg[0] + fvg[1]) / 2
        if side == 1:
            sl = s["ext"] - p.stop_buffer
            if not (sl < entry < c[i]): continue
            tp = entry + p.rr * (entry - sl)
        else:
            sl = s["ext"] + p.stop_buffer
            if not (c[i] < entry < sl): continue
            tp = entry - p.rr * (sl - entry)
        if abs(entry - sl) < p.min_stop:
            continue
        order = (side, entry, sl, tp, i + p.order_expiry)

    return pd.DataFrame(trades), pd.Series(eq_curve, index=t).ffill()


# ---------------------------------------------------------------- metrics
def metrics(tr, eq, p: Params):
    if tr.empty:
        return {"trades": 0}
    wins, losses = tr[tr.R > 0].R, tr[tr.R <= 0].R
    peak = eq.cummax()
    dd = ((eq - peak) / peak).min()
    daily = eq.resample("1D").last().dropna().pct_change().dropna()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else np.nan
    return {
        "trades": len(tr),
        "win_rate_%": round(100 * len(wins) / len(tr), 1),
        "avg_R": round(tr.R.mean(), 3),
        "profit_factor": round(wins.sum() / abs(losses.sum()), 2) if len(losses) and losses.sum() != 0 else np.inf,
        "total_return_%": round(100 * (eq.iloc[-1] / p.capital - 1), 2),
        "max_drawdown_%": round(100 * dd, 2),
        "sharpe_daily_ann": round(sharpe, 2),
        "longs/shorts": f"{(tr.side == 'long').sum()}/{(tr.side == 'short').sum()}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--rr", type=float, default=2.0)
    ap.add_argument("--risk", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=".")
    a = ap.parse_args()

    p = Params(rr=a.rr, risk_pct=a.risk)
    df = load_csv(a.csv) if a.csv else synthetic(seed=a.seed)
    print(f"Data: {'CSV ' + a.csv if a.csv else 'SYNTHETIC random walk (no real edge)'} | "
          f"{len(df):,} bars | {df.index[0]} -> {df.index[-1]}")
    tr, eq = backtest(df, p)
    m = metrics(tr, eq, p)
    for k, v in m.items():
        print(f"{k:>18}: {v}")

    if not tr.empty:
        tr.to_csv(f"{a.out}/tjr_trades.csv", index=False)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(10, 7), height_ratios=[3, 1])
        ax[0].plot(eq.index, eq.values, lw=1.2)
        ax[0].set_title("TJR-style strategy equity curve"); ax[0].set_ylabel("Equity")
        ax[1].bar(range(len(tr)), tr.R, color=np.where(tr.R > 0, "tab:green", "tab:red"))
        ax[1].set_ylabel("R per trade"); ax[1].set_xlabel("Trade #")
        fig.tight_layout(); fig.savefig(f"{a.out}/tjr_equity.png", dpi=130)


if __name__ == "__main__":
    main()
