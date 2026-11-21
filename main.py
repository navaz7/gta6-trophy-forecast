"""GTA VI trophy rarity forecast: saturating growth on the logit scale.

Model summary
-------------
Exophase earn rates move for two reasons at once: the tracked-player pool
grows (new, slower players dilute the early adopters who earn fast), and the
players already tracked keep progressing (more of them earn each trophy as
the days pass).  A single persistence forecast misses both.  This program
therefore fits, per trophy, three small models on the logit scale and blends
them with weights chosen by a within-case backtest:

  * time model    -- logit(rate) linear in ln(1 + days since release):
                     progression with naturally saturating (log-time) growth;
  * pool model    -- logit(rate) linear in ln(players tracked): captures the
                     dilution / catch-up balance as the pool expands;
  * earners model -- ln(earners + 1) linear in ln(1 + days since release),
                     divided by a separately forecast player count at the
                     target, so the numerator (people who have earned it) and
                     the denominator (people tracked) grow at their own pace.

Slopes are shrunk toward the pooled slope of the trophy's tier
(platinum / gold / silver / bronze), because trophies of a tier share a
shape: story trophies saturate high, grind and collectible trophies stay
low, and 55 trophies inform one another far better than each fits alone.
The quantile spread combines each trophy's residual volatility, the
disagreement between the three models, and a random-walk term that grows
with the square root of the horizon; low-rate trophies get a wider spread,
since on the logit scale they move the most.

Only the Python standard library is used, there is no randomness, and the
program reads every ``*.in`` file in the input directory and writes one
``*.out`` file per case, exactly as the competition contract requires.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Generous week-one ceilings per tier (percent): a soft guard against a
# runaway extrapolation, far above anything a sane fit reaches.
TIER_CAP = {"platinum": 20.0, "gold": 65.0, "silver": 85.0, "bronze": 98.0}
TIER_FLOOR = 0.02


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def logistic(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(text):
    """Parse an ISO-8601 timestamp (with Z, offsets, or a bare date)."""
    if not text:
        return None
    s = str(text).strip()
    try:
        if s.endswith("Z"):
            dt = datetime.fromisoformat(s[:-1] + "+00:00")
        elif len(s) == 10:
            dt = datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
        else:
            dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def ols(xs, ys):
    """Least-squares (intercept, slope); slope 0 when x has no spread."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return my, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return my - slope * mx, slope


def shrink_slope(own_slope, pooled_slope, n_points, strength=4.0):
    """Pull a noisy per-trophy slope toward its tier's pooled slope."""
    if pooled_slope is None:
        return own_slope
    w = n_points / (n_points + strength)
    return w * own_slope + (1.0 - w) * pooled_slope


def pooled_slope(series_list, x_of, y_of):
    """Common slope from within-series demeaning (fixed-effects style)."""
    num = 0.0
    den = 0.0
    used = 0
    for pts in series_list:
        if len(pts) < 3:
            continue
        xs = [x_of(p) for p in pts]
        ys = [y_of(p) for p in pts]
        if max(xs) - min(xs) <= 1e-9 or max(ys) - min(ys) <= 1e-9:
            continue
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        num += sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den += sum((x - mx) ** 2 for x in xs)
        used += 1
    if used == 0 or den <= 1e-12:
        return None
    return num / den


def forecast_players(player_pts, target_days):
    """Forecast players tracked at the target from the (day, players) series.

    Blends a power law (ln P linear in ln(1+days), growth that slows) with a
    capped exponential, and never forecasts a shrinking pool.
    """
    if not player_pts:
        return None
    last_day, last_p = player_pts[-1]
    distinct = []
    for d, p in player_pts:
        if not distinct or p != distinct[-1][1]:
            distinct.append((d, p))
    if len(distinct) < 2:
        return float(last_p)
    xs = [math.log1p(d) for d, _ in distinct]
    ys = [math.log(p) for _, p in distinct]
    a, b = ols(xs, ys)
    b = min(max(b, 0.0), 3.0)
    power_pred = a + b * math.log1p(target_days)
    xs2 = [d for d, _ in distinct]
    a2, b2 = ols(xs2, ys)
    b2 = min(max(b2, 0.0), 0.9)
    exp_pred = a2 + b2 * target_days
    blended = 0.65 * power_pred + 0.35 * exp_pred
    p_target = math.exp(blended)
    p_target = max(float(last_p), min(p_target, float(last_p) * 120.0))
    return p_target


def fit_predict(xs, ys, x_target, pooled):
    """Fit y ~ x with slope shrinkage; return (prediction, residual sd)."""
    a, b = ols(xs, ys)
    b = shrink_slope(b, pooled, len(xs))
    # Refit the intercept through the shrunken slope.
    a = sum(y - b * x for x, y in zip(xs, ys)) / len(xs)
    pred = a + b * x_target
    if len(xs) >= 3:
        resid = [y - (a + b * x) for x, y in zip(xs, ys)]
        sd = math.sqrt(sum(r * r for r in resid) / max(1, len(resid) - 2))
    else:
        sd = 0.45
    return pred, sd, b


def backtest_rmse(xs, ys, x_pool, pooled):
    """Fit on the first ~70% of points, score the rest (logit RMSE)."""
    n = len(xs)
    if n < 5:
        return None
    cut = max(3, int(round(n * 0.7)))
    if cut >= n:
        return None
    a, b = ols(xs[:cut], ys[:cut])
    b = shrink_slope(b, pooled, cut)
    a = sum(y - b * x for x, y in zip(xs[:cut], ys[:cut])) / cut
    errs = []
    for x, y in zip(xs[cut:], ys[cut:]):
        errs.append((a + b * x) - y)
    if not errs:
        return None
    return math.sqrt(sum(e * e for e in errs) / len(errs))


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    horizon_days = float(case["horizon_minutes"]) / 1440.0

    checkpoint = parse_time(case.get("at")) or parse_time(history[-1].get("at"))
    target_dt = parse_time(case.get("target"))
    if target_dt is None and checkpoint is not None:
        from datetime import timedelta

        target_dt = checkpoint + timedelta(minutes=float(case["horizon_minutes"]))
    release_dt = parse_time((case.get("game") or {}).get("release_date"))
    first_dt = parse_time(history[0].get("at"))
    if release_dt is None or (first_dt is not None and first_dt < release_dt):
        origin = first_dt
    else:
        origin = release_dt
    if origin is None:
        origin = checkpoint

    def days_of(dt):
        if dt is None or origin is None:
            return 0.0
        return max(0.0, (dt - origin).total_seconds() / 86400.0)

    target_days = days_of(target_dt)
    if target_days <= 0.0:
        target_days = days_of(checkpoint) + horizon_days

    # Global player series, compressed to moments the count actually changed.
    player_pts = []
    for read in history:
        p = read.get("players_tracked")
        if p is None or p <= 0:
            continue
        d = days_of(parse_time(read.get("at")))
        if not player_pts or p != player_pts[-1][1]:
            player_pts.append((d, float(p)))
    last_players = player_pts[-1][1] if player_pts else None
    p_target = forecast_players(player_pts, target_days)
    if p_target is None:
        p_target = last_players or 1.0

    # Per-trophy compressed series: (days, ln(1+days), ln players, rate).
    all_series = []
    for i in range(len(trophies)):
        pts = []
        for read in history:
            rates = read.get("rates") or []
            rate = rates[i] if i < len(rates) else None
            if rate is None:
                continue
            p = read.get("players_tracked") or last_players or 1
            d = days_of(parse_time(read.get("at")))
            key = (float(p), float(rate))
            if pts and pts[-1][3] == key[0] and pts[-1][4] == key[1]:
                continue
            pts.append((d, math.log1p(d), math.log(max(1.0, float(p))),
                        float(p), float(rate)))
        all_series.append(pts)

    # Pooled slopes per tier (and globally) for each regressor family.
    def tier_groups():
        groups = {}
        for idx, tr in enumerate(trophies):
            groups.setdefault(tr.get("tier", "?"), []).append(idx)
        return groups

    groups = tier_groups()
    x_time = lambda p: p[1]
    x_pool = lambda p: p[2]
    y_logit = lambda p: logit(p[4])
    y_earn = lambda p: math.log(max(0.0, p[4] / 100.0 * p[3]) + 1.0)

    pooled = {}
    for label, x_of, y_of in (("time", x_time, y_logit), ("pool", x_pool, y_logit),
                              ("earn", x_time, y_earn)):
        glob = pooled_slope([all_series[i] for i in range(len(trophies))], x_of, y_of)
        per_tier = {}
        for tier, idxs in groups.items():
            per_tier[tier] = pooled_slope([all_series[i] for i in idxs], x_of, y_of)
        pooled[label] = (glob, per_tier)

    def pooled_for(label, tier):
        glob, per_tier = pooled[label]
        val = per_tier.get(tier)
        return val if val is not None else glob

    rows = []
    for i, trophy in enumerate(trophies):
        tier = trophy.get("tier", "bronze")
        pts = all_series[i]
        # Last observed rate (the trophies forecast are those of the last
        # read; fall back to the most recent non-null read if needed).
        last_rate = None
        last_rates = history[-1].get("rates") or []
        if i < len(last_rates):
            last_rate = last_rates[i]
        if last_rate is None:
            for read in reversed(history):
                rr = read.get("rates") or []
                if i < len(rr) and rr[i] is not None:
                    last_rate = rr[i]
                    break
        if last_rate is None:
            # No observation at all: use the tier's median last rate.
            peers = []
            for j, tr2 in enumerate(trophies):
                if tr2.get("tier") == tier:
                    rr = history[-1].get("rates") or []
                    if j < len(rr) and rr[j] is not None:
                        peers.append(float(rr[j]))
            peers.sort()
            last_rate = peers[len(peers) // 2] if peers else 1.0
        last_rate = float(last_rate)
        l_last = logit(last_rate)

        preds = []
        weights = []
        resid_sds = []

        if len(pts) >= 2 and max(p[4] for p in pts) - min(p[4] for p in pts) > 1e-9:
            xs_t = [p[1] for p in pts]
            ys_l = [logit(p[4]) for p in pts]
            # --- time model ---
            pred, sd, _ = fit_predict(xs_t, ys_l, math.log1p(target_days),
                                      pooled_for("time", tier))
            preds.append(pred)
            resid_sds.append(sd)
            rmse = backtest_rmse(xs_t, ys_l, None, pooled_for("time", tier))
            weights.append(1.0 / ((rmse if rmse is not None else 0.55) ** 2 + 0.05))
            # --- pool model ---
            xs_p = [p[2] for p in pts]
            if max(xs_p) - min(xs_p) > 1e-9:
                pred2, sd2, _ = fit_predict(xs_p, ys_l, math.log(max(1.0, p_target)),
                                            pooled_for("pool", tier))
                preds.append(pred2)
                resid_sds.append(sd2)
                rmse2 = backtest_rmse(xs_p, ys_l, None, pooled_for("pool", tier))
                weights.append(1.0 / ((rmse2 if rmse2 is not None else 0.55) ** 2 + 0.05))
            # --- earners model ---
            ys_e = [math.log(max(0.0, p[4] / 100.0 * p[3]) + 1.0) for p in pts]
            pred_e, sd_e, _ = fit_predict(xs_t, ys_e, math.log1p(target_days),
                                          pooled_for("earn", tier))
            earners_target = max(0.0, math.exp(pred_e) - 1.0)
            rate_e = 100.0 * earners_target / max(1.0, p_target)
            preds.append(logit(rate_e))
            resid_sds.append(sd_e)
            rmse3 = backtest_rmse(xs_t, ys_e, None, pooled_for("earn", tier))
            weights.append(0.8 / ((rmse3 if rmse3 is not None else 0.7) ** 2 + 0.05))

        if preds:
            # No single sub-model may drag the forecast more than 2.2 logit
            # units from the last read: an earners extrapolation whose
            # player-growth denominator is under-forecast otherwise explodes.
            preds = [min(max(p, l_last - 2.2), l_last + 2.2) for p in preds]
            total_w = sum(weights)
            median_l = sum(w * p for w, p in zip(weights, preds)) / total_w
            # A touch of persistence keeps a wild extrapolation honest.
            median_l = 0.9 * median_l + 0.1 * l_last
            disagree = math.sqrt(
                sum(w * (p - median_l) ** 2 for w, p in zip(weights, preds)) / total_w)
            resid_sd = sum(resid_sds) / len(resid_sds)
        else:
            median_l = l_last
            disagree = 0.0
            resid_sd = 0.45

        # Soft tier ceiling / floor and a cap on the total logit move.
        cap_l = logit(TIER_CAP.get(tier, 98.0))
        floor_l = logit(TIER_FLOOR)
        median_l = min(median_l, cap_l)
        median_l = max(median_l, floor_l)
        median_l = min(max(median_l, l_last - 3.5), l_last + 3.5)

        # Spread: residual volatility + model disagreement + horizon walk,
        # widened for low-rate trophies, which move most on the logit scale.
        walk = 0.45 * math.sqrt(max(horizon_days, 0.01))
        sigma = math.sqrt(resid_sd ** 2 * 0.6 + disagree ** 2 + walk ** 2)
        if l_last < logit(2.0):
            sigma *= 1.15
        if l_last > logit(90.0):
            sigma *= 1.1
        sigma = min(max(sigma, 0.35), 1.8)

        qs = []
        for z in Z:
            q = logistic(median_l + z * sigma)
            qs.append(round4(min(99.99, max(0.01, q))))
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
