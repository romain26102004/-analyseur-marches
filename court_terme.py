"""
Analyse court terme : indicateurs quotidiens, probabilités historiques par
horizon, situations comparables, fourchettes probables et probabilité
d'atteindre un objectif ou un stop.

Principe d'honnêteté : chaque probabilité est accompagnée du nombre de cas
indépendants qui la fondent et d'un test indiquant si l'écart avec le hasard
est significatif. Sur le court terme, la plupart des signaux ne le sont pas.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from analyse import _macd, _rsi

# Libellé -> (séances de bourse, jours calendaires pour la crypto)
HORIZONS = {"1 semaine": (5, 7), "1 mois": (21, 30), "3 mois": (63, 91)}
QUANTILES = [5, 16, 50, 84, 95]
# Seuil de significativité, calibré sur des marchés simulés au hasard : avec 2,
# environ 5 % des tests sortaient « significatifs » par pur hasard ; avec 3, moins de 1 %.
SEUIL_Z = 3.0


def _lignes(horizon: str, crypto: bool) -> int:
    return HORIZONS[horizon][1 if crypto else 0]


# --------------------------------------------------------------------------
# Indicateurs quotidiens
# --------------------------------------------------------------------------

def calculer_indicateurs(h: pd.DataFrame, crypto: bool = False) -> pd.DataFrame:
    c = h["Close"]
    hi = h["High"] if "High" in h else c
    lo = h["Low"] if "Low" in h else c
    mois, an = (30, 365) if crypto else (21, 252)

    d = pd.DataFrame(index=h.index)
    d["close"] = c
    d["rsi"] = _rsi(c, 14)
    d["macd"], d["macd_signal"] = _macd(c)

    mm20, et20 = c.rolling(20).mean(), c.rolling(20).std()
    d["mm20"], d["mm50"], d["mm200"] = mm20, c.rolling(50).mean(), c.rolling(200).mean()
    d["pct_b"] = (c - (mm20 - 2 * et20)) / (4 * et20).replace(0, np.nan)

    tr = pd.concat([hi - lo, (hi - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    d["atr_pct"] = atr / c

    hausse, baisse = hi.diff(), -lo.diff()
    dm_plus = hausse.where((hausse > baisse) & (hausse > 0), 0.0)
    dm_moins = baisse.where((baisse > hausse) & (baisse > 0), 0.0)
    atr_nz = atr.replace(0, np.nan)
    d["di_plus"] = 100 * dm_plus.ewm(alpha=1 / 14, adjust=False).mean() / atr_nz
    d["di_moins"] = 100 * dm_moins.ewm(alpha=1 / 14, adjust=False).mean() / atr_nz
    dx = 100 * (d["di_plus"] - d["di_moins"]).abs() / (d["di_plus"] + d["di_moins"]).replace(0, np.nan)
    d["adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

    if "Volume" in h and h["Volume"].fillna(0).sum() > 0:
        v = h["Volume"].replace(0, np.nan)
        d["vol_rel"] = v / v.rolling(20).mean()
    else:
        d["vol_rel"] = np.nan

    d["perf_1s"] = c / c.shift(7 if crypto else 5) - 1
    d["perf_1m"] = c / c.shift(mois) - 1
    d["recul_52s"] = c / c.rolling(an, min_periods=20).max() - 1
    return d


def lire_indicateurs(d: pd.DataFrame) -> list[dict]:
    """Lecture de la dernière séance : signal +1 (haussier), -1 (baissier) ou 0 (neutre / info)."""
    x = d.iloc[-1]
    out: list[dict] = []

    def ajouter(nom, valeur, signal, lecture):
        out.append({"Indicateur": nom, "Valeur": valeur, "Signal": signal, "Lecture": lecture})

    def ok(v):
        return v is not None and not (isinstance(v, float) and math.isnan(v))

    if ok(x["rsi"]):
        r = x["rsi"]
        if r > 70:
            ajouter("RSI 14 jours", f"{r:.0f}", 0, "Suracheté : l'élan est fort, mais le risque de repli augmente.")
        elif r < 30:
            ajouter("RSI 14 jours", f"{r:.0f}", 0, "Survendu : un rebond technique est fréquent, sans être garanti.")
        elif r >= 50:
            ajouter("RSI 14 jours", f"{r:.0f}", 1, "Au-dessus de 50 : les hausses dominent récemment.")
        else:
            ajouter("RSI 14 jours", f"{r:.0f}", -1, "Sous 50 : les baisses dominent récemment.")

    if ok(x["macd"]) and ok(x["macd_signal"]):
        hausse = x["macd"] > x["macd_signal"]
        ajouter("MACD (12, 26, 9)", "au-dessus du signal" if hausse else "sous le signal", 1 if hausse else -1,
                "L'élan à court terme s'améliore." if hausse else "L'élan à court terme se dégrade.")

    if ok(x["pct_b"]):
        b = x["pct_b"]
        if b > 1:
            s, lec = 0, "Au-dessus de la bande haute : mouvement fort mais étiré."
        elif b < 0:
            s, lec = 0, "Sous la bande basse : chute forte, souvent suivie d'une stabilisation."
        elif b >= 0.5:
            s, lec = 1, "Dans la moitié haute du canal de volatilité."
        else:
            s, lec = -1, "Dans la moitié basse du canal de volatilité."
        ajouter("Bandes de Bollinger (%B)", f"{b:.2f}", s, lec)

    for col, nom in (("mm20", "Cours vs moyenne 20 jours"), ("mm50", "Cours vs moyenne 50 jours")):
        if ok(x[col]):
            ecart = x["close"] / x[col] - 1
            ajouter(nom, f"{ecart * 100:+.1f} %", 1 if ecart > 0 else -1,
                    "Au-dessus : tendance de court terme haussière." if ecart > 0
                    else "En dessous : tendance de court terme baissière.")

    if ok(x["adx"]):
        a = x["adx"]
        sens = "haussière" if x["di_plus"] > x["di_moins"] else "baissière"
        if a > 25:
            ajouter("ADX (force de la tendance)", f"{a:.0f}", 1 if sens == "haussière" else -1,
                    f"Tendance marquée et {sens} : les signaux de tendance sont plus fiables.")
        elif a < 20:
            ajouter("ADX (force de la tendance)", f"{a:.0f}", 0,
                    "Pas de tendance nette : le cours oscille dans une fourchette.")
        else:
            ajouter("ADX (force de la tendance)", f"{a:.0f}", 0, f"Tendance {sens} en formation.")

    if ok(x["vol_rel"]):
        v = x["vol_rel"]
        lec = ("Volume nettement au-dessus de la normale : le mouvement récent est suivi." if v > 1.5
               else "Volume faible : le mouvement récent manque de conviction." if v < 0.7
               else "Volume dans la normale.")
        ajouter("Volume vs moyenne 20 jours", f"× {v:.1f}", 0, lec)

    if ok(x["atr_pct"]):
        ajouter("Variation quotidienne typique (ATR)", f"{x['atr_pct'] * 100:.1f} %", 0,
                "Ordre de grandeur d'une séance normale : utile pour placer un stop.")

    for col, nom in (("perf_1s", "Performance 1 semaine"), ("perf_1m", "Performance 1 mois")):
        if ok(x[col]):
            ajouter(nom, f"{x[col] * 100:+.1f} %", 0, "Information.")

    if ok(x["recul_52s"]):
        ajouter("Distance au plus haut 52 semaines", f"{x['recul_52s'] * 100:+.1f} %", 0, "Information.")
    return out


# --------------------------------------------------------------------------
# Probabilités historiques
# --------------------------------------------------------------------------

def _rendements_futurs(close: pd.Series, n: int) -> pd.Series:
    return close.shift(-n) / close - 1


def _cas_independants(masque: pd.Series, n: int) -> int:
    """Les fenêtres qui se chevauchent ne sont pas indépendantes : on compte
    chaque épisode consécutif comme ceil(durée / horizon) observations."""
    if not masque.any():
        return 0
    episodes = (masque != masque.shift()).cumsum()
    durees = masque.groupby(episodes).sum()
    durees = durees[durees > 0]
    return int(sum(math.ceil(dd / n) for dd in durees))


def _conditions(d: pd.DataFrame) -> dict[str, pd.Series]:
    """Situations testées. Seules celles vraies aujourd'hui sont affichées."""
    tendance_h = d["mm50"] > d["mm200"]
    zones = pd.cut(d["rsi"], [-0.1, 30, 50, 70, 100.1], labels=["< 30", "30-50", "50-70", "> 70"])
    cond = {
        "RSI sous 30 (survendu)": d["rsi"] < 30,
        "RSI au-dessus de 70 (suracheté)": d["rsi"] > 70,
        "Cours sous la bande de Bollinger basse": d["pct_b"] < 0,
        "Cours au-dessus de la bande de Bollinger haute": d["pct_b"] > 1,
        "Cours au-dessus de la moyenne 50 jours": d["close"] > d["mm50"],
        "Cours sous la moyenne 50 jours": d["close"] < d["mm50"],
        "Tendance de fond haussière (moy. 50 j > 200 j)": tendance_h & d["mm200"].notna(),
        "Tendance de fond baissière (moy. 50 j < 200 j)": ~tendance_h & d["mm200"].notna(),
        "Baisse de plus de 10 % sur un mois": d["perf_1m"] < -0.10,
        "Hausse de plus de 15 % sur un mois": d["perf_1m"] > 0.15,
        "Volume plus de 2 fois supérieur à la normale": d["vol_rel"] > 2,
        "À moins de 3 % du plus haut 52 semaines": d["recul_52s"] > -0.03,
    }
    zone = zones.iloc[-1]
    if pd.notna(zone) and pd.notna(d["mm200"].iloc[-1]):
        th = bool(tendance_h.iloc[-1])
        cond[f"Situation combinée : RSI {zone} et tendance de fond {'haussière' if th else 'baissière'}"] = (
            (zones == zone) & (tendance_h == th) & d["mm200"].notna()
        )
    return {k: v.fillna(False).astype(bool) for k, v in cond.items()}


@dataclass
class StatHorizon:
    proba_hausse: float | None
    rendement_median: float | None
    cas: int
    ecart: float | None = None       # en points vs taux de base
    z: float | None = None
    verdict: str = ""


def _stat(masque: pd.Series, futurs: pd.Series, n: int, base: StatHorizon | None = None) -> StatHorizon:
    valide = masque & futurs.notna()
    if valide.sum() == 0:
        return StatHorizon(None, None, 0, verdict="Aucun cas")
    r = futurs[valide]
    p = float((r > 0).mean())
    cas = _cas_independants(valide, n)
    st = StatHorizon(p, float(r.median()), cas)
    if base is not None and base.proba_hausse is not None:
        pb = base.proba_hausse
        st.ecart = (p - pb) * 100
        st.z = (p - pb) / math.sqrt(max(pb * (1 - pb), 1e-9) / max(cas, 1))
        if cas < 15:
            st.verdict = "Trop peu de cas"
        elif abs(st.z) >= SEUIL_Z:
            st.verdict = "Écart significatif"
        else:
            st.verdict = "Pas d'écart fiable avec le hasard"
    return st


# --------------------------------------------------------------------------
# Simulation : fourchettes et objectif / stop
# --------------------------------------------------------------------------

def simuler(close: pd.Series, n_max: int, crypto: bool, n_sim: int = 4000, graine: int = 42):
    """Tire au hasard des variations quotidiennes passées (3 dernières années),
    recentrées (aucune tendance supposée) et ajustées à la volatilité actuelle."""
    an = 365 if crypto else 252
    lr = np.log(close).diff().dropna().iloc[-3 * an:]
    if len(lr) < 60:
        return None, None
    vol_hist = float(lr.std())
    vol_actuelle = float(np.sqrt((lr ** 2).ewm(alpha=0.06, adjust=False).mean().iloc[-1]))
    ratio = min(max(vol_actuelle / vol_hist, 0.5), 2.0) if vol_hist > 0 else 1.0
    tirages = (lr - lr.mean()).to_numpy() * ratio
    rng = np.random.default_rng(graine)
    idx = rng.integers(0, len(tirages), size=(n_sim, n_max))
    chemins = float(close.iloc[-1]) * np.exp(np.cumsum(tirages[idx], axis=1))
    return chemins, vol_actuelle * math.sqrt(an)


def proba_objectif_stop(chemins: np.ndarray, n: int, prix: float, objectif: float, stop: float) -> dict:
    """Probabilité de toucher l'objectif (+objectif) avant le stop (-stop) en n séances (clôtures)."""
    c = chemins[:, :n]
    haut, bas = prix * (1 + objectif), prix * (1 - stop)
    t_h, t_b = c >= haut, c <= bas
    premier_h = np.where(t_h.any(axis=1), t_h.argmax(axis=1), n + 1)
    premier_b = np.where(t_b.any(axis=1), t_b.argmax(axis=1), n + 1)
    obj = (premier_h < premier_b)
    stp = (premier_b <= premier_h) & (premier_b <= n)
    return {
        "objectif": float(obj.mean()),
        "stop": float(stp.mean()),
        "aucun": float(1 - obj.mean() - stp.mean()),
        "fin_en_hausse": float((c[:, -1] > prix).mean()),
    }


# --------------------------------------------------------------------------
# Synthèse
# --------------------------------------------------------------------------

@dataclass
class CourtTerme:
    crypto: bool
    prix: float
    indicateurs: list[dict]
    base: dict[str, StatHorizon]
    situations: dict[str, dict[str, StatHorizon]]
    fourchettes: dict[str, list[float]]   # horizon -> prix aux QUANTILES
    eventail: pd.DataFrame | None         # dates futures x quantiles
    chemins: np.ndarray | None
    vol_annuelle: float | None

    @property
    def resume_signaux(self) -> tuple[int, int, int]:
        s = [i["Signal"] for i in self.indicateurs]
        return s.count(1), s.count(-1), s.count(0)


def analyser_court_terme(historique: pd.DataFrame, crypto: bool = False) -> CourtTerme:
    h = historique.dropna(subset=["Close"])
    close = h["Close"]
    d = calculer_indicateurs(h, crypto)
    conditions = _conditions(d)
    tout = pd.Series(True, index=close.index)

    base, situations = {}, {}
    actives = [nom for nom, m in conditions.items() if bool(m.iloc[-1])]
    for nom in actives:
        situations[nom] = {}
    for hz in HORIZONS:
        n = _lignes(hz, crypto)
        futurs = _rendements_futurs(close, n)
        base[hz] = _stat(tout, futurs, n)
        for nom in actives:
            situations[nom][hz] = _stat(conditions[nom], futurs, n, base[hz])

    n_max = max(_lignes(hz, crypto) for hz in HORIZONS)
    chemins, vol = simuler(close, n_max, crypto)
    fourchettes, eventail = {}, None
    if chemins is not None:
        for hz in HORIZONS:
            fourchettes[hz] = list(np.percentile(chemins[:, _lignes(hz, crypto) - 1], QUANTILES))
        q = np.percentile(chemins, QUANTILES, axis=0).T
        debut = close.index[-1] + pd.Timedelta(days=1)
        dates = (pd.date_range(debut, periods=n_max, freq="D") if crypto
                 else pd.bdate_range(debut, periods=n_max))
        if close.index.tz is not None and dates.tz is None:
            dates = dates.tz_localize(close.index.tz)
        eventail = pd.DataFrame(q, index=dates, columns=[f"p{x}" for x in QUANTILES])

    return CourtTerme(
        crypto=crypto,
        prix=float(close.iloc[-1]),
        indicateurs=lire_indicateurs(d),
        base=base,
        situations=situations,
        fourchettes=fourchettes,
        eventail=eventail,
        chemins=chemins,
        vol_annuelle=vol,
    )
