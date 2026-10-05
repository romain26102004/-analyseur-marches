"""
Moteur d'analyse moyen / long terme pour actions, ETF et cryptomonnaies.

Données : Yahoo Finance (via la bibliothèque yfinance).

Principe : chaque critère reçoit une note de -2 (très défavorable) à +2
(très favorable). Les notes sont regroupées par pilier (technique,
fondamental, sentiment…), ramenées sur une échelle de -100 à +100, puis
pondérées selon le type d'actif pour donner un score global et une
orientation.

Ce n'est pas un conseil en investissement : c'est une synthèse d'indicateurs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:  # permet de tester la logique sans yfinance
    yf = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    _VADER = SentimentIntensityAnalyzer()
except Exception:  # bibliothèque absente : le pilier sentiment est ignoré
    _VADER = None


# --------------------------------------------------------------------------
# Paramètres
# --------------------------------------------------------------------------

# Symboles saisis seuls (« BTC ») convertis automatiquement en paire euro.
CRYPTOS_COURANTES = {
    "BTC", "ETH", "SOL", "XRP", "BNB", "ADA", "DOGE", "AVAX", "DOT", "LINK",
    "POL", "MATIC", "LTC", "TRX", "ATOM", "XLM", "UNI", "NEAR", "SHIB", "TON",
    "SUI", "APT", "ARB", "OP", "ETC", "BCH", "FIL", "ICP", "HBAR", "AAVE",
    "XMR", "PEPE", "RENDER", "INJ", "TAO",
}

TYPES_FR = {"EQUITY": "Action", "ETF": "ETF", "CRYPTOCURRENCY": "Crypto"}

# Poids de chaque pilier dans le score global, selon le type d'actif.
POIDS_PILIERS = {
    "EQUITY": {"Technique": 0.40, "Fondamental": 0.45, "Sentiment": 0.15},
    "ETF": {"Technique": 0.55, "Qualité du fonds": 0.30, "Sentiment": 0.15},
    "CRYPTOCURRENCY": {"Technique": 0.70, "Profil": 0.15, "Sentiment": 0.15},
}
POIDS_DEFAUT = {"Technique": 0.85, "Sentiment": 0.15}

# Score global minimal pour chaque orientation (en dessous du dernier : Vente).
SEUILS_ORIENTATION = [
    (40, "Achat"),
    (15, "Plutôt achat"),
    (-15, "Neutre"),
    (-40, "Plutôt vente"),
]


class DonneesIntrouvables(Exception):
    """Aucune donnée de cours pour ce ticker."""


@dataclass
class Critere:
    pilier: str
    nom: str
    valeur: str
    score: float  # de -2 à +2
    commentaire: str
    poids: float = 1.0  # 0 = critère purement informatif


@dataclass
class Analyse:
    ticker: str
    nom: str
    type_actif: str
    devise: str
    prix: float
    criteres: list[Critere]
    scores_piliers: dict[str, float]
    poids_piliers: dict[str, float]
    score_global: float
    orientation: str
    confiance: str
    historique: pd.DataFrame
    rsi_hebdo: pd.Series
    actualites: list[dict]
    performances: dict[str, float | None]
    avertissements: list[str] = field(default_factory=list)

    @property
    def type_fr(self) -> str:
        return TYPES_FR.get(self.type_actif, self.type_actif.title() or "Inconnu")

    def points_forts(self, n: int = 4) -> list[Critere]:
        cs = [c for c in self.criteres if c.poids > 0 and c.score > 0]
        return sorted(cs, key=lambda c: -c.score * c.poids)[:n]

    def points_faibles(self, n: int = 4) -> list[Critere]:
        cs = [c for c in self.criteres if c.poids > 0 and c.score < 0]
        return sorted(cs, key=lambda c: c.score * c.poids)[:n]


# --------------------------------------------------------------------------
# Outils
# --------------------------------------------------------------------------

def _num(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _pct(x: float, signe: bool = True) -> str:
    return f"{x * 100:+.1f} %" if signe else f"{x * 100:.1f} %"


def _palier(x: float, paliers: list[tuple[float, float]], defaut: float) -> float:
    """Renvoie la note du premier palier dont la borne est atteinte."""
    for borne, note in paliers:
        if x >= borne:
            return note
    return defaut


def _montant(x: float) -> str:
    if abs(x) >= 1e9:
        return f"{x / 1e9:,.1f} Md".replace(",", " ")
    if abs(x) >= 1e6:
        return f"{x / 1e6:,.0f} M".replace(",", " ")
    return f"{x:,.0f}".replace(",", " ")


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    perte = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rsi = 100 - 100 / (1 + gain / perte.replace(0, np.nan))
    return rsi.where(perte != 0, 100.0).where(gain.notna())


def _macd(s: pd.Series, rapide: int = 12, lent: int = 26, signal: int = 9):
    m = s.ewm(span=rapide, adjust=False).mean() - s.ewm(span=lent, adjust=False).mean()
    return m, m.ewm(span=signal, adjust=False).mean()


def _prix_il_y_a(close: pd.Series, **delta) -> float | None:
    cible = close.index[-1] - pd.DateOffset(**delta)
    if close.index[0] > cible:
        return None
    return float(close.loc[:cible].iloc[-1])


def _performances(close: pd.Series) -> dict[str, float | None]:
    dernier = float(close.iloc[-1])
    perf = {}
    for libelle, delta in (("1 mois", {"months": 1}), ("6 mois", {"months": 6}), ("1 an", {"years": 1})):
        p = _prix_il_y_a(close, **delta)
        perf[libelle] = dernier / p - 1 if p else None
    for annees in (3, 5):
        p = _prix_il_y_a(close, years=annees)
        perf[f"{annees} ans (par an)"] = (dernier / p) ** (1 / annees) - 1 if p else None
    return perf


def _volatilite(close: pd.Series, crypto: bool) -> float | None:
    recent = close.loc[close.index[-1] - pd.DateOffset(years=1):]
    if len(recent) < 30:
        return None
    rend = np.log(recent).diff().dropna()
    return float(rend.std() * math.sqrt(365 if crypto else 252))


def _recul_max(close: pd.Series, annees: int = 5) -> float:
    recent = close.loc[close.index[-1] - pd.DateOffset(years=annees):]
    return float((recent / recent.cummax() - 1).min())


# --------------------------------------------------------------------------
# Piliers d'analyse
# --------------------------------------------------------------------------

def analyse_technique(close: pd.Series, crypto: bool = False) -> list[Critere]:
    P = "Technique"
    c: list[Critere] = []
    prix = float(close.iloc[-1])

    if len(close) >= 200:
        mm50, mm200 = _sma(close, 50), _sma(close, 200)
        ecart = prix / mm200.iloc[-1] - 1
        if ecart > 0.40:
            s, com = 0.5, "Au-dessus de sa tendance de fond, mais très étiré : risque de correction à court terme."
        elif ecart > 0.03:
            s, com = 2, "Cours nettement au-dessus de sa moyenne 200 jours : tendance de fond haussière."
        elif ecart >= 0:
            s, com = 1, "Cours juste au-dessus de sa moyenne 200 jours : tendance haussière mais à surveiller."
        elif ecart > -0.03:
            s, com = -1, "Cours juste sous sa moyenne 200 jours : tendance de fond fragile."
        else:
            s, com = -2, "Cours nettement sous sa moyenne 200 jours : tendance de fond baissière."
        c.append(Critere(P, "Cours vs moyenne 200 jours", _pct(ecart), s, com, 1.5))

        haussier = mm50.iloc[-1] > mm200.iloc[-1]
        c.append(Critere(
            P, "Moyenne 50 j vs moyenne 200 j",
            "50 j au-dessus" if haussier else "50 j en dessous",
            1.5 if haussier else -1.5,
            "Configuration haussière (« golden cross »)." if haussier
            else "Configuration baissière (« death cross »).",
        ))

        mm200_ok = mm200.dropna()
        if len(mm200_ok) > 22:
            pente = mm200_ok.iloc[-1] / mm200_ok.iloc[-22] - 1
            s = 1 if pente > 0.01 else (-1 if pente < -0.01 else 0)
            com = {1: "La tendance de fond s'oriente à la hausse.",
                   -1: "La tendance de fond s'oriente à la baisse.",
                   0: "Tendance de fond à plat."}[s]
            c.append(Critere(P, "Pente de la moyenne 200 j (sur 1 mois)", _pct(pente), s, com))

    p12, p1 = _prix_il_y_a(close, months=12), _prix_il_y_a(close, months=1)
    if p12 and p1:
        mom = p1 / p12 - 1
        s = _palier(mom, [(0.20, 2), (0.05, 1), (-0.05, 0), (-0.20, -1)], -2)
        com = ("Forte dynamique sur un an : les tendances de ce type ont tendance à se prolonger." if s >= 1
               else "Dynamique annuelle neutre." if s == 0
               else "Dynamique annuelle négative.")
        c.append(Critere(P, "Momentum 12 mois (hors dernier mois)", _pct(mom), s, com, 1.5))

    hebdo = close.resample("W").last().dropna()
    if len(hebdo) >= 30:
        rsi = float(_rsi(hebdo).iloc[-1])
        if rsi > 75:
            s, com = -1, "Suracheté en hebdomadaire : mieux vaut attendre un repli avant d'entrer."
        elif rsi >= 60:
            s, com = 0.5, "Dynamique haussière saine."
        elif rsi >= 40:
            s, com = 0, "Zone neutre."
        elif rsi >= 30:
            s, com = -0.5, "Dynamique faible."
        else:
            s, com = 1, ("Survendu en hebdomadaire : point d'entrée potentiel sur le long terme, "
                         "à condition que le reste du dossier tienne.")
        c.append(Critere(P, "RSI hebdomadaire (14)", f"{rsi:.0f}", s, com))

        m, sig = _macd(hebdo)
        hausse = m.iloc[-1] > sig.iloc[-1]
        c.append(Critere(
            P, "MACD hebdomadaire", "au-dessus du signal" if hausse else "sous le signal",
            1 if hausse else -1,
            "Le momentum de moyen terme s'améliore." if hausse else "Le momentum de moyen terme se dégrade.",
        ))

    an = close.loc[close.index[-1] - pd.DateOffset(years=1):]
    recul = prix / float(an.max()) - 1
    c.append(Critere(P, "Recul depuis le plus haut sur 1 an", _pct(recul), 0,
                     "Information : distance au plus haut des 12 derniers mois.", 0))
    vol = _volatilite(close, crypto)
    if vol is not None:
        c.append(Critere(P, "Volatilité annualisée (1 an)", _pct(vol, False), 0,
                         "Information : amplitude habituelle des variations.", 0))
    return c


def analyse_fondamentale(info: dict) -> list[Critere]:
    P = "Fondamental"
    c: list[Critere] = []
    g = lambda k: _num(info.get(k))  # noqa: E731

    pe = g("forwardPE") or g("trailingPE")
    if pe is not None:
        libelle = "PER prévisionnel" if g("forwardPE") else "PER (12 derniers mois)"
        if pe <= 0:
            s, com = -2, "L'entreprise n'est pas bénéficiaire."
        else:
            s = _palier(-pe, [(-12, 2), (-18, 1), (-25, 0), (-40, -1)], -2)
            com = {2: "Valorisation basse par rapport aux bénéfices.",
                   1: "Valorisation raisonnable.",
                   0: "Valorisation dans la moyenne du marché.",
                   -1: "Valorisation élevée : beaucoup de croissance est déjà intégrée dans le prix.",
                   -2: "Valorisation très élevée."}[s]
        c.append(Critere(P, libelle, f"{pe:.1f}", s, com, 1.5))

    peg = g("trailingPegRatio") or g("pegRatio")
    if peg is not None and peg > 0:
        s = _palier(-peg, [(-1, 2), (-1.5, 1), (-2.5, 0)], -1)
        c.append(Critere(P, "PEG (PER rapporté à la croissance)", f"{peg:.2f}", s,
                         "Moins de 1 : croissance bon marché ; plus de 2,5 : croissance chère."))

    for cle, nom, poids in (("revenueGrowth", "Croissance du chiffre d'affaires", 1.5),
                            ("earningsGrowth", "Croissance des bénéfices", 1.0)):
        v = g(cle)
        if v is not None:
            s = _palier(v, [(0.15, 2), (0.05, 1), (0, 0), (-0.05, -1)], -2)
            c.append(Critere(P, nom + " (sur un an)", _pct(v), s,
                             "Croissance soutenue." if s >= 1 else "Croissance faible ou nulle." if s == 0
                             else "En recul.", poids))

    marge = g("profitMargins")
    if marge is not None:
        s = _palier(marge, [(0.20, 2), (0.10, 1), (0, 0)], -2)
        c.append(Critere(P, "Marge nette", _pct(marge, False), s,
                         "Très rentable." if s == 2 else "Rentable." if s == 1
                         else "Faiblement rentable." if s == 0 else "Déficitaire."))

    roe = g("returnOnEquity")
    if roe is not None:
        s = _palier(roe, [(0.20, 2), (0.10, 1), (0, 0)], -1)
        c.append(Critere(P, "Rentabilité des capitaux propres (ROE)", _pct(roe, False), s,
                         "Utilise très efficacement l'argent des actionnaires." if s == 2
                         else "Efficacité correcte." if s == 1 else "Efficacité faible."))

    secteur = str(info.get("sector") or "")
    dette = g("debtToEquity")
    if dette is not None and "Financial" not in secteur:  # non pertinent pour les banques
        s = _palier(-dette, [(-50, 1), (-100, 0.5), (-150, 0), (-250, -1)], -2)
        c.append(Critere(P, "Dette / capitaux propres", f"{dette:.0f} %", s,
                         "Endettement faible." if s >= 0.5 else "Endettement modéré." if s == 0
                         else "Endettement élevé : plus vulnérable si les taux montent ou l'activité ralentit."))

    fcf = g("freeCashflow")
    if fcf is not None:
        c.append(Critere(P, "Flux de trésorerie disponible", _montant(fcf), 1 if fcf > 0 else -1,
                         "Génère de la trésorerie." if fcf > 0 else "Consomme de la trésorerie."))

    reco, nb = g("recommendationMean"), g("numberOfAnalystOpinions") or 0
    if reco is not None and nb >= 3:
        s = _palier(-reco, [(-1.8, 1.5), (-2.5, 0.5), (-3.2, -0.5)], -1.5)
        c.append(Critere(P, "Consensus des analystes", f"{reco:.1f} / 5 ({int(nb)} analystes)", s,
                         "1 = achat fort, 5 = vente. À prendre avec recul : les analystes suivent souvent la tendance.",
                         0.5))

    cible, cours = g("targetMeanPrice"), g("currentPrice") or g("regularMarketPrice")
    if cible and cours and nb >= 3:
        pot = cible / cours - 1
        s = _palier(pot, [(0.20, 1.5), (0.05, 0.5), (-0.05, 0)], -1)
        c.append(Critere(P, "Potentiel vs objectif moyen des analystes", _pct(pot), s,
                         f"Objectif de cours moyen : {cible:,.2f}.".replace(",", " "), 0.5))
    return c


def analyse_etf(info: dict, close: pd.Series, perf: dict) -> list[Critere]:
    P = "Qualité du fonds"
    c: list[Critere] = []

    ter = _num(info.get("netExpenseRatio"))  # déjà en %
    if ter is None and _num(info.get("annualReportExpenseRatio")) is not None:
        ter = _num(info.get("annualReportExpenseRatio")) * 100
    if ter is not None:
        s = _palier(-ter, [(-0.20, 2), (-0.50, 1), (-1.0, 0)], -1)
        c.append(Critere(P, "Frais annuels (TER)", f"{ter:.2f} %", s,
                         "Frais très bas." if s == 2 else "Frais raisonnables." if s == 1
                         else "Frais élevés : ils rongent la performance sur le long terme."))

    actifs = _num(info.get("totalAssets"))
    if actifs:
        s = 1 if actifs >= 1e9 else (0 if actifs >= 1e8 else -1)
        c.append(Critere(P, "Encours du fonds", _montant(actifs), s,
                         "Fonds de grande taille, liquide." if s == 1 else "Taille correcte." if s == 0
                         else "Petit fonds : risque de fermeture et liquidité plus faible."))

    for annees, poids in ((5, 1.5), (3, 1.0)):
        v = perf.get(f"{annees} ans (par an)")
        if v is not None:
            s = _palier(v, [(0.10, 2), (0.05, 1), (0, 0)], -1)
            c.append(Critere(P, f"Performance annualisée sur {annees} ans", _pct(v), s,
                             "Très bonne performance passée." if s == 2 else "Performance correcte." if s == 1
                             else "Performance faible." if s == 0 else "Performance négative.", poids))

    c.append(Critere(P, "Pire baisse sur 5 ans", _pct(_recul_max(close), False), 0,
                     "Information : la plus forte chute subie, pour mesurer le risque.", 0))
    return c


def analyse_crypto(info: dict, close: pd.Series, perf: dict) -> list[Critere]:
    P = "Profil"
    c: list[Critere] = []

    mcap = _num(info.get("marketCap"))
    if mcap:
        if mcap >= 5e10:
            s, com = 1, "Crypto majeure, très établie."
        elif mcap >= 5e9:
            s, com = 0.5, "Crypto de taille importante."
        elif mcap >= 5e8:
            s, com = 0, "Capitalisation moyenne : plus risquée."
        else:
            s, com = -1.5, "Petite capitalisation : très spéculative."
        c.append(Critere(P, "Capitalisation", _montant(mcap), s, com, 1.5))

    vol = _volatilite(close, crypto=True)
    if vol is not None:
        s = _palier(-vol, [(-0.6, 1), (-0.9, 0)], -1)
        c.append(Critere(P, "Volatilité annualisée (1 an)", _pct(vol, False), s,
                         "Volatilité contenue pour une crypto." if s == 1 else "Volatilité habituelle." if s == 0
                         else "Très volatile : prévoir des variations extrêmes."))

    v = perf.get("3 ans (par an)")
    if v is not None:
        s = _palier(v, [(0.30, 2), (0.10, 1), (0, 0)], -1.5)
        c.append(Critere(P, "Performance annualisée sur 3 ans", _pct(v), s,
                         "A traversé un cycle complet en progressant." if s >= 1
                         else "Peu de progression sur un cycle." if s == 0 else "En perte sur 3 ans."))

    c.append(Critere(P, "Pire baisse sur 5 ans", _pct(_recul_max(close), False), 0,
                     "Information : la plus forte chute subie.", 0))
    return c


def _parser_actualites(brut) -> list[dict]:
    """Gère l'ancien et le nouveau format des actualités renvoyées par yfinance."""
    sortie = []
    for n in brut or []:
        if not isinstance(n, dict):
            continue
        ct = n.get("content") if isinstance(n.get("content"), dict) else n
        titre = ct.get("title")
        if not titre:
            continue
        prov = ct.get("provider")
        source = prov.get("displayName", "") if isinstance(prov, dict) else n.get("publisher", "")
        url = ct.get("canonicalUrl") or ct.get("clickThroughUrl")
        lien = url.get("url") if isinstance(url, dict) else n.get("link")
        date = ct.get("pubDate") or n.get("providerPublishTime")
        if isinstance(date, (int, float)):
            date = datetime.fromtimestamp(date, tz=timezone.utc).strftime("%Y-%m-%d")
        elif isinstance(date, str):
            date = date[:10]
        sortie.append({"titre": titre, "source": source, "lien": lien, "date": date or ""})
    return sortie


def analyse_sentiment(actualites: list[dict]) -> tuple[list[Critere], list[dict]]:
    if _VADER is None or not actualites:
        return [], actualites
    for a in actualites:
        a["sentiment"] = _VADER.polarity_scores(a["titre"])["compound"]
    moyenne = float(np.mean([a["sentiment"] for a in actualites]))
    s = _palier(moyenne, [(0.25, 2), (0.08, 1), (-0.08, 0), (-0.25, -1)], -2)
    com = {2: "Actualité très favorable.", 1: "Actualité plutôt favorable.", 0: "Actualité neutre.",
           -1: "Actualité plutôt défavorable.", -2: "Actualité très défavorable."}[s]
    return [Critere("Sentiment", f"Ton des {len(actualites)} dernières actualités", f"{moyenne:+.2f}", s, com)], actualites


# --------------------------------------------------------------------------
# Synthèse
# --------------------------------------------------------------------------

def _score_pilier(criteres: list[Critere]) -> float | None:
    w = sum(c.poids for c in criteres)
    if w == 0:
        return None
    return sum(c.score * c.poids for c in criteres) / (2 * w) * 100


def _orientation(score: float) -> str:
    for seuil, libelle in SEUILS_ORIENTATION:
        if score >= seuil:
            return libelle
    return "Vente"


def _confiance(criteres: list[Critere], scores: dict[str, float], score_global: float) -> str:
    nb = sum(1 for c in criteres if c.poids > 0)
    signes = {np.sign(v) for v in scores.values() if abs(v) >= 10}
    accord = len(signes) <= 1
    if nb >= 10 and accord and abs(score_global) >= 25:
        return "Élevée"
    if nb >= 6 and (accord or abs(score_global) >= 15):
        return "Moyenne"
    return "Faible"


# --------------------------------------------------------------------------
# Accès aux données
# --------------------------------------------------------------------------

# Noms courants -> ticker Yahoo. Permet de taper « airbus » ou « Nvidia » directement,
# sans dépendre de la recherche Yahoo (parfois bloquée sur les serveurs en ligne).
NOMS_COURANTS = {
    # France
    "airbus": "AIR.PA", "lvmh": "MC.PA", "totalenergies": "TTE.PA", "total": "TTE.PA",
    "loreal": "OR.PA", "l oreal": "OR.PA","hermes": "RMS.PA", "sanofi": "SAN.PA", "bnp": "BNP.PA",
    "bnp paribas": "BNP.PA", "axa": "CS.PA", "safran": "SAF.PA", "schneider": "SU.PA",
    "schneider electric": "SU.PA", "air liquide": "AI.PA", "kering": "KER.PA", "danone": "BN.PA",
    "vinci": "DG.PA", "societe generale": "GLE.PA", "credit agricole": "ACA.PA", "orange": "ORA.PA",
    "renault": "RNO.PA", "stellantis": "STLAP.PA", "michelin": "ML.PA", "capgemini": "CAP.PA",
    "dassault systemes": "DSY.PA", "thales": "HO.PA", "engie": "ENGI.PA", "carrefour": "CA.PA",
    "pernod ricard": "RI.PA", "essilorluxottica": "EL.PA", "saint gobain": "SGO.PA",
    "legrand": "LR.PA", "bouygues": "EN.PA", "veolia": "VIE.PA", "publicis": "PUB.PA",
    "ubisoft": "UBI.PA", "accor": "AC.PA", "alstom": "ALO.PA", "edenred": "EDEN.PA",
    "teleperformance": "TEP.PA", "stmicroelectronics": "STMPA.PA", "eurofins": "ERF.PA",
    "dassault aviation": "AM.PA",
    # Europe
    "siemens": "SIE.DE", "sap": "SAP.DE", "allianz": "ALV.DE", "volkswagen": "VOW3.DE",
    "bmw": "BMW.DE", "mercedes": "MBG.DE", "mercedes benz": "MBG.DE", "adidas": "ADS.DE",
    "bayer": "BAYN.DE", "basf": "BAS.DE", "deutsche bank": "DBK.DE", "rheinmetall": "RHM.DE",
    "infineon": "IFX.DE", "porsche": "P911.DE", "asml": "ASML.AS", "adyen": "ADYEN.AS",
    "ing": "INGA.AS", "philips": "PHIA.AS", "heineken": "HEIA.AS", "nestle": "NESN.SW",
    "novartis": "NOVN.SW", "roche": "ROG.SW", "novo nordisk": "NOVO-B.CO", "ferrari": "RACE.MI",
    "unilever": "ULVR.L", "shell": "SHEL.L", "hsbc": "HSBA.L", "astrazeneca": "AZN.L",
    "inditex": "ITX.MC", "santander": "SAN.MC",
    # États-Unis et Asie
    "apple": "AAPL", "microsoft": "MSFT", "nvidia": "NVDA", "amazon": "AMZN",
    "alphabet": "GOOGL", "google": "GOOGL", "meta": "META", "facebook": "META", "tesla": "TSLA",
    "netflix": "NFLX", "amd": "AMD", "intel": "INTC", "broadcom": "AVGO", "berkshire": "BRK-B",
    "berkshire hathaway": "BRK-B", "jpmorgan": "JPM", "visa": "V", "mastercard": "MA",
    "coca cola": "KO", "pepsico": "PEP", "mcdonalds": "MCD", "mcdonald s": "MCD", "mc donalds": "MCD","disney": "DIS", "nike": "NKE",
    "walmart": "WMT", "costco": "COST", "johnson & johnson": "JNJ", "pfizer": "PFE",
    "eli lilly": "LLY", "lilly": "LLY", "exxon": "XOM", "boeing": "BA", "palantir": "PLTR",
    "oracle": "ORCL", "salesforce": "CRM", "adobe": "ADBE", "paypal": "PYPL", "uber": "UBER",
    "airbnb": "ABNB", "coinbase": "COIN", "spotify": "SPOT", "ibm": "IBM", "qualcomm": "QCOM",
    "tsmc": "TSM", "alibaba": "BABA", "microstrategy": "MSTR", "starbucks": "SBUX",
    # ETF
    "msci world": "CW8.PA", "s&p 500": "ESE.PA", "sp500": "ESE.PA", "sp 500": "ESE.PA",
    "nasdaq": "PUST.PA", "nasdaq 100": "PUST.PA", "all world": "VWCE.DE",
    # Crypto
    "bitcoin": "BTC-EUR", "ethereum": "ETH-EUR", "ether": "ETH-EUR", "solana": "SOL-EUR",
    "ripple": "XRP-EUR", "cardano": "ADA-EUR", "dogecoin": "DOGE-EUR",
}


def _normaliser(texte: str) -> str:
    import re
    import unicodedata
    t = unicodedata.normalize("NFKD", texte).encode("ascii", "ignore").decode().lower()
    t = re.sub(r"[^a-z0-9& ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def resoudre_ticker(saisie: str) -> str:
    """Convertit la saisie en ticker Yahoo : nom courant, crypto seule ou ticker tel quel."""
    nom = _normaliser(saisie)
    if nom in NOMS_COURANTS:
        return NOMS_COURANTS[nom]
    s = saisie.strip().upper()
    return f"{s}-EUR" if s in CRYPTOS_COURANTES else s


def _suggestions_locales(texte: str, n: int = 5) -> list[dict]:
    import difflib
    nom = _normaliser(texte)
    if not nom:
        return []
    cles = [k for k in NOMS_COURANTS if nom in k or (len(k) >= 3 and k in nom)]
    cles += difflib.get_close_matches(nom, list(NOMS_COURANTS), n=n, cutoff=0.6)
    vus, sortie = set(), []
    for k in cles:
        t = NOMS_COURANTS[k]
        if t not in vus:
            vus.add(t)
            sortie.append({"symbole": t, "nom": k.title(), "type": "", "place": "suggestion"})
    return sortie[:n]


def rechercher(texte: str, n: int = 8) -> list[dict]:
    """Propose des tickers à partir d'un nom : liste locale, puis recherche Yahoo."""
    sortie = _suggestions_locales(texte)
    if yf is None:
        return sortie
    try:
        quotes = yf.Search(texte, max_results=n, news_count=0).quotes
    except Exception as e:
        print(f"[recherche] échec de la recherche Yahoo pour {texte!r} : {e}")
        return sortie
    deja = {s["symbole"] for s in sortie}
    for q in quotes or []:
        if q.get("symbol") in deja:
            continue
        typ = str(q.get("quoteType") or "").upper()
        if typ in TYPES_FR and q.get("symbol"):
            sortie.append({
                "symbole": q["symbol"],
                "nom": q.get("longname") or q.get("shortname") or q["symbol"],
                "type": TYPES_FR[typ],
                "place": q.get("exchDisp") or q.get("exchange", ""),
            })
    return sortie


def charger(ticker: str):
    """Récupère l'historique (10 ans), la fiche et les actualités d'une valeur."""
    if yf is None:
        raise RuntimeError("La bibliothèque yfinance n'est pas installée (pip install yfinance).")
    t = yf.Ticker(ticker)
    try:
        hist = t.history(period="10y", interval="1d", auto_adjust=True)
    except Exception as e:
        raise DonneesIntrouvables(str(e)) from e
    if hist is None or hist.empty or "Close" not in hist:
        raise DonneesIntrouvables(f"Aucun cours trouvé pour {ticker}.")
    try:
        info = t.info or {}
    except Exception:
        info = {}
    try:
        news = t.news or []
    except Exception:
        news = []
    return hist, info, news


def analyser(ticker: str, donnees=None) -> Analyse:
    """Analyse complète d'une valeur. `donnees` permet d'injecter (hist, info, news)."""
    hist, info, news = donnees if donnees is not None else charger(ticker)
    info = info or {}
    close = hist["Close"].dropna()
    close = close[close > 0]
    if close.empty:
        raise DonneesIntrouvables(f"Aucun cours exploitable pour {ticker}.")

    type_actif = str(info.get("quoteType") or "").upper()
    if not type_actif:
        type_actif = "CRYPTOCURRENCY" if ticker.upper().endswith(("-USD", "-EUR", "-USDT")) else "EQUITY"
    crypto = type_actif == "CRYPTOCURRENCY"

    perf = _performances(close)
    criteres = analyse_technique(close, crypto)
    avertissements: list[str] = []

    if type_actif == "EQUITY":
        fond = analyse_fondamentale(info)
        criteres += fond
        if sum(1 for c in fond if c.poids > 0) < 3:
            avertissements.append("Peu de données fondamentales disponibles : le score repose surtout "
                                  "sur l'analyse technique.")
    elif type_actif == "ETF":
        criteres += analyse_etf(info, close, perf)
    elif crypto:
        criteres += analyse_crypto(info, close, perf)
    else:
        avertissements.append(f"Type d'actif « {type_actif} » non spécifiquement géré : analyse technique uniquement.")

    crit_sent, actualites = analyse_sentiment(_parser_actualites(news))
    criteres += crit_sent

    if len(close) < 260:
        avertissements.append("Moins d'un an d'historique : plusieurs indicateurs de long terme sont indisponibles.")

    poids = POIDS_PILIERS.get(type_actif, POIDS_DEFAUT)
    scores: dict[str, float] = {}
    for pilier in poids:
        sp = _score_pilier([c for c in criteres if c.pilier == pilier])
        if sp is not None:
            scores[pilier] = sp
    total = sum(poids[p] for p in scores)
    poids_eff = {p: poids[p] / total for p in scores} if total else {}
    score_global = sum(scores[p] * poids_eff[p] for p in scores) if total else 0.0

    historique = hist.loc[close.index, [col for col in ("Open", "High", "Low", "Close", "Volume") if col in hist]].copy()
    historique["MM50"] = _sma(close, 50)
    historique["MM200"] = _sma(close, 200)

    return Analyse(
        ticker=ticker,
        nom=info.get("longName") or info.get("shortName") or ticker,
        type_actif=type_actif,
        devise=info.get("currency") or "",
        prix=float(close.iloc[-1]),
        criteres=criteres,
        scores_piliers=scores,
        poids_piliers=poids_eff,
        score_global=float(score_global),
        orientation=_orientation(score_global) if scores else "Données insuffisantes",
        confiance=_confiance(criteres, scores, score_global) if scores else "Aucune",
        historique=historique,
        rsi_hebdo=_rsi(close.resample("W").last().dropna()).dropna(),
        actualites=actualites,
        performances=perf,
        avertissements=avertissements,
    )
