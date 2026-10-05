"""
Analyseur de marchés : interface Streamlit.

Lancement :  streamlit run app.py
"""
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import analyse as moteur
import court_terme as ct

st.set_page_config(page_title="Analyseur de marchés", page_icon="📈", layout="wide")

COULEURS = {
    "Achat": "#1a9850",
    "Plutôt achat": "#66bd63",
    "Neutre": "#8c8c8c",
    "Plutôt vente": "#f46d43",
    "Vente": "#d73027",
    "Données insuffisantes": "#8c8c8c",
}


# --------------------------------------------------------------------------
# Données (mises en cache 1 h pour ne pas surcharger Yahoo Finance)
# --------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def analyser_cache(ticker: str):
    """Renvoie None si la valeur est introuvable (résultat mis en cache aussi,
    pour ne pas réinterroger Yahoo à chaque clic)."""
    try:
        return moteur.analyser(ticker)
    except moteur.DonneesIntrouvables:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def court_terme_cache(ticker: str):
    res = analyser_cache(ticker)
    return ct.analyser_court_terme(res.historique, crypto=res.type_actif == "CRYPTOCURRENCY")


@st.cache_data(ttl=86400, show_spinner=False)
def rechercher_cache(texte: str):
    return moteur.rechercher(texte)


def fmt_pct(x):
    return "—" if x is None else f"{x * 100:+.1f} %"


def badge(orientation: str) -> str:
    c = COULEURS.get(orientation, "#8c8c8c")
    return (f"<span style='background:{c};color:white;padding:6px 16px;border-radius:999px;"
            f"font-weight:600;font-size:1.15rem'>{orientation}</span>")


# --------------------------------------------------------------------------
# Graphiques
# --------------------------------------------------------------------------

def jauge(score: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score),
        number={"suffix": " / 100", "valueformat": "+d"},
        gauge={
            "axis": {"range": [-100, 100], "tickvals": [-100, -40, -15, 15, 40, 100]},
            "bar": {"color": "rgba(40,40,40,0.85)", "thickness": 0.25},
            "steps": [
                {"range": [-100, -40], "color": COULEURS["Vente"]},
                {"range": [-40, -15], "color": COULEURS["Plutôt vente"]},
                {"range": [-15, 15], "color": "#d9d9d9"},
                {"range": [15, 40], "color": COULEURS["Plutôt achat"]},
                {"range": [40, 100], "color": COULEURS["Achat"]},
            ],
        },
    ))
    fig.update_layout(height=230, margin=dict(l=25, r=25, t=20, b=10))
    return fig


def graphe_piliers(res) -> go.Figure:
    piliers = list(res.scores_piliers)
    valeurs = [res.scores_piliers[p] for p in piliers]
    fig = go.Figure(go.Bar(
        x=valeurs,
        y=[f"{p} ({res.poids_piliers[p] * 100:.0f} %)" for p in piliers],
        orientation="h",
        marker_color=[COULEURS["Plutôt achat"] if v >= 0 else COULEURS["Plutôt vente"] for v in valeurs],
        text=[f"{v:+.0f}" for v in valeurs],
        textposition="outside",
        cliponaxis=False,
    ))
    fig.update_xaxes(range=[-115, 115], zeroline=True, zerolinewidth=2)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=80 + 45 * max(len(piliers), 1), margin=dict(l=10, r=10, t=30, b=10),
                      title=dict(text="Score par pilier (poids)", font=dict(size=14)))
    return fig


def graphe_cours(res) -> go.Figure:
    h = res.historique
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=h.index, y=h["Close"], name="Cours", line=dict(width=1.6)), 1, 1)
    fig.add_trace(go.Scatter(x=h.index, y=h["MM50"], name="Moyenne 50 j", line=dict(width=1.2, dash="dot")), 1, 1)
    fig.add_trace(go.Scatter(x=h.index, y=h["MM200"], name="Moyenne 200 j", line=dict(width=1.6, dash="dash")), 1, 1)
    fig.add_trace(go.Scatter(x=res.rsi_hebdo.index, y=res.rsi_hebdo, name="RSI hebdo",
                             line=dict(width=1.2)), 2, 1)
    for niveau in (30, 70):
        fig.add_hline(y=niveau, line_dash="dot", line_color="gray", row=2, col=1)
    fig.update_yaxes(title_text=res.devise or "Cours", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    debut = max(h.index[0], h.index[-1] - pd.DateOffset(years=5))
    fig.update_xaxes(range=[debut, h.index[-1]])
    fig.update_layout(
        height=540,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08, x=0),
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis=dict(rangeselector=dict(buttons=[
            dict(count=6, label="6 mois", step="month", stepmode="backward"),
            dict(count=1, label="1 an", step="year", stepmode="backward"),
            dict(count=3, label="3 ans", step="year", stepmode="backward"),
            dict(count=5, label="5 ans", step="year", stepmode="backward"),
            dict(step="all", label="Tout"),
        ])),
    )
    return fig


# --------------------------------------------------------------------------
# Affichage d'une analyse
# --------------------------------------------------------------------------

def afficher_analyse(res):
    st.subheader(f"{res.nom}  ·  {res.ticker}")
    st.caption(f"{res.type_fr} · Dernier cours : {res.prix:,.2f} {res.devise}".replace(",", " "))
    for a in res.avertissements:
        st.warning(a)

    vue_lt, vue_ct = st.tabs(["📅 Moyen / long terme", "⚡ Court terme et probabilités"])
    with vue_lt:
        bloc_long_terme(res)
    with vue_ct:
        bloc_court_terme(res)


def bloc_long_terme(res):
    c1, c2, c3 = st.columns([1.1, 1, 1.2])
    with c1:
        st.markdown("**Orientation moyen / long terme**")
        st.markdown(badge(res.orientation), unsafe_allow_html=True)
        st.markdown(f"<div style='margin-top:14px'>Confiance : <b>{res.confiance}</b></div>",
                    unsafe_allow_html=True)
        st.caption("La confiance dépend du nombre d'indicateurs disponibles et de l'accord "
                   "entre les piliers (technique, fondamental, actualité).")
    with c2:
        st.plotly_chart(jauge(res.score_global), width="stretch")
    with c3:
        if res.scores_piliers:
            st.plotly_chart(graphe_piliers(res), width="stretch")

    cols = st.columns(len(res.performances))
    for col, (lib, val) in zip(cols, res.performances.items()):
        col.metric(f"Performance {lib}", fmt_pct(val))

    gauche, droite = st.columns(2)
    with gauche:
        st.markdown("##### ✅ Points forts")
        forts = res.points_forts()
        if not forts:
            st.write("Aucun point fort marquant.")
        for c in forts:
            st.markdown(f"- **{c.nom}** ({c.valeur}) : {c.commentaire}")
    with droite:
        st.markdown("##### ⚠️ Points faibles")
        faibles = res.points_faibles()
        if not faibles:
            st.write("Aucun point faible marquant.")
        for c in faibles:
            st.markdown(f"- **{c.nom}** ({c.valeur}) : {c.commentaire}")

    st.plotly_chart(graphe_cours(res), width="stretch")

    st.markdown("#### Détail des critères")
    piliers = list(dict.fromkeys(c.pilier for c in res.criteres))
    for pilier in piliers:
        score = res.scores_piliers.get(pilier)
        titre = f"{pilier} : score {score:+.0f} / 100" if score is not None else pilier
        with st.expander(titre, expanded=False):
            df = pd.DataFrame([{
                "Critère": c.nom,
                "Valeur": c.valeur,
                "Note (-2 à +2)": f"{c.score:+.1f}" if c.poids > 0 else "info",
                "Lecture": c.commentaire,
            } for c in res.criteres if c.pilier == pilier])
            st.dataframe(df, hide_index=True, width="stretch")

    if res.actualites:
        with st.expander(f"Dernières actualités ({len(res.actualites)})"):
            for a in res.actualites:
                s = a.get("sentiment")
                icone = "⚪" if s is None else ("🟢" if s > 0.05 else "🔴" if s < -0.05 else "⚪")
                titre = f"[{a['titre']}]({a['lien']})" if a.get("lien") else a["titre"]
                st.markdown(f"{icone} {titre}  \n<small>{a.get('source', '')} · {a.get('date', '')}</small>",
                            unsafe_allow_html=True)
            st.caption("Le ton est mesuré automatiquement sur les titres (en anglais) : indicatif seulement.")


# --------------------------------------------------------------------------
# Vue court terme
# --------------------------------------------------------------------------

SYMBOLES = {1: "🟢", -1: "🔴", 0: "⚪"}


def fmt_prix(x: float, devise: str = "") -> str:
    return f"{x:,.2f}".replace(",", " ") + (f" {devise}" if devise else "")


def _cellule(s) -> str:
    if s.proba_hausse is None:
        return "—"
    icone = {"Écart significatif": "✅", "Trop peu de cas": "❔"}.get(s.verdict, "➖")
    return f"{icone} {s.proba_hausse * 100:.0f} % ({s.ecart:+.0f} pts) · {s.cas} cas"


def _fourchette(f_bas: float, f_haut: float, prix: float, devise: str) -> str:
    return (f"{fmt_prix(f_bas)} → {fmt_prix(f_haut, devise)}  "
            f"({(f_bas / prix - 1) * 100:+.0f} % / {(f_haut / prix - 1) * 100:+.0f} %)")


def graphe_eventail(res, c) -> go.Figure:
    h = res.historique["Close"]
    passe = h.loc[h.index[-1] - pd.DateOffset(months=6):]
    e = c.eventail
    bleu = "#1f77b4"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=passe.index, y=passe, name="Cours", line=dict(width=1.8, color=bleu)))
    fig.add_trace(go.Scatter(x=e.index, y=e["p95"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=e.index, y=e["p5"], fill="tonexty", fillcolor="rgba(31,119,180,0.12)",
                             line=dict(width=0), name="9 chances sur 10"))
    fig.add_trace(go.Scatter(x=e.index, y=e["p84"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=e.index, y=e["p16"], fill="tonexty", fillcolor="rgba(31,119,180,0.28)",
                             line=dict(width=0), name="2 chances sur 3"))
    fig.add_trace(go.Scatter(x=e.index, y=e["p50"], name="Médiane", line=dict(dash="dot", color=bleu)))
    fig.update_layout(height=380, hovermode="x unified", legend=dict(orientation="h", y=1.12, x=0),
                      margin=dict(l=10, r=10, t=40, b=10), yaxis_title=res.devise or "Cours")
    return fig


def bloc_court_terme(res):
    try:
        c = court_terme_cache(res.ticker)
    except Exception as e:
        st.error(f"Analyse court terme impossible : {e}")
        return

    st.info("Sur le court terme, les marchés sont proches du hasard : une probabilité de hausse de 55 % "
            "est déjà un avantage rare. Chaque chiffre ci-dessous indique sur combien de cas il repose.")

    # 1. Indicateurs
    st.markdown("#### Indicateurs court terme")
    n_h, n_b, n_n = c.resume_signaux
    m1, m2, m3 = st.columns(3)
    m1.metric("🟢 Haussiers", n_h)
    m2.metric("🔴 Baissiers", n_b)
    m3.metric("⚪ Neutres ou informatifs", n_n)
    st.dataframe(pd.DataFrame([{
        "": SYMBOLES[i["Signal"]], "Indicateur": i["Indicateur"], "Valeur": i["Valeur"], "Lecture": i["Lecture"],
    } for i in c.indicateurs]), hide_index=True, width="stretch")

    # 2. Probabilités et fourchettes
    st.markdown("#### Probabilités et fourchettes par horizon")
    colonnes = {}
    for hz in ct.HORIZONS:
        b, f = c.base[hz], c.fourchettes.get(hz)
        colonnes[hz] = {
            "Périodes en hausse dans le passé": (f"{b.proba_hausse * 100:.0f} % ({b.cas} cas)"
                                                 if b.proba_hausse is not None else "—"),
            "Variation médiane passée": (f"{b.rendement_median * 100:+.1f} %"
                                         if b.rendement_median is not None else "—"),
            "Fourchette probable (2 chances sur 3)": _fourchette(f[1], f[3], c.prix, res.devise) if f else "—",
            "Fourchette large (9 chances sur 10)": _fourchette(f[0], f[4], c.prix, res.devise) if f else "—",
        }
    st.dataframe(pd.DataFrame(colonnes), width="stretch")
    st.caption("« Périodes en hausse » : part des périodes de cette durée qui ont fini en hausse sur tout "
               "l'historique disponible (10 ans au plus). Les fourchettes viennent d'une simulation qui rejoue "
               "au hasard les variations des 3 dernières années, ajustées à la volatilité actuelle, "
               "sans supposer de tendance.")
    if c.eventail is not None:
        st.plotly_chart(graphe_eventail(res, c), width="stretch")
    if c.vol_annuelle:
        st.caption(f"Volatilité actuelle : environ {c.vol_annuelle * 100:.0f} % par an.")

    # 3. Situations comparables
    st.markdown("#### Que s'est-il passé dans des situations comparables ?")
    if not c.situations:
        st.write("Aucune des situations testées n'est active aujourd'hui.")
    else:
        table = pd.DataFrame({nom: {hz: _cellule(s) for hz, s in d.items()}
                              for nom, d in c.situations.items()}).T
        table.index.name = "Situation actuelle"
        st.dataframe(table, width="stretch")
        st.caption("Chaque case : part des fois où la valeur a monté après cette situation, écart avec la "
                   "normale (en points), nombre de cas indépendants. ✅ écart significatif · "
                   "➖ pas d'écart fiable avec le hasard · ❔ trop peu de cas pour conclure. "
                   "Un écart passé, même significatif, peut disparaître si le marché change.")
        nb_sig = sum(s.verdict == "Écart significatif" for d in c.situations.values() for s in d.values())
        if nb_sig == 0:
            st.markdown("**Aucune situation ne se distingue du hasard pour cette valeur aujourd'hui.** "
                        "C'est le cas le plus fréquent, et c'est une information en soi.")

    # 4. Objectif et stop
    st.markdown("#### Objectif et stop")
    if c.chemins is None:
        st.write("Historique trop court pour simuler.")
        return
    a, b, d = st.columns(3)
    objectif = a.number_input("Objectif de hausse (%)", 1.0, 300.0, 10.0, 1.0, key=f"obj_{res.ticker}")
    stop = b.number_input("Stop de perte (%)", 1.0, 90.0, 5.0, 1.0, key=f"stop_{res.ticker}")
    hz = d.selectbox("Horizon", list(ct.HORIZONS), index=1, key=f"hz_{res.ticker}")
    p = ct.proba_objectif_stop(c.chemins, ct._lignes(hz, c.crypto), c.prix, objectif / 100, stop / 100)
    x1, x2, x3 = st.columns(3)
    x1.metric(f"Objectif atteint en premier ({fmt_prix(c.prix * (1 + objectif / 100))})", f"{p['objectif'] * 100:.0f} %")
    x2.metric(f"Stop touché en premier ({fmt_prix(c.prix * (1 - stop / 100))})", f"{p['stop'] * 100:.0f} %")
    x3.metric("Ni l'un ni l'autre", f"{p['aucun'] * 100:.0f} %")
    st.caption("Estimé sur 4 000 trajectoires simulées, à partir des cours de clôture (les mouvements en "
               "cours de séance ne sont pas comptés). Un objectif plus éloigné que le stop sera "
               "naturellement atteint moins souvent : c'est le rapport gain/risque qui compte, "
               "pas seulement la probabilité.")


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("Comment ça marche")
    st.markdown(
        "L'application récupère 10 ans de cours, la fiche de la valeur et ses dernières "
        "actualités sur **Yahoo Finance**, puis note une quinzaine de critères de **-2 à +2**.\n\n"
        "- **Technique** : tendance (moyennes 50 et 200 jours), momentum 12 mois, RSI et MACD hebdomadaires.\n"
        "- **Fondamental** (actions) : valorisation, croissance, marges, dette, trésorerie, analystes.\n"
        "- **Qualité du fonds** (ETF) : frais, encours, performance sur 3 et 5 ans.\n"
        "- **Profil** (crypto) : capitalisation, volatilité, performance sur 3 ans.\n"
        "- **Sentiment** : ton des dernières actualités.\n"
        "- **Court terme** (1 semaine, 1 mois, 3 mois) : indicateurs quotidiens, probabilités "
        "historiques, fourchettes probables et calcul objectif / stop.\n\n"
        "Le score global va de **-100 à +100** :\n\n"
        "| Score | Orientation |\n|---|---|\n"
        "| ≥ 40 | Achat |\n| 15 à 40 | Plutôt achat |\n| -15 à 15 | Neutre |\n"
        "| -40 à -15 | Plutôt vente |\n| < -40 | Vente |"
    )
    st.markdown("**Exemples de tickers**\n\n"
                "- Paris : `AIR.PA`, `MC.PA`, `TTE.PA`\n"
                "- États-Unis : `AAPL`, `MSFT`, `NVDA`\n"
                "- ETF : `CW8.PA`, `EWLD.PA`, `IWDA.AS`\n"
                "- Crypto : `BTC`, `ETH`, `SOL`\n\n"
                "Tu peux aussi taper un nom (« airbus »).")
    st.divider()
    st.caption("⚠️ Outil d'aide à la réflexion, pas un conseil en investissement. Les indicateurs "
               "décrivent le passé et la situation actuelle ; ils ne prédisent pas l'avenir. "
               "Les données Yahoo Finance peuvent être incomplètes ou en retard.")

st.title("📈 Analyseur de marchés")
st.caption("Orientation moyen / long terme pour actions, ETF et cryptomonnaies")

onglet_un, onglet_comp = st.tabs(["Analyser une valeur", "Comparer plusieurs valeurs"])

with onglet_un:
    saisie = st.text_input(
        "Valeur à analyser",
        placeholder="Ticker ou nom : AIR.PA, AAPL, CW8.PA, BTC, airbus…",
        help="Appuie sur Entrée pour lancer l'analyse.",
    )
    if saisie.strip():
        ticker = moteur.resoudre_ticker(saisie)
        res, erreur = None, None
        if ticker != saisie.strip().upper():
            st.caption(f"« {saisie.strip()} » interprété comme **{ticker}**")
        with st.spinner(f"Analyse de {ticker} en cours…"):
            try:
                res = analyser_cache(ticker)
            except Exception as e:  # réseau, limite de requêtes Yahoo…
                erreur = e

        if erreur is not None:
            st.error(f"Impossible de récupérer les données ({erreur}). Réessaie dans quelques minutes.")
        elif res is None:
            propositions = rechercher_cache(saisie.strip())
            if propositions:
                par_symbole = {p["symbole"]: p for p in propositions}
                choix = st.selectbox(
                    "Je n'ai pas trouvé cette valeur telle quelle. Tu voulais dire :",
                    list(par_symbole),
                    format_func=lambda s: " · ".join(
                        x for x in (f"{par_symbole[s]['nom']} ({s})", par_symbole[s]["type"],
                                    par_symbole[s]["place"]) if x),
                )
                with st.spinner(f"Analyse de {choix} en cours…"):
                    try:
                        res = analyser_cache(choix)
                    except Exception as e:
                        st.error(f"Analyse impossible pour {choix} : {e}")
                if res is None:
                    st.error(f"Aucune donnée trouvée pour {choix}.")
            else:
                st.error("Valeur introuvable. Essaie avec son ticker : AIR.PA (Airbus), NVDA (Nvidia), "
                         "SIE.DE (Siemens), MC.PA (LVMH)… Tu le trouves en cherchant le nom sur "
                         "fr.finance.yahoo.com.")
        if res is not None:
            afficher_analyse(res)

with onglet_comp:
    texte = st.text_area(
        "Valeurs à comparer (séparées par des virgules ou des retours à la ligne)",
        "AAPL, MSFT, AIR.PA, CW8.PA, BTC, ETH",
        height=90,
    )
    if st.button("Comparer", type="primary"):
        tickers = list(dict.fromkeys(
            moteur.resoudre_ticker(t) for t in re.split(r"[,;\n]+", texte) if t.strip()
        ))
        lignes, erreurs = [], []
        barre = st.progress(0.0, text="Analyse en cours…")
        for i, t in enumerate(tickers):
            barre.progress(i / len(tickers), text=f"Analyse de {t}… ({i + 1} / {len(tickers)})")
            try:
                r = analyser_cache(t)
                if r is None:
                    erreurs.append(t)
                    continue
                sp = r.scores_piliers
                lignes.append({
                    "Valeur": r.nom,
                    "Ticker": r.ticker,
                    "Type": r.type_fr,
                    "Orientation": r.orientation,
                    "Score": r.score_global,
                    "Confiance": r.confiance,
                    "Technique": sp.get("Technique"),
                    "Fondamental / fonds / profil": sp.get("Fondamental", sp.get("Qualité du fonds", sp.get("Profil"))),
                    "Sentiment": sp.get("Sentiment"),
                    "Perf. 1 an": r.performances.get("1 an"),
                    "Perf. 5 ans (par an)": r.performances.get("5 ans (par an)"),
                })
            except Exception:
                erreurs.append(t)
        barre.empty()
        st.session_state["comparaison"] = (lignes, erreurs)

    if "comparaison" in st.session_state:
        lignes, erreurs = st.session_state["comparaison"]
        if lignes:
            df = pd.DataFrame(lignes).sort_values("Score", ascending=False)
            style = (
                df.style
                .map(lambda v: f"background-color:{COULEURS[v]};color:white;font-weight:600"
                     if v in COULEURS else "", subset=["Orientation"])
                .format({
                    "Score": "{:+.0f}",
                    "Technique": "{:+.0f}",
                    "Fondamental / fonds / profil": "{:+.0f}",
                    "Sentiment": "{:+.0f}",
                    "Perf. 1 an": "{:+.1%}",
                    "Perf. 5 ans (par an)": "{:+.1%}",
                }, na_rep="—")
            )
            st.dataframe(style, hide_index=True, width="stretch")
            st.caption("Les scores vont de -100 à +100. Ouvre l'onglet « Analyser une valeur » pour le détail.")
        if erreurs:
            st.warning("Introuvables : " + ", ".join(erreurs))
