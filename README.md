# Analyseur de marchés

Application locale qui analyse une action, un ETF ou une cryptomonnaie et donne une **orientation de moyen / long terme** (Achat, Plutôt achat, Neutre, Plutôt vente, Vente), avec le détail du raisonnement.

## Installation (une seule fois)

1. Installe **Python 3.10 ou plus récent** : https://www.python.org/downloads/ (sous Windows, coche « Add Python to PATH »).
2. Mets `app.py`, `analyse.py`, `court_terme.py` et `requirements.txt` dans un même dossier.
3. Ouvre un terminal dans ce dossier et tape :

```bash
pip install -r requirements.txt
```

## Lancement

```bash
streamlit run app.py
```

L'application s'ouvre dans ton navigateur (adresse `http://localhost:8501`). Pour l'arrêter, fais `Ctrl + C` dans le terminal.

## Utilisation

- **Analyser une valeur** : tape un ticker et appuie sur Entrée.
  - Paris : `AIR.PA`, `MC.PA`, `TTE.PA` · Amsterdam : `.AS` · Francfort : `.DE` · Milan : `.MI`
  - États-Unis : `AAPL`, `MSFT`, `NVDA`
  - ETF : `CW8.PA`, `EWLD.PA`, `IWDA.AS`
  - Crypto : `BTC`, `ETH`, `SOL`… (convertis automatiquement en paire euro, ex. `BTC-EUR`)
  - Tu peux aussi taper un nom (« airbus ») : l'application te propose les tickers correspondants.
- **Comparer plusieurs valeurs** : liste plusieurs tickers et obtiens un tableau classé par score.

## Comment le score est calculé

Chaque critère reçoit une note de -2 à +2. Les notes sont regroupées par pilier (score de -100 à +100), puis pondérées :

| Type   | Technique | Fondamental / fonds / profil | Sentiment |
|--------|-----------|------------------------------|-----------|
| Action | 40 %      | 45 % (fondamental)           | 15 %      |
| ETF    | 55 %      | 30 % (qualité du fonds)      | 15 %      |
| Crypto | 70 %      | 15 % (profil)                | 15 %      |

- **Technique** : cours par rapport à la moyenne 200 jours, croisement 50/200 jours, pente de la tendance, momentum 12 mois, RSI et MACD hebdomadaires.
- **Fondamental** : PER, PEG, croissance du chiffre d'affaires et des bénéfices, marge nette, ROE, dette, flux de trésorerie, consensus et objectifs des analystes.
- **Qualité du fonds** : frais (TER), encours, performance annualisée sur 3 et 5 ans.
- **Profil crypto** : capitalisation, volatilité, performance sur 3 ans.
- **Sentiment** : ton des titres des dernières actualités.

Si un pilier n'a pas de données, son poids est redistribué sur les autres. Tous les seuils et les poids sont réglables en haut de `analyse.py`.

## Vue court terme (1 semaine, 1 mois, 3 mois)

Chaque analyse a un second onglet, « Court terme et probabilités » :

1. **Indicateurs quotidiens** : RSI 14 jours, MACD, bandes de Bollinger, cours par rapport aux moyennes 20 et 50 jours, ADX (force de la tendance), volume relatif, ATR (variation typique d'une séance).
2. **Probabilités et fourchettes** : pour chaque horizon, la part des périodes passées qui ont fini en hausse, la variation médiane, et la fourchette où le cours a 2 chances sur 3 (et 9 sur 10) de se trouver. Les fourchettes sont tirées d'une simulation qui rejoue les variations des 3 dernières années, ajustées à la volatilité actuelle, sans supposer de tendance.
3. **Situations comparables** : pour chaque situation vraie aujourd'hui (RSI survendu, cours sous la moyenne 50 jours, forte baisse sur un mois…), ce qui s'est passé ensuite dans le passé, avec le nombre de cas indépendants et un test qui indique si l'écart avec la normale est significatif ou peut s'expliquer par le hasard.
4. **Objectif et stop** : tu indiques un objectif de hausse, un stop et un horizon, et l'application estime la probabilité d'atteindre l'un ou l'autre en premier.

Le seuil de significativité a été calibré sur des marchés simulés au hasard : moins de 1 % des tests y ressortent « significatifs » à tort. La plupart du temps, aucune situation ne se distinguera du hasard : c'est normal sur le court terme.

## Limites

- **Ce n'est pas un conseil en investissement.** Les indicateurs résument le passé et la situation actuelle ; ils ne prédisent pas l'avenir et peuvent se tromper longtemps.
- Les données viennent de **Yahoo Finance** (gratuit, non officiel) : elles peuvent être incomplètes, surtout pour les ETF et les petites valeurs européennes, et Yahoo limite parfois le nombre de requêtes. Les résultats sont mis en cache une heure.
- L'analyse du sentiment fonctionne sur des titres en anglais et reste rudimentaire.
- Une même grille s'applique à tous les secteurs : un PER de 30 n'a pas le même sens pour une banque et pour une entreprise tech.
