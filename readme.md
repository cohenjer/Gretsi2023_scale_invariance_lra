# Code de "Régularisation implicite des factorisations de faible rang pénalisées"

## Installation

1. Installer la bonne branche de tensorly
2. Installer les dépendances

#### Installation de tensorly

La version utilisée de tensorly est une version modifiée. Il ne faut pas utiliser pip, mais aller récupérer la bonne branche à cette addresse:
` https://github.com/cohenjer/tensorly/tree/sparse_hals ` commit `97a168a32b1e7fa48b66e9b8e3d8872e7d0deeaf`
ou bien le code en .zip référencé par le tag `Gretsi2023` à la même adresse.
Ensuite il faut installer le package localement avec
`pip install -e .` dans la racine du tensorly téléchargé.

#### Installation des dépendances

Installer les packages suivants:
- numpy
- tensorly-viz
- plotly
- shootout `https://github.com/cohenjer/shootout`

## Utilisation

Sont d'intérêt pour le papier:
- xp_gretsi.py, qui permet de faire tourner les simulations
- plot_xp_gretsi.py, qui permet de calculer les visualisations
- init_sparse.py, qui contient du code pour effectuer l'initialisation équilibrée.


