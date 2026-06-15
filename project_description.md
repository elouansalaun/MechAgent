
# AI Mechanical Engineering Design Agent

Entrées :

- Coordonnées 3D des points de fixation et orientation de leurs axes.
- Conditions de chargement.
- Cahier des charges mécanique.
- Objectif d'optimisation : masse minimale, coût minimal, etc.

Sorties :

- Géométrie mécanique optimisée.
- Matériau sélectionné et justification.
- Résultats des simulations.
- Vérification des critères de conception.
- Rapport d'ingénierie détaillé.

L'agent doit être capable de prendre en charge l'ensemble de la chaîne de conception, depuis le cahier des charges jusqu'au rapport final.

## 1. Fonctionnement général

Je vois six grandes étapes dans le fonctionnement de l'agent.

1. Requirement Engineering

   L'agent interprète le cahier des charges, identifie les contraintes et transforme les informations en paramètres exploitables.
2. Mechanical Modeling

   Il construit une représentation paramétrique du composant à partir des points de fixation, axes, dimensions et zones de conception autorisées.
3. Simulation Strategy Selection

   Il détermine les analyses nécessaires : statique, fatigue, flambement, modal, etc. Il prépare les modèles et les conditions limites correspondantes.
4. Material Selection

   Il sélectionne les matériaux candidats en fonction des propriétés mécaniques, de la masse volumique, du prix et des conditions d'utilisation.
5. Design Optimization

   Il lance les optimisations géométriques, compare les résultats, modifie les paramètres et relance les simulations.
6. Engineering Validation & Reporting

   Il vérifie les critères, documente les hypothèses, justifie les décisions et génère un rapport final.

## 2. Comment l'agent prend ses décisions

C'est probablement la partie la plus intéressante du projet.

Il ne faut pas demander au LLM de calculer les contraintes ou de décider arbitrairement qu'une pièce est résistante. Il faut lui donner un ensemble d'outils d'ingénierie fiables.

Par exemple :

| Situation identifiée                | Analyse à lancer              |
| ----------------------------------- | ----------------------------- |
| Charge constante                    | Static FEM                    |
| Charge variable cyclique            | Fatigue analysis              |
| Compression importante              | Buckling analysis             |
| Vibrations ou excitation périodique | Modal analysis                |
| Charge variable dans le temps       | Transient structural analysis |
| Risque de déformation excessive     | Static FEM + déplacement      |

L'agent pourrait sélectionner plusieurs analyses simultanément.

Exemple :

Tu lui donnes :

- Force verticale de 2 000 N.
- Force appliquée cycliquement.
- Durée de vie souhaitée : 1 million de cycles.
- Déplacement maximal : 0,5 mm.
- Coefficient de sécurité minimal : 2.

L'agent pourrait décider :

1. De réaliser une analyse statique pour vérifier les contraintes et déplacements.
2. De réaliser une analyse de fatigue pour vérifier la durée de vie.
3. De comparer plusieurs matériaux.
4. D'optimiser la géométrie.
5. De refaire les simulations sur la géométrie optimisée.

Il doit également savoir qu'une analyse statique seule ne suffit pas à démontrer la tenue en fatigue.

## 3. Le véritable problème d'optimisation

Tu pourrais formaliser le problème mathématiquement.

Par exemple, pour une optimisation de masse :

\\[ \min\_{\mathbf{x},m} \quad M(\mathbf{x},m) \\]

Sous contraintes :

\\[ \sigma\_{\mathrm{VM}} \leq \frac{\sigma_y(m)}{SF} \\]

\\[ \delta\_{\max} \leq \delta\_{\mathrm{lim}} \\]

\\[ N\_{\mathrm{fatigue}} \geq N\_{\mathrm{required}} \\]

\\[ \mathbf{x}\in\mathcal{D} \\]

Où :

- \\(\mathbf{x}\\) : paramètres géométriques.
- \\(m\\) : matériau sélectionné.
- \\(M\\) : masse de la pièce.
- \\(\sigma\_{\mathrm{VM}}\\) : contrainte de Von Mises.
- \\(\sigma_y\\) : limite d'élasticité du matériau.
- \\(SF\\) : coefficient de sécurité.
- \\(\delta\_{\max}\\) : déplacement maximal.
- \\(\mathcal{D}\\) : domaine géométrique autorisé.

Pour l'optimisation du coût, tu remplaces la fonction objectif par :

\\[ \min\_{\mathbf{x},m} C(\mathbf{x},m) \\]

Tu pourrais même proposer plusieurs objectifs : masse minimale, coût minimal ou compromis masse/coût.

Point important : les points de fixation ne suffisent pas à définir une pièce unique. Il faudra également définir une enveloppe de conception, les zones interdites, les interfaces et les directions d'application des efforts. Ce sont des informations essentielles pour que l'optimisation ait un sens mécanique.

## 4. Architecture technique proposée

Je partirais sur une architecture modulaire, avec un agent orchestrateur et plusieurs outils spécialisés.

LangGraph – Engineering Agent

Orchestration, raisonnement, décisions et gestion des itérations

CAD Tool

CadQuery / FreeCAD

Génération géométrique

Material Tool

Python / Materials DB

Propriétés et coûts

FEM Tool

Gmsh / CalculiX

Calculs mécaniques

Fatigue Tool

Python / NumPy / SciPy

Durée de vie en fatigue

Optimization Engine

Optuna / SciPy / NSGA-II

Exploration des géométries et matériaux candidats

Engineering Report Generator

Rapport PDF, visualisations 3D, résultats FEM et justification des décisions

Quelques précisions techniques :

- CadQuery : pour créer des pièces paramétriques à partir de paramètres géométriques.
- Gmsh : pour générer les maillages.
- CalculiX : pour les analyses par éléments finis.
- Python : pour les calculs de fatigue (courbes S-N, règle de Miner, comptage rainflow).
- Optuna : pour optimiser les paramètres géométriques et comparer les matériaux.
- LangGraph : pour orchestrer l'ensemble du workflow.

Tout cela peut être réalisé avec des outils gratuits et open source.

## 5. Un exemple d'interface utilisateur

Je verrais bien une interface Streamlit dans laquelle l'utilisateur configure son problème.


Voici les informations que tu pourrais demander :

| Paramètre            | Exemple                   |
| -------------------- | ------------------------- |
| Component type       | Mechanical bracket        |
| Mounting points      | 3D coordinates + axes     |
| Load type            | Static / Cyclic           |
| Load magnitude       | 2,000 N                   |
| Required lifetime    | 1 million cycles          |
| Maximum displacement | 0.5 mm                    |
| Safety factor        | 2                         |
| Objective            | Minimize weight           |
| Material candidates  | Aluminum, Steel, Titanium |
| Manufacturing        | CNC machining             |

L'utilisateur pourrait aussi visualiser la géométrie finale, le maillage, les contraintes et les zones critiques.

## 6. Ce que je construirais en un mois

Attention à ne pas vouloir tout développer dès le départ. Je limiterais volontairement la première version à un composant mécanique de type support de fixation.

| Semaine | Développement                                                               |
| ------- | --------------------------------------------------------------------------- |
| 1       | Génération CAD paramétrique + FEM statique fonctionnelle                    |
| 2       | Intégration des matériaux, fatigue et optimisation                          |
| 3       | Développement de l'agent LangGraph et sélection automatique des simulations |
| 4       | Boucle d'optimisation autonome, interface et rapport PDF                    |

Pour la première version, je limiterais les analyses à la statique et à la fatigue. Le flambement et le modal pourraient venir ensuite.

Il faudra également prévoir une validation sur des cas analytiques simples et des résultats de référence : un agent qui produit des simulations ne prouve pas que celles-ci sont correctes. Il faudra vérifier les conditions limites, la convergence du maillage, les unités et les hypothèses de fatigue.

## 7. Ce qui rendrait le projet vraiment impressionnant

Je rajouterais trois fonctionnalités à terme :

1. Multi-objective optimization : permettre de comparer le poids, le coût et la durée de vie avec un front de Pareto.
2. Design alternatives : présenter plusieurs solutions et expliquer pourquoi l'une est préférable à l'autre.
3. Engineering decision trace : enregistrer chaque décision de l'agent, les simulations lancées, les résultats obtenus et les raisons d'une modification.

Le dernier point est particulièrement intéressant : tu pourrais montrer exactement pourquoi l'agent a choisi de passer d'un aluminium à un acier, ou pourquoi il a renforcé une zone particulière.

Mon conseil : ne développe pas un agent qui dessine librement des pièces dès le début. Développe un agent qui sait piloter un véritable processus de conception mécanique, avec des outils de calcul déterministes. C'est une démonstration beaucoup plus crédible de compétences en ingénierie et en AI Engineering.
