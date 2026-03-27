# 🏠 House Price Prediction — Advanced Regression

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-House%20Prices-20BEFF?logo=kaggle&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-22C55E)
![Score](https://img.shields.io/badge/R²-0.905-1D4ED8)

> **Projet Machine Learning End-to-End** — Prédiction du prix de vente de maisons à Ames, Iowa  
> Pipeline complet : EDA → Preprocessing → Feature Engineering → 5 Modèles → Optimisation

---

## 📋 Problématique

**Étant données les 80 caractéristiques d'une maison** (surface, qualité, localisation, âge, équipements...),  
prédire son **prix de vente final** avec la meilleure précision possible.

---

## 📊 Dataset

| Attribut | Valeur |
|----------|--------|
| **Source** | [Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) |
| **Auteur** | Dean De Cock (2011) |
| **Train** | 1 460 maisons × 81 colonnes |
| **Test** | 1 459 maisons × 80 colonnes |
| **Target** | `SalePrice` — prix de vente en $ |
| **Type** | Régression supervisée |

---

## 🏗️ Structure du Projet

```
house-price-prediction/
│
├── data/
│   ├── train.csv                    # Données d'entraînement (à télécharger)
│   ├── test.csv                     # Données de test (à télécharger)
│   ├── data_description.txt         # Description des 80 variables
│   ├── submission.csv               # Prédictions finales (généré)
│   └── README_data.md               # Instructions de téléchargement
│
├── notebooks/
│   ├── 01_EDA.ipynb                 # Analyse Exploratoire Complète
│   └── 02_Modeling.ipynb            # Preprocessing + Modélisation + Évaluation
│
├── src/
│   ├── preprocessing.py             # Pipeline de preprocessing modulaire
│   └── train.py                     # Script d'entraînement CLI
│
├── interface.html                   # Dashboard interactif (ouvrir dans navigateur)
├── Presentation.pptx                # Slides de soutenance (8 slides)
├── requirements.txt                 # Dépendances Python
├── .gitignore
└── README.md
```

---

## 🔬 Pipeline ML — 6 Étapes

### 1️⃣ Analyse Exploratoire (EDA)
- Distribution de `SalePrice` → skewed (1.88) → transformation `log1p` (0.12) ✅
- Analyse des **19 colonnes avec valeurs manquantes**
- Corrélations : `OverallQual` (r=0.79), `GrLivArea` (r=0.71), `GarageCars` (r=0.64)
- Boxplots et distributions des variables catégorielles (Neighborhood, qualités...)

### 2️⃣ Preprocessing
- **Valeurs manquantes contextuelles** : `PoolQC`/`Alley`/`Fence` → `'None'` (absence de feature)
- **Numériques** : imputation par médiane (robuste aux outliers)
- **Catégorielles** : imputation par mode

### 3️⃣ Feature Engineering — 11 Nouvelles Features
| Feature | Description | Importance |
|---------|-------------|-----------|
| `TotalSF` | `TotalBsmtSF + 1stFlrSF + 2ndFlrSF` | **#2** |
| `QualitySF` | `OverallQual × GrLivArea` | **#4** |
| `TotalBathrooms` | Somme pondérée des salles de bain | Top 10 |
| `HouseAge` | `YrSold − YearBuilt` | Top 10 |
| `RemodAge` | `YrSold − YearRemodAdd` | Top 15 |
| `HasPool`, `HasGarage`, `HasBasement` | Features binaires | - |
| `HasFireplace`, `Remodeled` | Features binaires | - |

### 4️⃣ Encodage
- **Ordinal** : qualités `Ex/Gd/TA/Fa/Po` → `5/4/3/2/1`
- **One-Hot Encoding** des variables nominales
- **StandardScaler** (centrage-réduction)

### 5️⃣ Modélisation — 5 Algorithmes

| Modèle | Famille | Justification | CV RMSE | R² |
|--------|---------|---------------|---------|-----|
| Ridge Regression | Linéaire L2 | Baseline, gère multicolinéarité | 0.135 | 0.865 |
| Lasso Regression | Linéaire L1 | Sélection automatique de features | 0.130 | 0.871 |
| Decision Tree | Arbre | Interprétable, non-linéaire | 0.201 | 0.762 |
| Random Forest | Ensemble Bagging | Robuste, peu de variance | 0.138 | 0.872 |
| **Gradient Boosting** ⭐ | **Ensemble Boosting** | **Haute performance** | **0.118** | **0.905** |

### 6️⃣ Optimisation & Évaluation
- **RandomizedSearchCV** : 40 itérations, 5-fold CV, 6 hyperparamètres
- **Feature Importance** : OverallQual (14.2%), TotalSF (11.8%), GrLivArea (9.8%)
- **Analyse des erreurs** : résidus normalement distribués, pas de pattern systématique

---

## 📈 Résultats Finaux

| Métrique | Valeur |
|----------|--------|
| **Meilleur modèle** | Gradient Boosting (optimisé) |
| **CV RMSE** | 0.118 (log-scale) |
| **R² Validation** | **0.905** |
| **Val RMSE** | 0.118 |
| **Erreur typique** | ≈ ±$12,000 sur prix médian $163K |

---

## ⚙️ Installation & Lancement

```bash
# 1. Cloner
git clone https://github.com/[username]/house-price-prediction.git
cd house-price-prediction

# 2. Environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Télécharger les données (voir data/README_data.md)

# 5. Lancer les notebooks
jupyter notebook notebooks/

# OU lancer le script d'entraînement directement
python src/train.py
```

**Ordre d'exécution des notebooks :**
1. `notebooks/01_EDA.ipynb` — Analyse exploratoire (exécuter toutes les cellules)
2. `notebooks/02_Modeling.ipynb` — Modélisation complète

**Interface interactive :**
```bash
# Ouvrir directement dans le navigateur
open interface.html    # Mac
start interface.html   # Windows
```

---

## 🛠️ Stack Technique

| Outil | Version | Usage |
|-------|---------|-------|
| Python | 3.10+ | Langage principal |
| Pandas | 2.0+ | Manipulation des données |
| NumPy | 1.24+ | Calcul numérique |
| Scikit-Learn | 1.3+ | Modèles ML et pipelines |
| Matplotlib | 3.7+ | Visualisations |
| Seaborn | 0.12+ | Visualisations statistiques |
| SciPy | 1.11+ | Tests statistiques |
| Jupyter | 7.0+ | Notebooks interactifs |

---

## 👤 Auteur

**[Votre Nom]**  
📧 [votre.email@example.com]  
🔗 [LinkedIn](https://linkedin.com/in/votre-profil)  
🐙 [GitHub](https://github.com/votre-username)

---

*Projet réalisé dans le cadre du cours Machine Learning — [Votre Institution] — Avril 2026*  
*Dataset source : Dean De Cock, "Ames, Iowa: Alternative to the Boston Housing Data", Journal of Statistics Education, 2011*
