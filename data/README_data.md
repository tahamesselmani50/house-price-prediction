# 📦 Dataset — Instructions de Téléchargement

Les fichiers CSV ne sont pas inclus dans ce repository (taille > 400KB).

## Option 1 — Téléchargement Manuel

1. Créer un compte sur [Kaggle](https://www.kaggle.com)
2. Accepter les règles : https://www.kaggle.com/c/house-prices-advanced-regression-techniques
3. Télécharger et placer dans ce dossier `data/` :
   - `train.csv`
   - `test.csv`

## Option 2 — Kaggle API

```bash
pip install kaggle
# Placer kaggle.json dans ~/.kaggle/
kaggle competitions download -c house-prices-advanced-regression-techniques
unzip house-prices-advanced-regression-techniques.zip -d data/
```

## Structure attendue

```
data/
├── train.csv               # 1,460 × 81
├── test.csv                # 1,459 × 80
└── data_description.txt    # Description des variables ✅
```
