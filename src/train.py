"""
train.py
========
Script d'entraînement principal — House Price Prediction.
Usage : python src/train.py
Auteur : [Votre Nom] | 2026
"""

import os, sys
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

sys.path.append(os.path.dirname(__file__))
from preprocessing import full_pipeline

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data')
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_PATH  = os.path.join(DATA_DIR, 'test.csv')
OUT_PATH   = os.path.join(DATA_DIR, 'submission.csv')


def main():
    print("=" * 58)
    print("   🏠  HOUSE PRICE PREDICTION — TRAINING PIPELINE")
    print("=" * 58)

    # 1. Chargement
    print("\n📦 Chargement des données...")
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    print(f"   Train : {train.shape} | Test : {test.shape}")

    # 2. Preprocessing
    print("\n🔧 Preprocessing + Feature Engineering...")
    X, X_test, y, scaler = full_pipeline(train, test)
    print(f"   Features finales : {X.shape[1]}")

    # 3. Split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # 4. Modèles
    models = {
        'Ridge':           Ridge(alpha=10, random_state=RANDOM_STATE),
        'Lasso':           Lasso(alpha=0.001, random_state=RANDOM_STATE, max_iter=5000),
        'Decision Tree':   DecisionTreeRegressor(max_depth=6, random_state=RANDOM_STATE),
        'Random Forest':   RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05,
            max_depth=4, subsample=0.8, random_state=RANDOM_STATE
        ),
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    print(f"\n{'Modèle':<24}  CV RMSE    R²")
    print("-" * 44)

    results = {}
    for name, model in models.items():
        cv = cross_val_score(model, X_tr, y_tr,
                             cv=kf, scoring='neg_root_mean_squared_error', n_jobs=-1)
        model.fit(X_tr, y_tr)
        r2   = r2_score(y_val, model.predict(X_val))
        results[name] = {'model': model, 'cv_rmse': -cv.mean(), 'r2': r2}
        print(f"   {name:<22}  {-cv.mean():.4f}    {r2:.4f}")

    # 5. Optimisation GB
    print("\n⚙️  Optimisation RandomizedSearchCV (40 iter)...")
    param_dist = {
        'n_estimators': [200, 300, 500],
        'learning_rate': [0.03, 0.05, 0.08],
        'max_depth': [3, 4, 5],
        'subsample': [0.7, 0.8, 0.9],
    }
    rs = RandomizedSearchCV(
        GradientBoostingRegressor(random_state=RANDOM_STATE),
        param_dist, n_iter=40, cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1, random_state=RANDOM_STATE
    )
    rs.fit(X_tr, y_tr)
    best = rs.best_estimator_
    r2_opt = r2_score(y_val, best.predict(X_val))

    # 6. Résultat
    print(f"\n🏆 Meilleur modèle : Gradient Boosting (optimisé)")
    print(f"   CV RMSE : {-rs.best_score_:.4f}")
    print(f"   Val R²  : {r2_opt:.4f}")

    # Sauvegarde du modèle et du scaler
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model.pkl')
    scaler_path = os.path.join(os.path.dirname(__file__), '..', 'scaler.pkl')
    joblib.dump(best, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\n💾 Modèle sauvegardé : {model_path}")
    print(f"💾 Scaler sauvegardé : {scaler_path}")

    # 7. Prédictions
    preds = np.expm1(best.predict(X_test))
    sub   = pd.DataFrame({'Id': test['Id'], 'SalePrice': preds})
    sub.to_csv(OUT_PATH, index=False)
    print(f"\n✅ submission.csv → {OUT_PATH}")
    print(f"   Prix médian prédit : ${np.median(preds):,.0f}")
    print("=" * 58)


if __name__ == '__main__':
    main()
