"""
preprocessing.py
================
Pipeline de préprocessing pour House Price Prediction.
Auteur : [Votre Nom] | 2026
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

NONE_COLS = [
    'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
    'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
    'BsmtFinType2', 'MasVnrType'
]

QUALITY_MAP  = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
QUALITY_COLS = [
    'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
    'HeatingQC', 'KitchenQual', 'FireplaceQu',
    'GarageQual', 'GarageCond', 'PoolQC'
]


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Imputation intelligente selon la nature des valeurs manquantes."""
    df = df.copy()
    for col in NONE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna('None')
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include='object').columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    return df


def encode_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Encodage ordinal des variables de qualité."""
    df = df.copy()
    for col in QUALITY_COLS:
        if col in df.columns:
            df[col] = df[col].map(QUALITY_MAP).fillna(0).astype(int)
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Création de 11 nouvelles features à fort pouvoir prédictif."""
    df = df.copy()
    df['TotalSF']        = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBathrooms'] = (df['FullBath'] + 0.5 * df['HalfBath'] +
                            df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath'])
    df['TotalPorchSF']   = (df['OpenPorchSF'] + df['EnclosedPorch'] +
                            df['3SsnPorch'] + df['ScreenPorch'])
    df['HouseAge']       = df['YrSold'] - df['YearBuilt']
    df['RemodAge']       = df['YrSold'] - df['YearRemodAdd']
    df['HasPool']        = (df['PoolArea']    > 0).astype(int)
    df['HasGarage']      = (df['GarageArea']  > 0).astype(int)
    df['HasBasement']    = (df['TotalBsmtSF'] > 0).astype(int)
    df['HasFireplace']   = (df['Fireplaces']  > 0).astype(int)
    df['Remodeled']      = (df['YearBuilt'] != df['YearRemodAdd']).astype(int)
    df['QualitySF']      = df['OverallQual'] * df['GrLivArea']
    return df


def full_pipeline(train: pd.DataFrame, test: pd.DataFrame):
    """
    Pipeline complet : imputation → FE → encodage ordinal → OHE → standardisation.
    Returns : X_train, X_test, y, scaler
    """
    y = np.log1p(train['SalePrice'])

    train_p = feature_engineering(encode_quality(impute_missing(train)))
    test_p  = feature_engineering(encode_quality(impute_missing(test)))

    X      = train_p.drop(['Id', 'SalePrice'], axis=1)
    X_test = test_p.drop(['Id'], axis=1)

    X_all  = pd.concat([X, X_test], axis=0)
    X_all  = pd.get_dummies(X_all, drop_first=True)
    X      = X_all.iloc[:len(train_p)]
    X_test = X_all.iloc[len(train_p):]

    scaler = StandardScaler()
    X_sc   = pd.DataFrame(scaler.fit_transform(X),      columns=X.columns)
    Xt_sc  = pd.DataFrame(scaler.transform(X_test),     columns=X.columns)

    return X_sc, Xt_sc, y, scaler


if __name__ == '__main__':
    import pandas as pd
    train = pd.read_csv('../data/train.csv')
    test  = pd.read_csv('../data/test.csv')
    X, Xt, y, sc = full_pipeline(train, test)
    print(f'✅ Pipeline OK — X:{X.shape}  Xt:{Xt.shape}  y:{y.shape}')
