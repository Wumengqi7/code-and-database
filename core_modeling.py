"""SFRC fatigue-life prediction: concise core-modeling reference.

Demonstrates the central training design on an already curated dataset.
This is NOT the full experiment implementation or an exact-results reproduction.
SHAP, figures, diagnostic tests and raw-data screening are not included.
"""
import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
from fancyimpute import IterativeImputer
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

FEATURES = ['S', 'Vf', 'L/d', 'fc', 'W/C']
AUXILIARY = ['Fiber length (mm)', 'Fiber diameter (μm)']
SEEDS = [42, 123, 456, 789, 999]


def prepare_cases(data):
    """Expects a curated table with log_Nf, features and case metadata.

    Source screening, censoring exclusions and geometric derivations must
    follow the criteria described in the manuscript; they are not scripted here.
    """
    steel = data[data['Fiber type'].str.contains('steel fiber', case=False, na=False)]
    eligible = steel[steel['Data Quality Classification'].isin(['A', 'B'])]
    hooked = eligible[eligible['Fiber type'].str.contains('Hooked-end', na=False)]
    mortar = hooked[hooked['Substrate type'].str.contains('mortar', na=False)]
    concrete = hooked[hooked['Substrate type'].str.contains('Concrete', na=False)]
    return {'Case A': mortar, 'Case B': concrete,
            'Case A+B': pd.concat([mortar, concrete]), 'All-in-One': eligible}


def preprocess(train, valid):
    """Fit missing-value treatment and scaling on training rows only."""
    columns = FEATURES + [c for c in AUXILIARY if c in train.columns]
    columns = [c for c in columns if not train[c].isna().all()]
    if any(c not in columns for c in FEATURES):
        raise ValueError('A model feature is entirely missing in a training fold.')
    imputer = IterativeImputer(random_state=42, max_iter=5)
    train_imp = pd.DataFrame(imputer.fit_transform(train[columns]), columns=columns)
    valid_imp = pd.DataFrame(imputer.transform(valid[columns]), columns=columns)
    scaler = StandardScaler()
    return scaler.fit_transform(train_imp[FEATURES]), scaler.transform(valid_imp[FEATURES])


def optimize_parameters(X, y, seed):
    """Illustrate Optuna search using fold-local preprocessing."""
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'learning_rate': trial.suggest_float('learning_rate', .01, .1, log=True),
            'subsample': trial.suggest_float('subsample', .7, .9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', .7, 1.),
            'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'seed': 42,
        }
        scores = []
        for train_idx, val_idx in cv.split(X):
            Xt, Xv = preprocess(X.iloc[train_idx], X.iloc[val_idx])
            dtrain = xgb.DMatrix(Xt, label=y.iloc[train_idx])
            dvalid = xgb.DMatrix(Xv, label=y.iloc[val_idx])
            model = xgb.train(params, dtrain, num_boost_round=500,
                              evals=[(dvalid, 'validation')],
                              early_stopping_rounds=20, verbose_eval=False)
            scores.append(np.sqrt(mean_squared_error(y.iloc[val_idx], model.predict(dvalid))))
        return float(np.mean(scores))

    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=min(20, max(1, len(X) // 2)))
    return study.best_params


def train_case(frame, seed):
    """Train with CV-selected boosting rounds and evaluate on held-out data."""
    columns = FEATURES + [c for c in AUXILIARY if c in frame.columns]
    frame = frame.dropna(subset=['log_Nf'])
    X, y = frame[columns], frame['log_Nf']
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=seed)
    params = optimize_parameters(Xtr, ytr, seed)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rounds = []
    for tr, va in cv.split(Xtr):
        Xt, Xv = preprocess(Xtr.iloc[tr], Xtr.iloc[va])
        model = xgb.train(params, xgb.DMatrix(Xt, label=ytr.iloc[tr]),
                          num_boost_round=500,
                          evals=[(xgb.DMatrix(Xv, label=ytr.iloc[va]), 'validation')],
                          early_stopping_rounds=20, verbose_eval=False)
        rounds.append(model.best_iteration + 1)
    Xtr_s, Xte_s = preprocess(Xtr, Xte)
    model = xgb.train(params, xgb.DMatrix(Xtr_s, label=ytr),
                      num_boost_round=max(1, int(np.median(rounds))))
    pred = model.predict(xgb.DMatrix(Xte_s))
    return {'r2': r2_score(yte, pred),
            'rmse': np.sqrt(mean_squared_error(yte, pred)),
            'mae': mean_absolute_error(yte, pred)}


if __name__ == '__main__':
    # The input must be a curated dataset; this example omits raw-data governance.
    data = pd.read_excel('SFRC_curated_dataset.xlsx')
    for name, case in prepare_cases(data).items():
        for seed in SEEDS:
            print(name, seed, train_case(case, seed))
