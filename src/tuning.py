from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import optuna

def XGBoptimizer(trial, df_train, df_val, tree_method):
    params = {
        "eta": trial.suggest_categorical("eta", [0.01,0.015, 0.025, 0.05, 0.1]),
        "max_depth": trial.suggest_int("max_depth", 2, 25),
        "gamma": trial.suggest_uniform("gamma", 0.05, 1.0),
        "min_child_weight": trial.suggest_categorical("min_child_weight", [1, 3, 5, 7]),
        "subsample": trial.suggest_categorical("subsample", [0.6, 0.7, 0.8, 0.9, 1.0]),
        "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.6, 0.7, 0.8, 0.9, 1.0]),
        "lambda": trial.suggest_categorical("lambda", [0.01, 0.1, 0.5, 1.0]),
        "alpha": trial.suggest_categorical("alpha", [0.01, 0.1, 0.5, 1.0]),
        "tree_method": tree_method
    }

    xgb_clf = XGBClassifier(**params)
    xgb_clf.fit(df_train.drop("answered_correctly", axis=1), df_train["answered_correctly"])
    val_pred = xgb_clf.predict(df_val.drop("answered_correctly", axis=1))

    return roc_auc_score(df_val["answered_correctly"], val_pred)

def optimizeXGB(df_train, df_val, tree_method="auto", n_trials=10):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: XGBoptimizer(trial, df_train, df_val, tree_method), n_trials=n_trials)
    return study