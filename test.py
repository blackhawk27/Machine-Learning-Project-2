from joblib import Parallel, delayed
import torch
from dtuimldmtools import train_neural_net
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import torch
from dtuimldmtools import train_neural_net
from scipy.stats import ttest_rel




# Definer stien til din CSV-fil
filename = r"ObesityDataSet_raw_and_data_sinthetic.csv"


# Indlæs CSV-filen i en Pandas DataFrame
df = pd.read_csv(filename)

# Fjern 'NObeyesdad' og 'Weight' fra features
X = df.drop(columns=["NObeyesdad", "Weight"])  

# Gem target
y = df["Weight"].values

# Gem attributnavne (før encoding)
attributeNames = X.columns.tolist()

# Hent class labels
classLabels = df["NObeyesdad"].values  # Label

# Konverter labels (y) til numeriske værdier
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(classLabels)
classNames = label_encoder.classes_  # Unikke klasser

# Identificer feature-typer
categorical_features = ["Gender", "family_history_with_overweight", "FAVC", "SMOKE", "SCC", "CAEC", "CALC", "MTRANS"]
numeric_features = ["Age", "Height", "NCP", "CH2O", "FAF", "TUE", "FCVC"]

# One-hot encod de kategoriske kolonner
encoder = OneHotEncoder(drop="first", sparse_output=False)
#X_categorical = encoder.fit_transform(X[categorical_features])
X_categorical = encoder.fit_transform(X[categorical_features]).astype(float)



# Standardiser numeriske features (Z-score normalisering)
scaler = StandardScaler()
X_numeric = scaler.fit_transform(X[numeric_features])

# Kombiner alle features


X = np.hstack((X_numeric, X_categorical))


# Opdater attributnavne korrekt
ohe_feature_names = encoder.get_feature_names_out(categorical_features).tolist()
attributeNames = numeric_features + ohe_feature_names

# Bestem dataset-dimensioner
N, M = X.shape
C = len(classNames)

def compute_ridge_cv_error(lmbda, X_train, y_train, inner_cv):
    fold_errors = []
    for itrain, ival in inner_cv.split(X_train):
        model = Ridge(alpha=lmbda)
        model.fit(X_train[itrain], y_train[itrain])
        y_pred = model.predict(X_train[ival])
        err = mean_squared_error(y_train[ival], y_pred)
        fold_errors.append(err)
    return np.mean(fold_errors), lmbda

def compute_ann_cv_error(h, X_train, y_train, inner_cv, input_dim):
    fold_errors = []
    for itrain, ival in inner_cv.split(X_train):
        model = lambda: torch.nn.Sequential(
            torch.nn.Linear(input_dim, h),
            torch.nn.Tanh(),
            torch.nn.Linear(h, 1)
        )
        loss_fn = torch.nn.MSELoss()

        net, final_loss, _ = train_neural_net(
            model,
            loss_fn,
            X=torch.Tensor(X_train[itrain]),
            y=torch.Tensor(y_train[itrain]).unsqueeze(1),
            n_replicates=1,
            max_iter=2000
        )

        y_pred = net(torch.Tensor(X_train[ival])).detach().numpy().squeeze()
        err = mean_squared_error(y_train[ival], y_pred)
        fold_errors.append(err)
    return np.mean(fold_errors), h

def two_fold_10_layer_cross_validation(X, y, attributeNames, lambdas, hidden_units_list):
    K_outer = 10
    K_inner = 10
    CV_outer = KFold(n_splits=K_outer, shuffle=True, random_state=42)

    results = []

    for outer_fold, (train_idx, test_idx) in enumerate(CV_outer.split(X)):
        print(f"\nFold {outer_fold + 1}/{K_outer}")
        X_train_outer, y_train_outer = X[train_idx], y[train_idx]
        X_test_outer, y_test_outer = X[test_idx], y[test_idx]

        inner_cv = KFold(n_splits=K_inner, shuffle=True, random_state=42)

        # Ridge paralleliseret
        ridge_results = Parallel(n_jobs=-1)(
            delayed(compute_ridge_cv_error)(lmbda, X_train_outer, y_train_outer, inner_cv)
            for lmbda in lambdas
        )
        lambda_star = min(ridge_results, key=lambda x: x[0])[1]
        ridge_final = Ridge(alpha=lambda_star).fit(X_train_outer, y_train_outer)
        E_test_ridge = mean_squared_error(y_test_outer, ridge_final.predict(X_test_outer))

        # ANN paralleliseret
        ann_results = Parallel(n_jobs=-1)(
            delayed(compute_ann_cv_error)(h, X_train_outer, y_train_outer, inner_cv, X.shape[1])
            for h in hidden_units_list
        )
        h_star = min(ann_results, key=lambda x: x[0])[1]

        model = lambda: torch.nn.Sequential(
            torch.nn.Linear(X.shape[1], h_star),
            torch.nn.Tanh(),
            torch.nn.Linear(h_star, 1)
        )
        loss_fn = torch.nn.MSELoss()
        net_ann, _, _ = train_neural_net(
            model,
            loss_fn,
            X=torch.Tensor(X_train_outer),
            y=torch.Tensor(y_train_outer).unsqueeze(1),
            n_replicates=1,
            max_iter=2000
        )
        y_ann_pred = net_ann(torch.Tensor(X_test_outer)).detach().numpy().squeeze()
        E_test_ann = mean_squared_error(y_test_outer, y_ann_pred)

        # Baseline
        y_baseline = np.ones_like(y_test_outer) * np.mean(y_train_outer)
        E_test_baseline = mean_squared_error(y_test_outer, y_baseline)

        results.append({
            "fold": outer_fold + 1,
            "h*": h_star,
            "E_ann": E_test_ann,
            "lambda*": lambda_star,
            "E_ridge": E_test_ridge,
            "E_baseline": E_test_baseline
        })

    results_df = pd.DataFrame(results)
    print("\nResultattabel:\n", results_df)

    return results_df



lambdas = np.logspace(-2, 6, 10)
hidden_units_list = [1, 100, 400, 700, 1000]

results = two_fold_10_layer_cross_validation(X, y, attributeNames, lambdas, hidden_units_list)








from scipy.stats import t as t_dist
import numpy as np

def correlated_t_test(E_A, E_B, K, alpha=0.05):
    """
    Udfør correlated t-test baseret på Method 11.4.1 (Setup II)
    E_A og E_B: generaliseringsfejl (MSE) for to modeller over K folds
    """
    r = E_A - E_B  # forskel i fejl pr. fold
    r_hat = np.mean(r)
    sigma_hat = np.std(r, ddof=1)

    rho = 1.0 / K
    J = K

    t_hat = r_hat / (sigma_hat * np.sqrt(1/J + rho / (1 - rho)))
    nu = J - 1

    # Konfidensinterval
    zL = t_dist.ppf(alpha / 2, df=nu)
    zU = t_dist.ppf(1 - alpha / 2, df=nu)
    ci_L = r_hat + zL * sigma_hat * np.sqrt(1/J + rho / (1 - rho))
    ci_U = r_hat + zU * sigma_hat * np.sqrt(1/J + rho / (1 - rho))

    # p-værdi
    p_val = 2 * t_dist.cdf(-abs(t_hat), df=nu)

    return t_hat, p_val, (ci_L, ci_U)


# K = antal splits
# Antal splits = antal outer folds
K = 10

t, p, ci = correlated_t_test(results["E_ann"].values, results["E_ridge"].values, K)
print(f"\nCorrelated t-test (ANN vs Ridge): t = {t:.4f}, p = {p:.4f}, CI = [{ci[0]:.4f}, {ci[1]:.4f}]")

t, p, ci = correlated_t_test(results["E_ann"].values, results["E_baseline"].values, K)
print(f"Correlated t-test (ANN vs Baseline): t = {t:.4f}, p = {p:.4f}, CI = [{ci[0]:.4f}, {ci[1]:.4f}]")

t, p, ci = correlated_t_test(results["E_ridge"].values, results["E_baseline"].values, K)
print(f"Correlated t-test (Ridge vs Baseline): t = {t:.4f}, p = {p:.4f}, CI = [{ci[0]:.4f}, {ci[1]:.4f}]")