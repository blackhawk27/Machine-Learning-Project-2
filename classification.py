from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
import torch
import numpy as np
import pandas as pd
from dtuimldmtools import train_neural_net



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



def compute_logreg_cv_error(C_val, X_train, y_train, inner_cv):
    fold_errors = []
    for itrain, ival in inner_cv.split(X_train):
        model = LogisticRegression(
            penalty='l2', C=C_val, multi_class='multinomial', solver='lbfgs', max_iter=1000)
        model.fit(X_train[itrain], y_train[itrain])
        y_pred = model.predict(X_train[ival])
        err = np.mean(y_pred != y_train[ival])
        fold_errors.append(err)
    return np.mean(fold_errors), C_val


def compute_ann_cv_error(h, X_train, y_train, inner_cv, input_dim, num_classes):
    fold_errors = []
    for itrain, ival in inner_cv.split(X_train):
        model = lambda: torch.nn.Sequential(
            torch.nn.Linear(input_dim, h),
            torch.nn.Tanh(),
            torch.nn.Linear(h, num_classes)
        )
        loss_fn = torch.nn.CrossEntropyLoss()
        net, final_loss, _ = train_neural_net(
            model,
            loss_fn,
            X=torch.Tensor(X_train[itrain]),
            y=torch.Tensor(y_train[itrain]).long(),
            n_replicates=1,
            max_iter=2000
        )
        y_pred = net(torch.Tensor(X_train[ival])).argmax(dim=1).numpy()
        err = np.mean(y_pred != y_train[ival])
        fold_errors.append(err)
    return np.mean(fold_errors), h


def multiclass_classification_CV(X, y, classNames, lambdas, hidden_units_list):
    K_outer = 10
    K_inner = 10
    num_classes = len(classNames)
    CV_outer = KFold(n_splits=K_outer, shuffle=True, random_state=42)

    results = []

    for outer_fold, (train_idx, test_idx) in enumerate(CV_outer.split(X)):
        X_train_outer, y_train_outer = X[train_idx], y[train_idx]
        X_test_outer, y_test_outer = X[test_idx], y[test_idx]

        inner_cv = KFold(n_splits=K_inner, shuffle=True, random_state=42)

        # Logistic Regression
        Cs = [1 / l if l > 0 else 1e6 for l in lambdas]
        logreg_results = Parallel(n_jobs=-1)(
            delayed(compute_logreg_cv_error)(C, X_train_outer, y_train_outer, inner_cv)
            for C in Cs
        )
        C_star = min(logreg_results, key=lambda x: x[0])[1]
        logreg_final = LogisticRegression(
            penalty='l2', C=C_star, multi_class='multinomial', solver='lbfgs', max_iter=1000)
        logreg_final.fit(X_train_outer, y_train_outer)
        E_test_logreg = np.mean(logreg_final.predict(X_test_outer) != y_test_outer)

        # ANN
        ann_results = Parallel(n_jobs=-1)(
            delayed(compute_ann_cv_error)(h, X_train_outer, y_train_outer, inner_cv, X.shape[1], num_classes)
            for h in hidden_units_list
        )
        h_star = min(ann_results, key=lambda x: x[0])[1]
        model = lambda: torch.nn.Sequential(
            torch.nn.Linear(X.shape[1], h_star),
            torch.nn.Tanh(),
            torch.nn.Linear(h_star, num_classes)
        )
        loss_fn = torch.nn.CrossEntropyLoss()
        net_ann, _, _ = train_neural_net(
            model,
            loss_fn,
            X=torch.Tensor(X_train_outer),
            y=torch.Tensor(y_train_outer).long(),
            n_replicates=1,
            max_iter=2000
        )
        y_ann_pred = net_ann(torch.Tensor(X_test_outer)).argmax(dim=1).numpy()
        E_test_ann = np.mean(y_ann_pred != y_test_outer)

        # Baseline
        from collections import Counter
        most_common_class = Counter(y_train_outer).most_common(1)[0][0]
        y_baseline = np.ones_like(y_test_outer) * most_common_class
        E_test_baseline = np.mean(y_baseline != y_test_outer)

        results.append({
            "fold": outer_fold + 1,
            "h*": h_star,
            "E_ann": E_test_ann,
            "C* (1/lambda)": C_star,
            "E_logreg": E_test_logreg,
            "E_baseline": E_test_baseline
        })

    return pd.DataFrame(results)



lambdas = np.logspace(-2, 6, 10)  
hidden_units_list = [1, 15, 30, 50, 100]

#results = multiclass_classification_CV(X, y, classNames, lambdas, hidden_units_list)

#print("\nResultattabel:\n", results)



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

#t, p, ci = correlated_t_test(results["E_ann"].values, results["E_logreg"].values, K)
#print(f"\nANN vs LogReg: t = {t:.4f}, p = {p:.4f}, CI = [{ci[0]:.4f}, {ci[1]:.4f}]")

#t, p, ci = correlated_t_test(results["E_ann"].values, results["E_baseline"].values, K)
#print(f"\nANN vs Baseline: t = {t:.4f}, p = {p:.4f}, CI = [{ci[0]:.4f}, {ci[1]:.4f}]")

#t, p, ci = correlated_t_test(results["E_logreg"].values, results["E_baseline"].values, K)
#print(f"\nLogReg vs Baseline: t = {t:.4f}, p = {p:.4f}, CI = [{ci[0]:.4f}, {ci[1]:.4f}]")





def plot_logreg_coefficients(X, y, attributeNames, classNames):
    # Fit multinomial logistic regression
    model = LogisticRegression(
        penalty='l2',
        C=1.0,
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000
    )
    model.fit(X, y)

    # Extract weights and bias
    W = model.coef_
    b = model.intercept_.reshape(-1, 1)

    # Combine weights and bias
    W_full = np.hstack((W, b))
    full_feature_names = attributeNames + ['bias']

    # Create DataFrame
    df_coef = pd.DataFrame(W_full, columns=full_feature_names, index=[f"Class {cls}" for cls in classNames])

    # Plot
    plt.figure(figsize=(18, 6))
    sns.heatmap(
        df_coef,
        annot=True,
        cmap="coolwarm",       # blue-orange diverging colormap
        center=0,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"shrink": 0.6}
    )
    plt.title("Logistic Regression Coefficients (Multinomial)", fontsize=16)
    plt.ylabel("Class", fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.show()



plot_logreg_coefficients(X, y, attributeNames, classNames)