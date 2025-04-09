import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import zscore, ttest_rel, t as t_dist
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, accuracy_score

from joblib import Parallel, delayed
import torch
from dtuimldmtools import train_neural_net






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




# ╔════════════════════════════════════════════════════════════════════╗
# ║                  Tjek om data er standartniseret                   ║
# ╚════════════════════════════════════════════════════════════════════╝


# Tjek standardiseringen
means = X_numeric.mean(axis=0)
stds = X_numeric.std(axis=0)

#print("Middelværdier (bør være ~0):", means)
#print("Standardafvigelser (bør være ~1):", stds)




# ╔════════════════════════════════════════════════════════════════════╗
# ║            Regulariserings parameter på lineær model               ║
# ╚════════════════════════════════════════════════════════════════════╝



# X og y skal være z-normaliserede!
# Eksempel på interval for lambda (logaritmisk skala)
lambdas = np.logspace(-2, 6, 30)  # 30 værdier fra 10^-4 til 10^4

# K-fold CV setup
K = 10
CV = KFold(n_splits=K, shuffle=True, random_state=2)



def plot_10_fold_cross_validation_gen_Error():

    # Liste til at gemme generaliseringsfejl (MSE)
    gen_errors = []
    train_errors = []

    for lmbda in lambdas:
        fold_val_errors = []
        fold_train_errors = []

        for train_idx, val_idx in CV.split(X):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            model = Ridge(alpha=lmbda)
            model.fit(X_train, y_train)

            y_val_pred = model.predict(X_val)
            y_train_pred = model.predict(X_train)

            val_error = mean_squared_error(y_val, y_val_pred)
            train_error = mean_squared_error(y_train, y_train_pred)

            fold_val_errors.append(val_error)
            fold_train_errors.append(train_error)

        gen_errors.append(np.mean(fold_val_errors))
        train_errors.append(np.mean(fold_train_errors))

    plt.semilogx(lambdas, train_errors, marker='o', label='Train error')
    plt.semilogx(lambdas, gen_errors, marker='o', label='Test/Validation error')
    plt.xlabel("λ (log scale)")
    plt.ylabel("Mean Squared Error")
    plt.title("Train vs Validation Error vs λ")
    plt.legend()
    plt.grid(True)
    #plt.show()



    best_lambda = lambdas[np.argmin(gen_errors)]
    print(f"Bedste λ: {best_lambda}")
    return best_lambda



best_lambda = plot_10_fold_cross_validation_gen_Error()




model = Ridge(alpha=best_lambda)
model.fit(X, y)  # Brug hele datasættet

w = model.coef_
b = model.intercept_


# Udskriv alle koefficienter (feature-effekter)
print("\nModelkoefficienter (feature-effekt på vægt):\n")
for name, weight in zip(attributeNames, w):
    print(f"{name}: {weight:.3f}")



# Sortér og udskriv features efter størst absolut effekt
print("\nVigtigste features (sorteret efter betydning):\n")
for name, weight in sorted(zip(attributeNames, w), key=lambda x: -abs(x[1])):
    print(f"{name:30s}: {weight:.3f}")



def bar_plot_weights():

    # Konverter til DataFrame for bedre håndtering og sortering
    coef_df = pd.DataFrame({
        "Feature": attributeNames,
        "Weight": w
    })

    # Sortér efter absolut værdi (vigtigste først)
    coef_df["AbsWeight"] = np.abs(coef_df["Weight"])
    coef_df_sorted = coef_df.sort_values(by="AbsWeight", ascending=False)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=coef_df_sorted, y="Feature", x="Weight", palette="coolwarm")
    plt.title("Ridge Regression – Feature Weights")
    plt.xlabel("Vægt (positiv eller negativ effekt på vægt)")
    plt.ylabel("Feature")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#bar_plot_weights()






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
hidden_units_list = [1, 15, 30, 50, 100]

#results = two_fold_10_layer_cross_validation(X, y, attributeNames, lambdas, hidden_units_list)










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

#t, p, ci = correlated_t_test(results["E_ann"].values, results["E_ridge"].values, K)
#print(f"\nCorrelated t-test (ANN vs Ridge): t = {t:.4f}, p = {p:.4f}, CI = [{ci[0]:.4f}, {ci[1]:.4f}]")

#t, p, ci = correlated_t_test(results["E_ann"].values, results["E_baseline"].values, K)
#print(f"Correlated t-test (ANN vs Baseline): t = {t:.4f}, p = {p:.4f}, CI = [{ci[0]:.4f}, {ci[1]:.4f}]")

#t, p, ci = correlated_t_test(results["E_ridge"].values, results["E_baseline"].values, K)
#print(f"Correlated t-test (Ridge vs Baseline): t = {t:.4f}, p = {p:.4f}, CI = [{ci[0]:.4f}, {ci[1]:.4f}]")







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



#plot_logreg_coefficients(X, y, attributeNames, classNames)