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

#Sebastian Surrøv


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




def two_fold_10_layer_cross_validation(X, y, attributeNames, lambdas, hidden_units_list):
    import torch
    from dtuimldmtools import train_neural_net
    from sklearn.model_selection import KFold
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error

    K_outer = 5
    K_inner = 5
    CV_outer = KFold(n_splits=K_outer, shuffle=True, random_state=42)

    results = []

    for outer_fold, (train_idx, test_idx) in enumerate(CV_outer.split(X)):
        print(f"\nFold {outer_fold + 1}/{K_outer}")
        X_train_outer, y_train_outer = X[train_idx], y[train_idx]
        X_test_outer, y_test_outer = X[test_idx], y[test_idx]

        # === Inner CV for Ridge ===
        inner_cv = KFold(n_splits=K_inner, shuffle=True)
        val_errors_ridge = []

        for lmbda in lambdas:
            fold_errors = []
            for itrain, ival in inner_cv.split(X_train_outer):
                ridge_model = Ridge(alpha=lmbda)
                ridge_model.fit(X_train_outer[itrain], y_train_outer[itrain])
                y_val_pred = ridge_model.predict(X_train_outer[ival])
                err = mean_squared_error(y_train_outer[ival], y_val_pred)
                fold_errors.append(err)
            val_errors_ridge.append(np.mean(fold_errors))

        lambda_star = lambdas[np.argmin(val_errors_ridge)]
        ridge_final = Ridge(alpha=lambda_star).fit(X_train_outer, y_train_outer)
        y_ridge_pred = ridge_final.predict(X_test_outer)
        E_test_ridge = mean_squared_error(y_test_outer, y_ridge_pred)

        # === Inner CV for ANN ===
        val_errors_ann = []
        for h in hidden_units_list:
            fold_errors = []
            for itrain, ival in inner_cv.split(X_train_outer):
                model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(X.shape[1], h),
                    torch.nn.Tanh(),
                    torch.nn.Linear(h, 1)
                )
                loss_fn = torch.nn.MSELoss()

                net, final_loss, _ = train_neural_net(
                    model,
                    loss_fn,
                    X=torch.Tensor(X_train_outer[itrain]),
                    y=torch.Tensor(y_train_outer[itrain]).unsqueeze(1),
                    n_replicates=1,
                    max_iter=4000
                )
                y_val_pred = net(torch.Tensor(X_train_outer[ival])).detach().numpy().squeeze()
                fold_errors.append(mean_squared_error(y_train_outer[ival], y_val_pred))
            val_errors_ann.append(np.mean(fold_errors))

        h_star = hidden_units_list[np.argmin(val_errors_ann)]

        # Træn final ANN
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
            max_iter=4000
        )
        y_ann_pred = net_ann(torch.Tensor(X_test_outer)).detach().numpy().squeeze()
        E_test_ann = mean_squared_error(y_test_outer, y_ann_pred)

        # === Baseline ===
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
hidden_units_list = [1, 50, 100, 300, 500]

results = two_fold_10_layer_cross_validation(X, y, attributeNames, lambdas, hidden_units_list)




t_stat, p_val = ttest_rel(results["E_ann"], results["E_ridge"])
print(f"\nPaired t-test for E_ann and E_ridge: t = {t_stat:.4f}, p = {p_val:.4f}")

t_stat, p_val = ttest_rel(results["E_ann"], results["E_baseline"])
print(f"\nPaired t-test for E_ann and E_baseline: t = {t_stat:.4f}, p = {p_val:.4f}")

t_stat, p_val = ttest_rel(results["E_baseline"], results["E_ridge"])
print(f"\nPaired t-test for E_baseline and E_ridge: t = {t_stat:.4f}, p = {p_val:.4f}")
