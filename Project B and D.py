import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import matplotlib.pyplot as plt


# =========================
# Utils: metrics
# =========================
def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def r2_variance_explained(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """VE / R2 = 1 - MSE / Var(observed) (population variance, ddof=0)"""
    var = float(np.var(y_true))
    if var == 0:
        return 0.0
    return 1.0 - mse(y_true, y_pred) / var


# =========================
# GD for linear regression (batch GD)
# Model: y_hat = [1, X] @ w
# Loss: mean((y_hat - y)^2)
# Grad: (2/n) X^T (y_hat - y)
# =========================
def add_bias(X: np.ndarray) -> np.ndarray:
    return np.c_[np.ones((X.shape[0], 1)), X]


def predict_linear(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return add_bias(X) @ w


def gradient_descent_linear_regression(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.01,
    n_iters: int = 5000,
    tol: float = 1e-9,
    verbose: bool = False,
    print_every: int = 500,
):
    Xb = add_bias(X)  # (n, p+1)
    n, d = Xb.shape

    w = np.zeros(d)
    loss_history = []
    prev_loss = None

    for it in range(n_iters):
        y_hat = Xb @ w
        err = y_hat - y

        loss = float(np.mean(err ** 2))
        loss_history.append(loss)

        grad = (2.0 / n) * (Xb.T @ err)
        w = w - lr * grad

        if verbose and (it % print_every == 0):
            print(f"iter={it}, loss={loss:.6f}")

        if not np.isfinite(loss):
            if verbose:
                print("Loss became nan/inf -> stop. Try smaller lr.")
            break

        if prev_loss is not None and abs(prev_loss - loss) < tol:
            if verbose:
                print(f"Converged at iter={it}, loss={loss:.6f}")
            break
        prev_loss = loss

    return w, loss_history


# =========================
# Univariate runners
# =========================
def run_univariate_fixed(
    X_train_all: np.ndarray,
    X_test_all: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    set_name: str,
    lr: float,
    n_iters: int,
):
    print("\n============================")
    print(f"Univariate Set (FIXED HP): {set_name}")
    print(f"lr={lr}, n_iters={n_iters}")
    print("============================")

    rows = []
    for j, fname in enumerate(feature_names):
        Xtr = X_train_all[:, [j]]
        Xte = X_test_all[:, [j]]

        w, loss = gradient_descent_linear_regression(Xtr, y_train, lr=lr, n_iters=n_iters, verbose=False)

        ytr_hat = predict_linear(Xtr, w)
        yte_hat = predict_linear(Xte, w)

        rows.append(
            {
                "feature": fname,
                "lr": lr,
                "n_iters": n_iters,
                "w0_bias": float(w[0]),
                "w1": float(w[1]),
                "train_mse": mse(y_train, ytr_hat),
                "train_r2": r2_variance_explained(y_train, ytr_hat),
                "test_mse": mse(y_test, yte_hat),
                "test_r2": r2_variance_explained(y_test, yte_hat),
                "final_loss": float(loss[-1]) if len(loss) else np.nan,
            }
        )

    df = pd.DataFrame(rows).sort_values(by="train_r2", ascending=False)

    for _, r in df.iterrows():
        print(
            f"{str(r['feature'])[:32]:32s} | m={r['w1']: .6f}  b={r['w0_bias']: .6f} | "
            f"Train R2={r['train_r2']: .4f}  Test R2={r['test_r2']: .4f} "
            f"| Train MSE={r['train_mse']:.3f}  Test MSE={r['test_mse']:.3f}"
        )

    return df


def run_univariate_raw_search(
    X_train_all: np.ndarray,
    X_test_all: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
):
    print("\n===================================================")
    print("RAW Univariate: Hyperparameter Search (BEST per feature)")
    print("===================================================")

    # adaptive lr per feature based on Lipschitz approx
    lr_multipliers = [0.1, 0.3, 1.0, 3.0]
    iter_grid = [50000, 150000, 300000]

    best_rows = []

    for j, fname in enumerate(feature_names):
        Xtr = X_train_all[:, [j]]
        Xte = X_test_all[:, [j]]

        # lr_base ~ 0.9 / L , where L = (2/n) * sum(x^2)
        xcol = Xtr[:, 0]
        n = xcol.shape[0]
        # This sum can overflow warning on some platforms; it's fine for our use here.
        L = (2.0 / n) * float(np.sum(xcol ** 2))
        lr_base = 0.9 / L if L > 0 else 1e-7
        lr_grid = [m * lr_base for m in lr_multipliers]

        best = None

        for lr in lr_grid:
            for n_iters in iter_grid:
                w, loss = gradient_descent_linear_regression(
                    Xtr,
                    y_train,
                    lr=lr,
                    n_iters=n_iters,
                    tol=1e-12,
                    verbose=False,
                )

                if len(loss) == 0 or (not np.isfinite(loss[-1])) or np.isnan(loss[-1]):
                    continue

                ytr_hat = predict_linear(Xtr, w)
                yte_hat = predict_linear(Xte, w)

                r2_tr = r2_variance_explained(y_train, ytr_hat)
                r2_te = r2_variance_explained(y_test, yte_hat)

                if (best is None) or (r2_tr > best["train_r2"]):
                    best = {
                        "feature": fname,
                        "best_lr": lr,
                        "best_n_iters": n_iters,
                        "w0_bias": float(w[0]),
                        "w1": float(w[1]),
                        "train_mse": mse(y_train, ytr_hat),
                        "train_r2": r2_tr,
                        "test_mse": mse(y_test, yte_hat),
                        "test_r2": r2_te,
                        "final_loss": float(loss[-1]),
                        "lr_base": lr_base,
                    }

        if best is None:
            best = {
                "feature": fname,
                "best_lr": np.nan,
                "best_n_iters": np.nan,
                "w0_bias": np.nan,
                "w1": np.nan,
                "train_mse": np.nan,
                "train_r2": -np.inf,
                "test_mse": np.nan,
                "test_r2": -np.inf,
                "final_loss": np.nan,
                "lr_base": lr_base,
            }

        best_rows.append(best)

        print(
            f"{str(fname)[:32]:32s}  best Train R2={best['train_r2']:.4f}  "
            f"Test R2={best['test_r2']:.4f}  lr={best['best_lr']:.3e}  "
            f"iters={best['best_n_iters']}  lr_base={best['lr_base']:.3e}"
        )

    best_df = pd.DataFrame(best_rows).sort_values(by="train_r2", ascending=False)

    print("\nTop RAW univariate features by Train R2:")
    print(best_df[["feature", "train_r2", "test_r2", "best_lr", "best_n_iters"]].head(8).to_string(index=False))

    pos_count = int((best_df["train_r2"] > 0).sum())
    print(f"\nRAW univariate features with Train R2 > 0: {pos_count}")
    if pos_count >= 2:
        print("✅ Requirement met: At least two RAW univariate models have positive Train R2.")
    else:
        print("❌ Still not met: expand lr_multipliers / iter_grid.")

    return best_df


# =========================
# Save helpers
# =========================


def save_loss_plot(loss_history: list[float], out_png: str, title: str) -> None:
    """Save a loss-vs-iteration plot to a PNG file."""
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_loss_csv(loss_history: list[float], out_csv: str) -> None:
    """Save loss history to a CSV file."""
    pd.DataFrame({"iteration": range(len(loss_history)), "mse_loss": loss_history}).to_csv(out_csv, index=False)


def save_multivariate_results(
    out_csv: str,
    model_name: str,
    feature_names: list[str],
    Xtr: np.ndarray,
    Xte: np.ndarray,
    ytr: np.ndarray,
    yte: np.ndarray,
    w: np.ndarray,
):
    """Save multivariate weights + train/test metrics to a CSV."""
    yhat_tr = predict_linear(Xtr, w)
    yhat_te = predict_linear(Xte, w)

    rows = []
    rows.append(
        {
            "model": model_name,
            "param": "b",
            "feature": "(intercept)",
            "value": float(w[0]),
            "train_mse": mse(ytr, yhat_tr),
            "train_r2": r2_variance_explained(ytr, yhat_tr),
            "test_mse": mse(yte, yhat_te),
            "test_r2": r2_variance_explained(yte, yhat_te),
        }
    )

    for i, fname in enumerate(feature_names, start=1):
        rows.append(
            {
                "model": model_name,
                "param": f"m{i}",
                "feature": fname,
                "value": float(w[i]),
                "train_mse": mse(ytr, yhat_tr),
                "train_r2": r2_variance_explained(ytr, yhat_tr),
                "test_mse": mse(yte, yhat_te),
                "test_r2": r2_variance_explained(yte, yhat_te),
            }
        )

    pd.DataFrame(rows).to_csv(out_csv, index=False)


# =========================
# Multivariate evaluation print
# =========================
def eval_and_print(model_name: str, Xtr: np.ndarray, Xte: np.ndarray, ytr: np.ndarray, yte: np.ndarray, w: np.ndarray):
    yhat_tr = predict_linear(Xtr, w)
    yhat_te = predict_linear(Xte, w)

    print(f"\n=== {model_name} ===")
    print("Train MSE:", mse(ytr, yhat_tr))
    print("Train R2 :", r2_variance_explained(ytr, yhat_tr))
    print("Test  MSE:", mse(yte, yhat_te))
    print("Test  R2 :", r2_variance_explained(yte, yhat_te))


# =========================
# Part C: p-values via OLS (statsmodels)
# =========================
def part_c_pvalues(X: np.ndarray, y: np.ndarray, feature_names: list[str], title: str, out_csv: str | None = None):
    Xc = sm.add_constant(X)  # adds intercept
    model = sm.OLS(y, Xc).fit()

    coef = model.params
    pvals = model.pvalues

    names = ["const"] + feature_names
    out = pd.DataFrame({"feature": names, "coef": coef, "p_value": pvals})
    out = out.sort_values(by="p_value", ascending=True)

    print(f"\n=== Part C: {title} ===")
    print(out.to_string(index=False))
    if out_csv is not None:
        out.to_csv(out_csv, index=False)
    return out


# =========================
# (Optional) Q2.1 / Q2.2 helper checks (if you want to verify)
# =========================
def q2_single_sample_update_mse_mean(m: np.ndarray, b: float, x: np.ndarray, y: float, alpha: float):
    """
    One-sample update but consistent with mean-MSE gradient form:
    grad_m = 2 * e * x, grad_b = 2 * e
    (since n=1, (2/n)=2)
    """
    y_hat = float(np.dot(m, x) + b)
    e = y_hat - y
    m_new = m - alpha * (2.0 * e * x)
    b_new = b - alpha * (2.0 * e)
    return m_new, b_new


def q2_batch_update_5samples_mse_mean(m: np.ndarray, b: float, X: np.ndarray, y: np.ndarray, alpha: float):
    """
    One batch update using mean MSE:
    grad_m = (2/n) X^T e
    grad_b = (2/n) sum(e)
    """
    y_hat = X @ m + b
    e = y_hat - y
    n = X.shape[0]
    grad_m = (2.0 / n) * (X.T @ e)
    grad_b = (2.0 / n) * float(np.sum(e))
    m_new = m - alpha * grad_m
    b_new = b - alpha * grad_b
    return m_new, b_new


# =========================
# MAIN
# =========================
def main():
    # ---- Load data (put Concrete_Data.xls in same folder) ----
    data_path = "Concrete_Data.xls"
    df = pd.read_excel(data_path)

    print(df.head())
    print("Shape:", df.shape)

    feature_names = df.columns[:-1].tolist()
    target_name = df.columns[-1]
    print("\nFeatures:", feature_names)
    print("Target:", target_name)

    # ---- Split (test = rows 501–630, using python 0-index 500:630) ----
    test_df = df.iloc[500:630]
    train_df = df.drop(df.index[500:630])

    X_train = train_df.iloc[:, :-1].values
    y_train = train_df.iloc[:, -1].values.astype(float)

    X_test = test_df.iloc[:, :-1].values
    y_test = test_df.iloc[:, -1].values.astype(float)

    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

    # ---- Standardize predictors (fit only on train) ----
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---- Multivariate (Scaled) ----
    w_mv_scaled, loss_mv_scaled = gradient_descent_linear_regression(
        X_train_scaled, y_train, lr=0.05, n_iters=20000, tol=1e-9, verbose=True, print_every=500
    )
    eval_and_print("Multivariate (Scaled)", X_train_scaled, X_test_scaled, y_train, y_test, w_mv_scaled)
    save_loss_plot(loss_mv_scaled, "loss_multivariate_scaled.png", "Multivariate (Scaled) - Loss over Iterations")
    save_loss_csv(loss_mv_scaled, "loss_multivariate_scaled.csv")

    print("\n--- Weights: Multivariate (Scaled) ---")
    print("b =", w_mv_scaled[0])
    for i, name in enumerate(feature_names, start=1):
        print(f"m{i} ({name}) =", w_mv_scaled[i])

    save_multivariate_results(
        out_csv="multivariate_scaled_results.csv",
        model_name="Multivariate (Scaled)",
        feature_names=feature_names,
        Xtr=X_train_scaled,
        Xte=X_test_scaled,
        ytr=y_train,
        yte=y_test,
        w=w_mv_scaled,
    )

    # ---- Multivariate (Raw) ----
    w_mv_raw, loss_mv_raw = gradient_descent_linear_regression(
        X_train, y_train, lr=1e-7, n_iters=30000, tol=1e-9, verbose=True, print_every=500
    )
    eval_and_print("Multivariate (Raw)", X_train, X_test, y_train, y_test, w_mv_raw)
    save_loss_plot(loss_mv_raw, "loss_multivariate_raw.png", "Multivariate (Raw) - Loss over Iterations")
    save_loss_csv(loss_mv_raw, "loss_multivariate_raw.csv")

    print("\n--- Weights: Multivariate (Raw) ---")
    print("b =", w_mv_raw[0])
    for i, name in enumerate(feature_names, start=1):
        print(f"m{i} ({name}) =", w_mv_raw[i])

    save_multivariate_results(
        out_csv="multivariate_raw_results.csv",
        model_name="Multivariate (Raw)",
        feature_names=feature_names,
        Xtr=X_train,
        Xte=X_test,
        ytr=y_train,
        yte=y_test,
        w=w_mv_raw,
    )

    # ---- Univariate (Scaled) fixed ----
    uni_scaled_df = run_univariate_fixed(
        X_train_scaled, X_test_scaled, y_train, y_test, feature_names, set_name="Scaled", lr=0.05, n_iters=15000
    )

    # ---- Univariate (Raw) fixed baseline ----
    uni_raw_fixed_df = run_univariate_fixed(
        X_train, X_test, y_train, y_test, feature_names, set_name="Raw (fixed baseline)", lr=1e-7, n_iters=20000
    )

    # ---- Raw Univariate best search ----
    raw_best_df = run_univariate_raw_search(X_train, X_test, y_train, y_test, feature_names)

    # ---- Save CSVs ----
    uni_scaled_df.to_csv("univariate_scaled_results.csv", index=False)
    uni_raw_fixed_df.to_csv("univariate_raw_fixed_baseline.csv", index=False)
    raw_best_df.to_csv("univariate_raw_best_search.csv", index=False)

    print("\nSaved:")
    print(" - univariate_scaled_results.csv")
    print(" - univariate_raw_fixed_baseline.csv")
    print(" - univariate_raw_best_search.csv")
    print(" - multivariate_scaled_results.csv")
    print(" - multivariate_raw_results.csv")
    print(" - part_c_scaled_pvalues.csv")
    print(" - part_c_raw_pvalues.csv")
    print(" - part_c_log1p_pvalues.csv")
    print("\nDone ✅")

    # ---- Part C: p-values (Scaled / Raw / Log(x+1)) ----
    _ = part_c_pvalues(X_train_scaled, y_train, feature_names, title="Scaled", out_csv="part_c_scaled_pvalues.csv")
    _ = part_c_pvalues(X_train, y_train, feature_names, title="Raw", out_csv="part_c_raw_pvalues.csv")

    # log(x+1) transform on predictors (raw -> log1p), y stays raw
    X_train_log = np.log1p(X_train)
    _ = part_c_pvalues(X_train_log, y_train, feature_names, title="Log(x+1)", out_csv="part_c_log1p_pvalues.csv")


if __name__ == "__main__":
    main()
