// Código Python (scikit-learn) que replica el análisis, sin código de figuras. cfg se inyecta como JSON en la cabecera del script.
const PY = String.raw`
import json, sys, warnings
import numpy as np, pandas as pd
from scipy.optimize import nnls, minimize
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

warnings.filterwarnings("ignore")
FAM = CFG["family"]; BIN = FAM == "binary"; EPS = 1e-6


# ------------------------------------------------------------------ datos
def load_design(df, predictors):
    """Matriz de diseño: numéricas tal cual; categóricas en indicadores (sin la referencia); NaN se conserva."""
    cols, names, groups = [], [], []
    for v in predictors:
        s = df[v["name"]]
        if v["type"] == "num":
            groups.append({"name": v["name"], "cols": [len(names)]}); names.append(v["name"]); cols.append(pd.to_numeric(s, errors="coerce").to_numpy(float))
        else:
            ix = []
            for lv in v["levels"][1:]:
                ix.append(len(names)); names.append(f"{v['name']}: {lv}")
                cols.append(np.where(s.isna(), np.nan, (s.astype(str) == str(lv)).astype(float)))
            groups.append({"name": v["name"], "cols": ix})
    return np.column_stack(cols), names, groups


def load_xy(path, predictors):
    df = pd.read_csv(path, na_values=CFG["na_values"], keep_default_na=True)
    df = df[df[CFG["outcome"]].notna()].reset_index(drop=True)
    X, names, groups = load_design(df, predictors)
    y = (df[CFG["outcome"]].astype(str) == str(CFG["event"])).to_numpy(float) if BIN else pd.to_numeric(df[CFG["outcome"]], errors="coerce").to_numpy(float)
    return X, y, names, groups


# ------------------------------------------------------------------ algoritmos: fit_learner devuelve una función predict(X)
def clip(p): return np.clip(p, EPS, 1 - EPS) if BIN else p


def fit_enet(Z, y, alpha, nfolds, rule, seed):
    """Red elástica (objetivo de glmnet): λ elegido por validación cruzada interna (mínimo o 1-ES) y reajuste con todos los datos."""
    n = len(y)
    lam_max = np.max(np.abs(Z.T @ (y - y.mean()))) / (n * max(alpha, 1e-3))
    lams = np.logspace(-4, 0, 30)[::-1] * lam_max
    def model(lam):
        if BIN:
            if alpha == 0: return LogisticRegression(penalty="l2", C=1 / (n * lam), max_iter=5000)
            return LogisticRegression(penalty="elasticnet", l1_ratio=alpha, C=1 / (n * lam), solver="saga", max_iter=5000)
        return ElasticNet(alpha=lam, l1_ratio=alpha, max_iter=10000) if alpha > 0 else ElasticNet(alpha=lam, l1_ratio=1e-8, max_iter=10000)
    cvf = (StratifiedKFold if BIN else KFold)(nfolds, shuffle=True, random_state=seed)
    err = np.zeros((nfolds, len(lams)))
    for f, (tr, te) in enumerate(cvf.split(Z, y if BIN else None)):
        for k, lam in enumerate(lams):
            m = model(lam).fit(Z[tr], y[tr])
            err[f, k] = log_loss(y[te], clip(m.predict_proba(Z[te])[:, 1]), labels=[0, 1]) if BIN else np.mean((y[te] - m.predict(Z[te])) ** 2)
    mu, se = err.mean(0), err.std(0, ddof=1) / np.sqrt(nfolds)
    kmin = int(np.argmin(mu))
    k = kmin if rule == "min" else int(np.min(np.where(mu <= mu[kmin] + se[kmin])[0]))  # lams decrece: el primero es el λ mayor
    return model(lams[k]).fit(Z, y)


def fit_learner(spec, Xtr, ytr, seed):
    k, P = spec["learner"], spec["params"]
    imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(Xtr); Xtr = imp.transform(Xtr)
    sc = StandardScaler().fit(Xtr); Ztr = sc.transform(Xtr)
    prep = lambda X: imp.transform(X)
    if k == "mean":
        m = ytr.mean(); return lambda X: np.full(len(X), m)
    if k == "glm":
        mdl = (LogisticRegression(penalty=None, max_iter=5000) if BIN else LinearRegression()).fit(Ztr, ytr)
        return lambda X: clip(mdl.predict_proba(sc.transform(prep(X)))[:, 1] if BIN else mdl.predict(sc.transform(prep(X))))
    if k == "enet":
        mdl = fit_enet(Ztr, ytr, P["alpha"], int(P["nfolds"]), P["rule"], seed)
        return lambda X: clip(mdl.predict_proba(sc.transform(prep(X)))[:, 1] if BIN else mdl.predict(sc.transform(prep(X))))
    if k == "knn":
        mdl = (KNeighborsClassifier if BIN else KNeighborsRegressor)(n_neighbors=min(int(P["k"]), len(ytr))).fit(Ztr, ytr)
        return lambda X: mdl.predict_proba(sc.transform(prep(X)))[:, 1] if BIN else mdl.predict(sc.transform(prep(X)))
    if k == "tree":
        mdl = DecisionTreeRegressor(max_depth=int(P["maxDepth"]), min_samples_leaf=int(P["minNode"]), min_samples_split=2 * int(P["minNode"]), min_impurity_decrease=P["cp"] * np.var(ytr), random_state=seed).fit(Xtr, ytr)
        return lambda X: clip(mdl.predict(prep(X)))
    if k == "rf":
        p = Xtr.shape[1]; mtry = int(P["mtry"]) if P["mtry"] > 0 else max(1, round(np.sqrt(p) if BIN else p / 3))
        mdl = RandomForestRegressor(n_estimators=int(P["ntree"]), max_features=min(mtry, p), min_samples_leaf=int(P["minNode"]), min_samples_split=max(2, 2 * int(P["minNode"]) - 2), random_state=seed, n_jobs=-1).fit(Xtr, ytr)
        return lambda X: clip(mdl.predict(prep(X)))
    if k == "gbm":
        kw = dict(n_estimators=int(P["nrounds"]), learning_rate=P["lr"], max_depth=int(P["depth"]), min_samples_leaf=int(P["minNode"]), min_samples_split=2 * int(P["minNode"]), subsample=P["subsample"], random_state=seed)
        mdl = (GradientBoostingClassifier(**kw) if BIN else GradientBoostingRegressor(**kw)).fit(Xtr, ytr)
        return lambda X: mdl.predict_proba(prep(X))[:, 1] if BIN else mdl.predict(prep(X))
    if k == "nnet":
        kw = dict(hidden_layer_sizes=(int(P["size"]),), activation="tanh", alpha=P["decay"], learning_rate_init=P["lr"], max_iter=int(P["epochs"]), solver="adam", random_state=seed)
        mdl = (MLPClassifier(**kw) if BIN else MLPRegressor(**kw)).fit(Ztr, ytr)
        return lambda X: clip(mdl.predict_proba(sc.transform(prep(X)))[:, 1] if BIN else mdl.predict(sc.transform(prep(X))))
    raise ValueError(k)


# ------------------------------------------------------------------ validación y métricas
def folds(y, V, seed):
    return list((StratifiedKFold if BIN else KFold)(V, shuffle=True, random_state=seed).split(np.zeros(len(y)), y if BIN else None))


def cv_predict(lib, X, y, V, seed):
    Z = np.zeros((len(lib), len(y)))
    for v, (tr, te) in enumerate(folds(y, V, seed)):
        for k, spec in enumerate(lib): Z[k, te] = fit_learner(spec, X[tr], y[tr], seed + 7 * v + k)(X[te])
    return Z


def metrics(p, y):
    if BIN:
        auc = roc_auc_score(y, p); n1 = y.sum(); n0 = len(y) - n1; q1 = auc / (2 - auc); q2 = 2 * auc * auc / (1 + auc)
        se = np.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n0 - 1) * (q2 - auc ** 2)) / (n1 * n0)); prev = y.mean(); br = brier_score_loss(y, p)
        return {"auc": auc, "auc_lo": max(0, auc - 1.96 * se), "auc_hi": min(1, auc + 1.96 * se), "brier": br, "brier_scaled": 1 - br / (prev * (1 - prev)), "logloss": log_loss(y, np.clip(p, 1e-9, 1 - 1e-9)), "oe": y.sum() / p.sum()}
    sse = np.sum((y - p) ** 2); return {"rmse": np.sqrt(sse / len(y)), "mae": np.mean(np.abs(y - p)), "r2": 1 - sse / np.sum((y - y.mean()) ** 2)}


# ------------------------------------------------------------------ Super Learner
def sl_weights(Z, y, method):
    K = Z.shape[0]
    if method == "loglik" and BIN:
        f = lambda a: -np.sum(y * np.log(np.clip(a @ Z, EPS, 1)) + (1 - y) * np.log(np.clip(1 - a @ Z, EPS, 1)))
        r = minimize(f, np.full(K, 1 / K), method="SLSQP", bounds=[(0, 1)] * K, constraints={"type": "eq", "fun": lambda a: a.sum() - 1}); return np.clip(r.x, 0, None) / np.clip(r.x, 0, None).sum()
    w, _ = nnls(Z.T, y); s = w.sum()
    if s > 0: return w / s
    w = np.zeros(K); w[np.argmin(((Z - y) ** 2).mean(1))] = 1; return w


class SuperLearner:
    def __init__(self, lib, X, y, V, seed, method):
        self.lib = lib; self.Z = cv_predict(lib, X, y, V, seed); self.w = sl_weights(self.Z, y, method)
        self.cv_risk = ((self.Z - y) ** 2).mean(1)
        self.fits = [fit_learner(s, X, y, seed + 100 + k) for k, s in enumerate(lib)]
    def predict_each(self, X): return np.array([f(X) for f in self.fits])
    def predict(self, X): return self.w @ self.predict_each(X)


def nested_sl(lib, X, y, Vo, Vi, seed, method):
    out = np.zeros(len(y)); each = np.zeros((len(lib), len(y)))
    for v, (tr, te) in enumerate(folds(y, Vo, seed + 999)):
        sl = SuperLearner(lib, X[tr], y[tr], Vi, seed + 31 * v, method); each[:, te] = sl.predict_each(X[te]); out[te] = sl.w @ each[:, te]
    return out, each


# ------------------------------------------------------------------ importancia por permutación y selección de variables
def grouped_permutation_importance(predict, X, y, groups, R, seed):
    rng = np.random.default_rng(seed); score = (lambda p: roc_auc_score(y, p)) if BIN else (lambda p: -np.sqrt(np.mean((y - p) ** 2)))
    base = score(predict(X)); out = {}
    for g in groups:
        d = []
        for _ in range(R):
            Xp = X.copy(); perm = rng.permutation(len(X)); Xp[:, g["cols"]] = X[perm][:, g["cols"]]; d.append(base - score(predict(Xp)))
        out[g["name"]] = float(np.mean(d))
    return out


def select_variables(lib, X, y, groups, thr, R, Vsel, Vcv, seed, method, force=()):
    """Importancia media en pliegues de prueba (el Super Learner se ajusta sin ellos); se conservan las variables con caída > thr (AUC) o thr × DE(y) (RMSE)."""
    acc = {g["name"]: [] for g in groups}
    for v, (tr, te) in enumerate(folds(y, Vsel, seed + 555)):
        sl = SuperLearner(lib, X[tr], y[tr], Vcv, seed + v, method)
        for k, d in grouped_permutation_importance(sl.predict, X[te], y[te], groups, R, seed + v).items(): acc[k].append(d)
    cut = thr if BIN else thr * y.std(ddof=1)
    imp = {k: float(np.mean(v)) for k, v in acc.items()}; keep = [k for k, d in imp.items() if d > cut or k in force] or [max(imp, key=imp.get)]
    return imp, keep


# ------------------------------------------------------------------ flujo principal
def main(data_path, new_path=None):
    preds = CFG["predictors"]; sel = CFG["selection"]
    X, y, names, groups = load_xy(data_path, preds); sch = CFG["scheme"]; lib = CFG["library"]; seed = CFG["seed"]; method = CFG["method"]
    print(f"{len(y)} pacientes, {X.shape[1]} columnas" + (f", {int(y.sum())} eventos" if BIN else ""))
    if sel["run"]:
        imp, keep = select_variables(lib, X, y, groups, sel["thr"], sel["R"], sel["Vsel"], sel["Vcv"], seed, method, sel["force"])
        col = "caida_AUC" if BIN else "aumento_RMSE"; pd.DataFrame({"variable": list(imp), col: list(imp.values()), "se_conserva": [k in keep for k in imp]}).sort_values(col, ascending=False).to_csv("importancia_permutacion.csv", index=False)
        print("Variables conservadas:", keep); preds = [p for p in preds if p["name"] in keep]; X, y, names, groups = load_xy(data_path, preds)
    elif sel["selected"]:
        preds = [p for p in preds if p["name"] in sel["selected"]]; X, y, names, groups = load_xy(data_path, preds)
    # validación cruzada (repetida): métricas promedio de cada algoritmo; pesos del Super Learner con la primera repetición
    reps = max(1, sch["repeats"]) if sch["type"] == "cv" else 1; all_m, first = [], None
    Xtr, ytr = X, y
    if sch["type"] == "holdout":
        te = folds(y, max(2, round(1 / sch["test_frac"])), seed)[0][1]; mask = np.ones(len(y), bool); mask[te] = False; Xtr, ytr, Xte, yte = X[mask], y[mask], X[~mask], y[~mask]
    for r in range(reps):
        Z = cv_predict(lib, Xtr, ytr, sch["V"], seed + 1000 * r); all_m.append([metrics(Z[k], ytr) for k in range(len(lib))]); first = Z if first is None else first
    table = pd.DataFrame([{kk: np.mean([m[k][kk] for m in all_m]) for kk in all_m[0][k]} for k in range(len(lib))]); table.insert(0, "algoritmo", [s["label"] for s in lib])
    sl = SuperLearner(lib, Xtr, ytr, sch["V"], seed, method); table["peso_SL"] = sl.w
    print(table.round(3).to_string(index=False)); table.to_csv("resultados_algoritmos.csv", index=False)
    pd.DataFrame({"fila": np.arange(1, len(ytr) + 1), "observado": ytr, **{s["label"]: first[k] for k, s in enumerate(lib)}, "Super_Learner_cv": sl.w @ first}).to_csv("predicciones_fuera_de_muestra.csv", index=False)
    if sch["type"] == "holdout": print("Conjunto de prueba (Super Learner):", {k: round(float(v), 3) for k, v in metrics(sl.predict(Xte), yte).items()})
    if CFG["nested"]["on"] and sch["type"] == "cv":
        ens, each = nested_sl(lib, X, y, CFG["nested"]["Vo"], CFG["nested"]["Vi"], seed, method)
        print("Super Learner con validación cruzada anidada:", {k: round(float(v), 3) for k, v in metrics(ens, y).items()})
    if new_path:  # predicción para pacientes nuevos (mismas columnas de predictores)
        dn = pd.read_csv(new_path, na_values=CFG["na_values"]); Xn, _, _ = load_design(dn, preds); out = pd.DataFrame({"Super_Learner": sl.predict(Xn)})
        for k, s in enumerate(lib): out[s["label"]] = sl.fits[k](Xn)
        out.to_csv("predicciones_nuevos.csv", index=False); print("Predicciones guardadas en predicciones_nuevos.csv")


if __name__ == "__main__":
    # uso: python replicar.py datos.csv [nuevos_pacientes.csv]
    main(sys.argv[1] if len(sys.argv) > 1 else CFG["data_file"], sys.argv[2] if len(sys.argv) > 2 else None)
`;

// cfg: {data_file, na_values, outcome, family, event, predictors:[{name,type,levels}], library:[{learner,label,params}], scheme:{type,V,repeats,test_frac}, method, seed,
//       nested:{on,Vo,Vi}, selection:{run,thr,R,Vsel,Vcv,force,selected}, meta:{n,events,date}}
export function pythonScript(cfg) {
  const head = `#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Código de replicación · EPI-Maching-Learning (EPI-LAB)  —  generado el ${cfg.meta.date}
Desenlace: ${cfg.outcome}${cfg.family === 'binary' ? ` (evento: ${cfg.event})` : ''} · ${cfg.meta.n} pacientes${cfg.meta.events != null ? `, ${cfg.meta.events} eventos` : ''} · ${cfg.predictors.length} predictores.

Requisitos: Python >= 3.9, numpy, pandas, scipy, scikit-learn >= 1.2.  Uso:  python replicar.py datos.csv [nuevos_pacientes.csv]

Este script reproduce el flujo del análisis (preparación de datos, algoritmos, validación cruzada, Super Learner, importancia por permutación y
predicción de pacientes nuevos) con scikit-learn y SciPy. NO incluye código de figuras. La aplicación calcula con una implementación propia en
JavaScript, verificada frente a scikit-learn y SciPy: árbol CART, boosting de gradiente, k vecinos y regresión penalizada (a λ fijo) coinciden con
diferencias < 1e-9; el bosque aleatorio y la red neuronal son equivalentes pero no idénticos (otro generador aleatorio), y los pliegues y la rejilla de λ
de la validación cruzada interna también difieren. Espere diferencias pequeñas en las métricas (p. ej., AUC ± 0,01), no valores idénticos.
"""
`;
  return `${head}\nimport json
CFG = json.loads(r'''${JSON.stringify(cfg, null, 1).replace(/'''/g, "'")}''')\n${PY}`;
}
