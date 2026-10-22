# Referencias de scikit-learn y SciPy para tests/validate.mjs (árbol CART, boosting, k vecinos, NNLS).
import json, numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.optimize import nnls
r = np.random.default_rng(3); n, nt = 500, 300
def gen(m):
    X = r.normal(size=(m, 6)); X[:, 5] = (r.random(m) < 0.4).astype(float)
    eta = 1.2 * X[:, 0] - 0.8 * X[:, 1] * (X[:, 2] > 0) + 0.7 * X[:, 5] - 0.4
    return X, eta
X, eta = gen(n); Xt, etat = gen(nt)
yb = (r.random(n) < 1 / (1 + np.exp(-eta))).astype(float); ybt = (r.random(nt) < 1 / (1 + np.exp(-etat))).astype(float)
yc = eta + r.normal(size=n)
out = dict(X=X.tolist(), Xt=Xt.tolist(), yb=yb.tolist(), ybt=ybt.tolist(), yc=yc.tolist())
out['tree'] = DecisionTreeRegressor(max_depth=4, min_samples_leaf=10, min_samples_split=20).fit(X, yc).predict(Xt).tolist()
gb = GradientBoostingClassifier(n_estimators=60, learning_rate=0.1, max_depth=3, min_samples_leaf=10, subsample=1.0, criterion='squared_error', random_state=0).fit(X, yb)
out['gbm_bin'] = gb.predict_proba(Xt)[:, 1].tolist()
gr = GradientBoostingRegressor(n_estimators=60, learning_rate=0.1, max_depth=3, min_samples_leaf=10, subsample=1.0, criterion='squared_error', random_state=0).fit(X, yc)
out['gbm_reg'] = gr.predict(Xt).tolist()
sc = StandardScaler().fit(X); out['knn'] = KNeighborsClassifier(n_neighbors=15).fit(sc.transform(X), yb).predict_proba(sc.transform(Xt))[:, 1].tolist()
out['rf_auc'] = float(roc_auc_score(ybt, RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=0).fit(X, yb).predict_proba(Xt)[:, 1]))
A = r.random((200, 5)); b = A @ np.array([0.5, 0.0, 0.3, 0.0, 0.2]) + 0.05 * r.normal(size=200); w, _ = nnls(A, b); out['nnls'] = dict(A=A.tolist(), b=b.tolist(), w=w.tolist())
json.dump(out, open('tests/reference.json', 'w'))
print('ok')
