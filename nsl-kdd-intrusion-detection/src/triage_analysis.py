import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from src.data_loader import load_and_combine_data, assign_column_names
from src.preprocessing import SELECTED_FEATURES, drop_unused_column_and_create_target
from src.artifact_pipeline import build_model_input

full = assign_column_names(load_and_combine_data("data/KDDTrain+.txt", "data/KDDTest+.txt"))
diff_all = full["difficulty"].astype(float).values
cls_all = full["class"].astype(str).values
d = drop_unused_column_and_create_target(full.copy())
y = d["label"].astype(int)
frame = d[SELECTED_FEATURES].copy()
idx = np.arange(len(frame))
_, te_idx = train_test_split(idx, test_size=.2, random_state=42, stratify=y)
X = frame.iloc[te_idx].copy(); yt = y.iloc[te_idx].values
diff = diff_all[te_idx]; cls = cls_all[te_idx]
print("holdout rows", len(X))

def probs(name):
    b = joblib.load(f"artifacts/{name}.pkl")
    # holdout may contain unseen categories -> mirror training-time eval (-1 sentinel)
    from src.artifact_pipeline import prepare_raw_frame, add_engineered_features
    enc = prepare_raw_frame(X, b["encoders"], allow_unknown=True)
    base = add_engineered_features(enc) if b["engineered"] else enc
    Z = pd.DataFrame(b["scaler"].transform(base[b["feature_names"]]), columns=b["feature_names"])
    return b["model"].predict_proba(Z)[:, 1]

p_rf, p_xgb = probs("random_forest"), probs("xgboost")
c_rf, c_xgb = (p_rf > .5).astype(int), (p_xgb > .5).astype(int)
for n, c in [("RF", c_rf), ("XGB", c_xgb)]:
    print(f"{n} forced accuracy: {(c==yt).mean()*100:.2f}%")
fam = lambda s: "normal" if s=="normal" else ("U2R/R2L" if s in {"buffer_overflow","loadmodule","perl","rootkit","ftp_write","guess_passwd","imap","multihop","phf","spy","warezclient","warezmaster","sqlattack","xterm","ps","httptunnel","named","sendmail","snmpgetattack","snmpguess","worm","xlock","xsnoop"} else "DoS/Probe")
for delta in (0.0, 0.05, 0.10):
    agree = c_rf == c_xgb
    ok = (np.abs(p_rf-.5) >= delta) & (np.abs(p_xgb-.5) >= delta)
    auto = agree & ok; tri = ~auto
    acc = (c_rf[auto] == yt[auto]).mean()*100
    tri_acc_if_forced = (c_rf[tri] == yt[tri]).mean()*100 if tri.any() else float("nan")
    print(f"\ndelta={delta:.2f}: coverage={auto.mean()*100:.2f}% triage={tri.mean()*100:.2f}% ({tri.sum()} rows) selective_acc={acc:.3f}%")
    print(f"  mean difficulty  automated={diff[auto].mean():.2f}  triage={diff[tri].mean():.2f}  (overall {diff.mean():.2f})")
    print(f"  accuracy RF would have had on triaged rows: {tri_acc_if_forced:.1f}%")
    fams = pd.Series([fam(c) for c in cls[tri]]).value_counts(normalize=True)*100
    print("  triage composition:", {k: round(v,1) for k,v in fams.items()})
    print("  wrong in automated:", int((c_rf[auto]!=yt[auto]).sum()), "of", int(auto.sum()))
