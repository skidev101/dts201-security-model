import pickle

import pandas as pd
import numpy as np
# import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    ConfusionMatrixDisplay, roc_curve
)
import os
import matplotlib.pyplot as plt

PLOT_DIR  = "../outputs/plots"
MODEL_DIR = "../outputs/models"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# PRESCRIPTIVE RULES
# Derived from literature + feature importance analysis.
# These are the model's "prescriptions" — security recommendations.
# ─────────────────────────────────────────────
PRESCRIPTIONS = {
    "THEFT/ROBBERY": [
        "🔒 Install CCTV cameras at parking lots, campus gates, and lecture hall corridors",
        "💡 Improve lighting in poorly lit areas (hostels, parking, shortcuts)",
        "🎒 Run 'Don't Leave Valuables Unattended' awareness campaigns",
        "🛡️  Deploy security at peak theft hours (7-9 AM, 12-2 PM, 5-7 PM)",
    ],
    "ASSAULT/VIOLENCE": [
        "🚨 Install emergency call points / panic buttons at strategic locations",
        "👮 Increase visible security patrols during evening and night hours",
        "📵 Establish a zero-tolerance policy for fighting with immediate suspension",
        "🧠 Introduce conflict resolution and mental health support programs",
    ],
    "SEXUAL HARASSMENT/ASSAULT": [
        "📢 Implement a clear, confidential sexual harassment reporting mechanism",
        "🏃 Provide student escort services for late-night movement on campus",
        "💡 Ensure hostel corridors and bathrooms are well-lit and monitored",
        "📚 Conduct mandatory consent and awareness training for all students",
    ],
    "VANDALISM/TRESPASSING": [
        "🚧 Install perimeter fencing and controlled access gates",
        "📹 Use CCTV monitoring at campus boundaries",
        "🪪 Enforce strict ID card policies for all persons on campus",
        "🌙 Increase patrols at night when vandalism peaks",
    ],
    "DRUG-RELATED": [
        "🔍 Conduct routine searches at campus entrances",
        "🤝 Partner with law enforcement for intelligence sharing",
        "📞 Create anonymous tip lines for reporting drug activity",
        "💊 Provide drug counseling and rehabilitation referrals for students",
    ],
    "HIGH_RISK_TIME_NIGHT": [
        "🌃 Increase patrols between 8 PM - 2 AM (peak high-risk window)",
        "🔦 Ensure all campus pathways are adequately lit at night",
        "🚌 Provide safe late-night shuttle transport between hostels and key buildings",
    ],
    "HIGH_RISK_WEEKEND": [
        "📅 Maintain full weekend security coverage (Saturdays and Sundays)",
        "🎉 Require event security plans for all weekend social gatherings",
    ],
}

def load_data():
    df = pd.read_csv("data/processed/kaggle_clean.csv", low_memory=False)
    return df

def prepare_features(df):
    """Select and encode features for the classifier."""
    features = []

    # Time features
    if "HOUR" in df.columns:
        df["HOUR_SIN"] = np.sin(2 * np.pi * df["HOUR"] / 24)
        df["HOUR_COS"] = np.cos(2 * np.pi * df["HOUR"] / 24)
        features += ["HOUR", "HOUR_SIN", "HOUR_COS"]

    if "IS_WEEKEND" in df.columns:
        features.append("IS_WEEKEND")

    # Crime category (encoded)
    le = LabelEncoder()
    if "CRIME_CATEGORY" in df.columns:
        df["CRIME_CAT_ENC"] = le.fit_transform(df["CRIME_CATEGORY"].fillna("OTHER"))
        features.append("CRIME_CAT_ENC")
        category_encoder = le
    else:
        category_encoder = None

    # Victim age
    if "VICT_AGE" in df.columns:
        df["VICT_AGE"] = pd.to_numeric(df["VICT_AGE"], errors="coerce").fillna(df.get("VICT_AGE", pd.Series()).median())
        df["VICT_AGE"] = df["VICT_AGE"].clip(0, 100)
        features.append("VICT_AGE")

    # Campus-specific flag
    if "IS_CAMPUS_SPECIFIC" in df.columns:
        features.append("IS_CAMPUS_SPECIFIC")

    

    features = [f for f in features if f in df.columns]
    X = df[features].fillna(0)
    y = df["HIGH_RISK"].fillna(0).astype(int)

    print(f"   Features used: {features}")
    return X, y, features, category_encoder


def train_model(X, y):
    """Train a Random Forest with cross-validation."""
    print("🌲 Training Random Forest classifier...")

    if y.nunique() < 2:
        raise ValueError("Target variable has only one class. Check preprocessing.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
    print(f"   CV ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    return model, X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test, features):
    """Generate evaluation metrics and plots."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n📈 Model Evaluation:")
    print(classification_report(y_test, y_pred, target_names=["Low Risk", "High Risk"]))
    roc = roc_auc_score(y_test, y_prob)
    print(f"   ROC-AUC Score: {roc:.4f}")

    # ── Confusion matrix ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Low Risk", "High Risk"])
    disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("Confusion Matrix", fontweight="bold", fontsize=13)

    # ── ROC curve ──
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[1].plot(fpr, tpr, color="#1a3a5c", lw=2, label=f"AUC = {roc:.3f}")
    axes[1].plot([0,1],[0,1], "k--", lw=1)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve", fontweight="bold", fontsize=13)
    axes[1].legend()
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/10_model_evaluation.png", dpi=120)
    plt.close()
    print("   ✅ Plot 10: Model evaluation (confusion matrix + ROC)")

    return roc


def plot_feature_importance(model, features):
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#c73e1d" if imp > importances.median() else "#2e86ab" for imp in importances]
    ax.barh(importances.index, importances.values, color=colors, edgecolor="white")
    ax.set_title("Feature Importances\n(What factors most predict high-risk incidents?)",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/11_feature_importance.png", dpi=120)
    plt.close()
    print("   ✅ Plot 11: Feature importance chart")


def generate_prescriptive_rules(df, model, features):
    """
    Core prescriptive logic:
    Analyze patterns and map them to actionable recommendations.
    Returns a structured dict of findings + prescriptions.
    """
    print("\n📋 Generating prescriptive rules...")
    rules = []

    # Rule 1: Most dangerous crime category
    if "CRIME_CATEGORY" in df.columns:
        top_crime = df.groupby("CRIME_CATEGORY")["HIGH_RISK"].mean().idxmax()
        crime_rate = df.groupby("CRIME_CATEGORY")["HIGH_RISK"].mean().max()
        rules.append({
            "finding": f"'{top_crime}' has the highest risk rate ({crime_rate*100:.1f}%)",
            "priority": "HIGH",
            "prescriptions": PRESCRIPTIONS.get(top_crime, [])
        })

    # Rule 2: Peak risk hours
    if "HOUR" in df.columns:
        hourly_risk = df.groupby("HOUR")["HIGH_RISK"].mean()
        peak_hours = hourly_risk[hourly_risk > hourly_risk.quantile(0.75)].index.tolist()
        hour_fmt = ", ".join([f"{h}:00" for h in sorted(peak_hours)])
        rules.append({
            "finding": f"Highest risk hours: {hour_fmt}",
            "priority": "HIGH",
            "prescriptions": PRESCRIPTIONS["HIGH_RISK_TIME_NIGHT"]
        })

    # Rule 3: Weekend risk
    if "IS_WEEKEND" in df.columns:
        weekday_risk = df[df["IS_WEEKEND"]==0]["HIGH_RISK"].mean()
        weekend_risk = df[df["IS_WEEKEND"]==1]["HIGH_RISK"].mean()
        if weekend_risk > weekday_risk * 1.1:
            rules.append({
                "finding": f"Weekend risk ({weekend_risk*100:.1f}%) is higher than weekday risk ({weekday_risk*100:.1f}%)",
                "priority": "MEDIUM",
                "prescriptions": PRESCRIPTIONS["HIGH_RISK_WEEKEND"]
            })

    # Rule 4: Top crime types with prescriptions
    if "CRIME_CATEGORY" in df.columns:
        crime_counts = df["CRIME_CATEGORY"].value_counts().head(3)
        for crime, count in crime_counts.items():
            if crime in PRESCRIPTIONS:
                rules.append({
                    "finding": f"'{crime}' accounts for {count:,} incidents",
                    "priority": "MEDIUM" if count < crime_counts.max() else "HIGH",
                    "prescriptions": PRESCRIPTIONS[crime]
                })

    print(f"   ✅ Generated {len(rules)} prescriptive rules")
    return rules


def plot_prescriptive_summary(rules):
    """Visual summary of prescriptions."""
    finding_labels = [r["finding"][:55] + "..." if len(r["finding"]) > 55 else r["finding"] for r in rules[:6]]
    priorities = [1 if r["priority"]=="HIGH" else 0.5 for r in rules[:6]]
    colors = ["#c73e1d" if p == 1 else "#f18f01" for p in priorities]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(finding_labels)), priorities, color=colors, edgecolor="white")
    ax.set_yticks(range(len(finding_labels)))
    ax.set_yticklabels(finding_labels, fontsize=9)
    ax.set_xlabel("Priority Score")
    ax.set_title("Prescriptive Model: Risk Findings & Priority Levels",
                 fontsize=13, fontweight="bold", pad=15)
    import matplotlib.patches as mpatches
    high = mpatches.Patch(color="#c73e1d", label="HIGH Priority")
    med  = mpatches.Patch(color="#f18f01", label="MEDIUM Priority")
    ax.legend(handles=[high, med], loc="lower right")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/12_prescriptive_summary.png", dpi=120)
    plt.close()
    print("   ✅ Plot 12: Prescriptive summary chart")


def run():
    df = load_data()
    print(f"📂 Loaded {len(df):,} records\n")

    X, y, features, encoder = prepare_features(df)
    model, X_train, X_test, y_train, y_test = train_model(X, y)
    roc = evaluate_model(model, X_test, y_test, features)
    plot_feature_importance(model, features)

    rules = generate_prescriptive_rules(df, model, features)
    plot_prescriptive_summary(rules)

    # Save model
    with open(f"{MODEL_DIR}/campus_security_model.pkl", "wb") as f:
        pickle.dump({"model": model, "features": features, "rules": rules}, f)
    print("Model saved to outputs/models/campus_security_model.pkl")

    print(f"\n STEP 3 COMPLETE! ROC-AUC = {roc:.4f}")
    return model, rules, roc


if __name__ == "__main__":
    run()
