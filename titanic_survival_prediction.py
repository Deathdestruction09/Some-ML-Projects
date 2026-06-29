"""
=======================================================================
  TITANIC SURVIVAL PREDICTION — Complete ML Pipeline
=======================================================================
  Dataset  : Kaggle Titanic (train.csv / test.csv)
  Covers   : EDA · Preprocessing · Feature Engineering · Model Training
             Hyperparameter Tuning · Evaluation · Submission CSV

  How to use
  ----------
  1.  Download the Kaggle dataset:
        kaggle competitions download -c titanic
      Or grab it from: https://www.kaggle.com/c/titanic/data

  2.  Place train.csv and test.csv in the same folder as this script.

  3.  Install dependencies (once):
        pip install pandas numpy matplotlib seaborn scikit-learn xgboost

  4.  Run:
        python titanic_survival_prediction.py
=======================================================================
"""

# ─────────────────────────────────────────────
#  0.  IMPORTS
# ─────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection   import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing     import LabelEncoder, StandardScaler
from sklearn.impute            import SimpleImputer
from sklearn.pipeline          import Pipeline
from sklearn.metrics           import (accuracy_score, classification_report,
                                       confusion_matrix, roc_auc_score, roc_curve)

# Models
from sklearn.linear_model      import LogisticRegression
from sklearn.tree              import DecisionTreeClassifier
from sklearn.ensemble          import (RandomForestClassifier,
                                       GradientBoostingClassifier,
                                       VotingClassifier)
from sklearn.svm               import SVC
from xgboost                   import XGBClassifier

print("=" * 65)
print("  TITANIC SURVIVAL PREDICTION")
print("=" * 65)


# ─────────────────────────────────────────────
#  1.  LOAD DATA
# ─────────────────────────────────────────────
print("\n[1/7]  Loading data …")

train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

print(f"  Train shape : {train.shape}")
print(f"  Test  shape : {test.shape}")
print(f"\n  Train columns : {list(train.columns)}")
print(f"\n  First 5 rows :")
print(train.head())


# ─────────────────────────────────────────────
#  2.  EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────
print("\n[2/7]  Exploratory Data Analysis …")

# --- 2a. Missing values
print("\n  Missing values (train) :")
missing = train.isnull().sum()
missing_pct = (missing / len(train) * 100).round(2)
print(pd.DataFrame({"Count": missing, "Pct %": missing_pct})
        .query("Count > 0").sort_values("Count", ascending=False))

# --- 2b. Survival rate overview
print(f"\n  Overall survival rate : {train['Survived'].mean()*100:.1f}%")

print("\n  Survival by Pclass :")
print(train.groupby("Pclass")["Survived"].mean().round(3))

print("\n  Survival by Sex :")
print(train.groupby("Sex")["Survived"].mean().round(3))

# --- 2c. Visualisations (saved to PNG files)
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Titanic EDA", fontsize=16, fontweight="bold")

# Survival count
sns.countplot(x="Survived", data=train, ax=axes[0, 0],
              palette=["#e74c3c", "#2ecc71"])
axes[0, 0].set_title("Survival Count")
axes[0, 0].set_xticklabels(["Died", "Survived"])

# Survival by class
sns.barplot(x="Pclass", y="Survived", data=train, ax=axes[0, 1],
            palette="Blues_d")
axes[0, 1].set_title("Survival Rate by Pclass")

# Survival by sex
sns.barplot(x="Sex", y="Survived", data=train, ax=axes[0, 2],
            palette=["#3498db", "#e91e8c"])
axes[0, 2].set_title("Survival Rate by Sex")

# Age distribution
train["Age"].dropna().plot.hist(ax=axes[1, 0], bins=30,
                                color="#9b59b6", edgecolor="white")
axes[1, 0].set_title("Age Distribution")
axes[1, 0].set_xlabel("Age")

# Fare distribution
train["Fare"].plot.hist(ax=axes[1, 1], bins=40,
                        color="#e67e22", edgecolor="white")
axes[1, 1].set_title("Fare Distribution")
axes[1, 1].set_xlabel("Fare")

# Correlation heatmap
num_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
sns.heatmap(train[num_cols].corr(), annot=True, fmt=".2f",
            cmap="coolwarm", ax=axes[1, 2], linewidths=0.5)
axes[1, 2].set_title("Correlation Heatmap")

plt.tight_layout()
plt.savefig("titanic_eda.png", dpi=120)
plt.close()
print("  Saved: titanic_eda.png")


# ─────────────────────────────────────────────
#  3.  FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n[3/7]  Feature Engineering …")

def engineer_features(df):
    df = df.copy()

    # ── Title extraction from Name ──────────────────────────────────
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    rare_titles = df["Title"].value_counts()[df["Title"]
                  .value_counts() < 10].index
    df["Title"] = df["Title"].replace(rare_titles, "Rare")
    df["Title"] = df["Title"].replace({
        "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"
    })

    # ── Family size & alone flag ────────────────────────────────────
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"]    = (df["FamilySize"] == 1).astype(int)

    # ── Fare per person (group ticket aware) ───────────────────────
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]

    # ── Deck from Cabin ────────────────────────────────────────────
    df["Deck"] = df["Cabin"].str[0].fillna("Unknown")

    # ── Age bands ──────────────────────────────────────────────────
    df["AgeBand"] = pd.cut(df["Age"],
                           bins=[0, 12, 18, 35, 60, 100],
                           labels=["Child", "Teen", "YoungAdult",
                                   "Adult", "Senior"],
                           right=False)

    # ── Fare bands ─────────────────────────────────────────────────
    df["FareBand"] = pd.qcut(df["Fare"].clip(upper=512),
                             q=4,
                             labels=["Low", "Mid", "High", "VHigh"])

    # ── Drop columns not useful for model ──────────────────────────
    df.drop(columns=["Name", "Ticket", "Cabin"], inplace=True)

    return df


train = engineer_features(train)
test  = engineer_features(test)

print(f"  New features added: Title, FamilySize, IsAlone, "
      f"FarePerPerson, Deck, AgeBand, FareBand")
print(f"  Train columns after engineering: {list(train.columns)}")


# ─────────────────────────────────────────────
#  4.  PREPROCESSING
# ─────────────────────────────────────────────
print("\n[4/7]  Preprocessing …")

# --- 4a. Impute missing values
# Age   → median by Title & Pclass
def impute_age(df):
    medians = df.groupby(["Title", "Pclass"])["Age"].median()
    def fill(row):
        if pd.isnull(row["Age"]):
            try:
                return medians.loc[(row["Title"], row["Pclass"])]
            except KeyError:
                return df["Age"].median()
        return row["Age"]
    df["Age"] = df.apply(fill, axis=1)
    return df

train = impute_age(train)
test  = impute_age(test)

# Embarked → mode
train["Embarked"].fillna(train["Embarked"].mode()[0], inplace=True)
test["Embarked"].fillna(test["Embarked"].mode()[0],   inplace=True)

# Fare (test only has 1 missing)
test["Fare"].fillna(test["Fare"].median(), inplace=True)

# AgeBand / FareBand derived from Age/Fare after imputation – recompute
for df in [train, test]:
    df["AgeBand"] = pd.cut(df["Age"],
                           bins=[0, 12, 18, 35, 60, 100],
                           labels=["Child", "Teen", "YoungAdult",
                                   "Adult", "Senior"],
                           right=False)
    df["FareBand"] = pd.qcut(df["Fare"].clip(upper=512),
                             q=4,
                             labels=["Low", "Mid", "High", "VHigh"])

# --- 4b. Encode categorical columns
cat_cols = ["Sex", "Embarked", "Title", "Deck", "AgeBand", "FareBand"]
le = LabelEncoder()

for col in cat_cols:
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col]  = le.transform(test[col].astype(str))

print("  Categorical columns encoded:", cat_cols)

# --- 4c. Define features & target
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
            "Embarked", "Title", "FamilySize", "IsAlone",
            "FarePerPerson", "Deck", "AgeBand", "FareBand"]

X      = train[FEATURES]
y      = train["Survived"]
X_test = test[FEATURES]

print(f"  Feature matrix shape : {X.shape}")
print(f"  Positive class rate  : {y.mean()*100:.1f}%")

# --- 4d. Scale
scaler  = StandardScaler()
X_scaled      = pd.DataFrame(scaler.fit_transform(X),  columns=FEATURES)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=FEATURES)

# --- 4e. Train / Validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)
print(f"  Train : {X_train.shape[0]} rows  |  Val : {X_val.shape[0]} rows")


# ─────────────────────────────────────────────
#  5.  MODEL TRAINING & COMPARISON
# ─────────────────────────────────────────────
print("\n[5/7]  Training & comparing models …")

models = {
    "Logistic Regression" : LogisticRegression(max_iter=500, random_state=42),
    "Decision Tree"       : DecisionTreeClassifier(random_state=42),
    "Random Forest"       : RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting"   : GradientBoostingClassifier(random_state=42),
    "XGBoost"             : XGBClassifier(use_label_encoder=False,
                                          eval_metric="logloss",
                                          random_state=42),
    "SVM"                 : SVC(probability=True, random_state=42),
}

results = {}
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train,
                                cv=5, scoring="accuracy")
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    val_acc  = accuracy_score(y_val, val_pred)
    results[name] = {
        "CV Mean"  : cv_scores.mean(),
        "CV Std"   : cv_scores.std(),
        "Val Acc"  : val_acc,
    }
    print(f"  {name:<25}  CV={cv_scores.mean()*100:.2f}% ±{cv_scores.std()*100:.2f}%"
          f"  Val={val_acc*100:.2f}%")

results_df = pd.DataFrame(results).T.sort_values("Val Acc", ascending=False)
print(f"\n  Best model : {results_df.index[0]}")

# Model comparison bar chart
fig, ax = plt.subplots(figsize=(10, 5))
results_df["Val Acc"].sort_values().plot.barh(
    ax=ax, color="#3498db", edgecolor="white")
ax.set_xlabel("Validation Accuracy")
ax.set_title("Model Comparison — Validation Accuracy")
ax.axvline(x=0.8, color="red", linestyle="--", linewidth=1, label="0.80 target")
ax.legend()
plt.tight_layout()
plt.savefig("titanic_model_comparison.png", dpi=120)
plt.close()
print("  Saved: titanic_model_comparison.png")


# ─────────────────────────────────────────────
#  6.  HYPERPARAMETER TUNING (Random Forest)
# ─────────────────────────────────────────────
print("\n[6/7]  Hyperparameter tuning (Random Forest) …")

param_grid = {
    "n_estimators"  : [100, 200, 300],
    "max_depth"     : [None, 5, 10, 15],
    "min_samples_split" : [2, 5, 10],
    "max_features"  : ["sqrt", "log2"],
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5,
                           scoring="accuracy", n_jobs=-1, verbose=0)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
print(f"  Best params : {grid_search.best_params_}")
print(f"  Best CV acc : {grid_search.best_score_*100:.2f}%")

# Also tune XGBoost
xgb_params = {
    "n_estimators"  : [100, 200],
    "max_depth"     : [3, 5, 7],
    "learning_rate" : [0.05, 0.1, 0.2],
    "subsample"     : [0.8, 1.0],
}
xgb = XGBClassifier(use_label_encoder=False,
                    eval_metric="logloss", random_state=42)
xgb_gs = GridSearchCV(xgb, xgb_params, cv=5,
                      scoring="accuracy", n_jobs=-1, verbose=0)
xgb_gs.fit(X_train, y_train)
best_xgb = xgb_gs.best_estimator_
print(f"  XGBoost best params : {xgb_gs.best_params_}")
print(f"  XGBoost best CV acc : {xgb_gs.best_score_*100:.2f}%")

# Ensemble (soft voting)
ensemble = VotingClassifier(
    estimators=[
        ("rf",  best_rf),
        ("xgb", best_xgb),
        ("lr",  LogisticRegression(max_iter=500, random_state=42)),
    ],
    voting="soft"
)
ensemble.fit(X_train, y_train)
ens_pred = ensemble.predict(X_val)
ens_acc  = accuracy_score(y_val, ens_pred)
print(f"  Ensemble (soft voting) Val Acc : {ens_acc*100:.2f}%")

# Choose final model
FINAL_MODEL_NAME = "Ensemble"
final_model      = ensemble


# ─────────────────────────────────────────────
#  7.  FINAL EVALUATION
# ─────────────────────────────────────────────
print("\n[7/7]  Final Evaluation on Validation Set …")

y_pred      = final_model.predict(X_val)
y_pred_prob = final_model.predict_proba(X_val)[:, 1]

print(f"\n  Accuracy  : {accuracy_score(y_val, y_pred)*100:.2f}%")
print(f"  ROC-AUC   : {roc_auc_score(y_val, y_pred_prob)*100:.2f}%")
print(f"\n  Classification Report :\n")
print(classification_report(y_val, y_pred,
                             target_names=["Died", "Survived"]))

# --- Confusion matrix
cm = confusion_matrix(y_val, y_pred)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Died", "Survived"],
            yticklabels=["Died", "Survived"],
            ax=axes[0])
axes[0].set_title("Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

# --- ROC Curve
fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
auc_score   = roc_auc_score(y_val, y_pred_prob)
axes[1].plot(fpr, tpr, color="#2ecc71", lw=2,
             label=f"ROC (AUC = {auc_score:.3f})")
axes[1].plot([0, 1], [0, 1], "k--", lw=1)
axes[1].set_xlim([0, 1]); axes[1].set_ylim([0, 1.02])
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curve")
axes[1].legend(loc="lower right")

plt.tight_layout()
plt.savefig("titanic_evaluation.png", dpi=120)
plt.close()
print("  Saved: titanic_evaluation.png")

# --- Feature importance (from best RF)
feat_imp = pd.Series(best_rf.feature_importances_,
                     index=FEATURES).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))
feat_imp.plot.bar(ax=ax, color="#9b59b6", edgecolor="white")
ax.set_title("Feature Importance (Random Forest)")
ax.set_ylabel("Importance")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("titanic_feature_importance.png", dpi=120)
plt.close()
print("  Saved: titanic_feature_importance.png")

print("\n  Top-5 important features :")
print(feat_imp.head())


# ─────────────────────────────────────────────
#  8.  GENERATE SUBMISSION FILE
# ─────────────────────────────────────────────
print("\n  Generating submission.csv …")

# Retrain final model on ALL training data for best possible prediction
final_model.fit(X_scaled, y)          # full train set (scaled)
test_predictions = final_model.predict(X_test_scaled)

submission = pd.DataFrame({
    "PassengerId" : test["PassengerId"],
    "Survived"    : test_predictions,
})
submission.to_csv("submission.csv", index=False)
print("  Saved: submission.csv  — ready to upload to Kaggle!")
print(f"  Predicted survivors in test set: {test_predictions.sum()} / {len(test_predictions)}")


# ─────────────────────────────────────────────
#  SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("  PIPELINE COMPLETE")
print("=" * 65)
print("""
  Files generated
  ───────────────
  titanic_eda.png              →  Exploratory data analysis charts
  titanic_model_comparison.png →  Accuracy of 6 baseline models
  titanic_evaluation.png       →  Confusion matrix + ROC curve
  titanic_feature_importance.png → Which features matter most
  submission.csv               →  Kaggle-ready predictions

  What you learned (pipeline stages)
  ───────────────────────────────────
  1. EDA           — understand missing data, distributions, correlations
  2. Feature Eng.  — extract Title, FamilySize, Deck, age/fare bands
  3. Imputation    — smart median imputation grouped by Title & Pclass
  4. Encoding      — LabelEncoder for all categoricals
  5. Scaling       — StandardScaler so distance-based models work well
  6. Baseline      — 6 models compared with 5-fold CV
  7. Tuning        — GridSearchCV on Random Forest & XGBoost
  8. Ensemble      — Soft-voting of RF + XGBoost + Logistic Regression
  9. Evaluation    — Accuracy, ROC-AUC, confusion matrix, feature importance
 10. Submission    — Retrain on full data → submission.csv
""")
