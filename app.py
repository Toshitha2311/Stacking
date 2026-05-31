import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    StackingClassifier, StackingRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score
)
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Stacking Ensemble Learning",
    page_icon="🧱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* Dark background */
.stApp {
    background: #0d0f14;
    color: #e8eaf0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #13161e;
    border-right: 1px solid #2a2d3a;
}

/* Hero header */
.hero-header {
    background: linear-gradient(135deg, #1a1f2e 0%, #0d0f14 60%);
    border: 1px solid #2a2d3a;
    border-radius: 16px;
    padding: 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(99,102,241,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.6rem;
    background: linear-gradient(90deg, #818cf8, #c084fc, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.hero-sub {
    color: #6b7280;
    font-size: 1rem;
    margin-top: 0.5rem;
    font-family: 'JetBrains Mono', monospace;
}

/* Metric cards */
.metric-card {
    background: #13161e;
    border: 1px solid #2a2d3a;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #818cf8; }
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    color: #818cf8;
}
.metric-label {
    font-size: 0.75rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.25rem;
}

/* Section headers */
.section-header {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.3rem;
    color: #c084fc;
    border-bottom: 1px solid #2a2d3a;
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem 0;
}

/* Concept cards */
.concept-card {
    background: #13161e;
    border: 1px solid #2a2d3a;
    border-left: 3px solid #818cf8;
    border-radius: 10px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}
.concept-title {
    font-weight: 700;
    color: #c084fc;
    font-size: 1rem;
    margin-bottom: 0.4rem;
}
.concept-body {
    color: #9ca3af;
    font-size: 0.88rem;
    line-height: 1.6;
}

/* Tags */
.tag {
    display: inline-block;
    background: rgba(129,140,248,0.12);
    color: #818cf8;
    border: 1px solid rgba(129,140,248,0.3);
    border-radius: 6px;
    padding: 0.15rem 0.5rem;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    margin: 0.15rem;
}

/* Stacking winner badge */
.winner-badge {
    background: linear-gradient(135deg, rgba(129,140,248,0.2), rgba(192,132,252,0.2));
    border: 1px solid #818cf8;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    text-align: center;
    font-weight: 700;
    color: #c084fc;
    font-size: 1.1rem;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #13161e;
    border-radius: 10px;
    gap: 4px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #6b7280;
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: rgba(129,140,248,0.15) !important;
    color: #818cf8 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border: none;
    border-radius: 10px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    padding: 0.6rem 1.5rem;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.88; }

/* Selectbox / slider labels */
label { color: #9ca3af !important; }

/* DataFrames */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* Divider */
hr { border-color: #2a2d3a; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MATPLOTLIB DARK THEME
# ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0d0f14",
    "axes.facecolor":    "#13161e",
    "axes.edgecolor":    "#2a2d3a",
    "axes.labelcolor":   "#9ca3af",
    "xtick.color":       "#6b7280",
    "ytick.color":       "#6b7280",
    "text.color":        "#e8eaf0",
    "grid.color":        "#1f2433",
    "grid.linewidth":    0.6,
    "axes.titlecolor":   "#c084fc",
    "axes.titlesize":    11,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "figure.dpi":        130,
    "font.family":       "monospace",
})
PALETTE = ["#818cf8", "#c084fc", "#f472b6", "#34d399", "#fbbf24", "#60a5fa"]


# ═══════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════
with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🧱 Stacking Lab")
    st.markdown("---")

    task = st.selectbox("**Task Type**", ["Classification", "Regression"])
    st.markdown("---")

    st.markdown("**Base Learners**")
    if task == "Classification":
        use_rf  = st.checkbox("Random Forest",        value=True)
        use_gb  = st.checkbox("Gradient Boosting",    value=True)
        use_svm = st.checkbox("SVM",                  value=True)
        use_knn = st.checkbox("K-Nearest Neighbors",  value=False)
        use_dt  = st.checkbox("Decision Tree",        value=False)
    else:
        use_rf  = st.checkbox("Random Forest",        value=True)
        use_gb  = st.checkbox("Gradient Boosting",    value=True)
        use_svr = st.checkbox("SVR",                  value=True)
        use_knn = st.checkbox("K-Nearest Neighbors",  value=False)
        use_dt  = st.checkbox("Decision Tree",        value=False)

    st.markdown("**Meta Learner**")
    if task == "Classification":
        meta_choice = st.selectbox("Select Meta Learner", ["Logistic Regression", "Random Forest"])
    else:
        meta_choice = st.selectbox("Select Meta Learner", ["Ridge Regression", "Random Forest"])

    st.markdown("---")
    test_size = st.slider("Test Set Size", 0.10, 0.40, 0.20, 0.05)
    cv_folds  = st.slider("CV Folds",      3,    10,   5,    1)
    run_btn   = st.button("🚀  Run Experiment")

    st.markdown("---")
    st.markdown("""
    <div style='color:#4b5563; font-size:0.75rem; line-height:1.6;'>
    <b style='color:#6b7280'>How stacking works</b><br>
    1. Base learners train on data<br>
    2. Their predictions become new features<br>
    3. Meta learner trains on those features<br>
    4. Final prediction from meta learner
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════
st.markdown("""
<div class="hero-header">
  <div class="hero-title">🧱 Stacking Ensemble Lab</div>
  <div class="hero-sub">
    Base Learners → Meta Learner → Superior Predictions
    &nbsp;·&nbsp; Classification &amp; Regression
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# CONCEPTS TAB + EXPERIMENT TAB
# ═══════════════════════════════════════════════
tab_concepts, tab_exp, tab_code = st.tabs(
    ["📚  Concepts", "⚗️  Experiment", "💻  Code"]
)

# ─────────────────────────────────────────────
# TAB 1 – CONCEPTS
# ─────────────────────────────────────────────
with tab_concepts:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-header">Core Concepts</div>', unsafe_allow_html=True)

        for title, body in [
            ("🔵 Base Learners",
             "Multiple diverse models trained on the original training data. "
             "Each model specialises in capturing different patterns. "
             "Their out-of-fold predictions form a new feature matrix for the meta learner."),
            ("🟣 Meta Learner",
             "A higher-level model that takes base-learner predictions as input. "
             "It learns how to best combine them, correcting individual errors "
             "and exploiting complementary strengths."),
            ("⚙️ Model Training",
             "Base learners use K-Fold cross-validation to produce out-of-fold predictions "
             "(avoids data leakage). The meta learner then trains on these stacked predictions, "
             "and the final model predicts on unseen test data."),
            ("📊 Performance Comparison",
             "Stacking typically outperforms any single base learner by reducing both bias "
             "and variance. Gains are largest when base learners are diverse and individually strong."),
        ]:
            st.markdown(f"""
            <div class="concept-card">
              <div class="concept-title">{title}</div>
              <div class="concept-body">{body}</div>
            </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-header">Stacking Architecture</div>', unsafe_allow_html=True)

        # Simple architecture diagram via matplotlib
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")

        # Training data box
        ax.add_patch(mpatches.FancyBboxPatch((3.5,6.6), 3, 0.9, boxstyle="round,pad=0.1",
            fc="#1e2130", ec="#818cf8", lw=1.5))
        ax.text(5, 7.05, "Training Data", ha="center", va="center",
                color="#e8eaf0", fontsize=8, fontweight="bold")

        # Base learners
        base_x = [1.2, 4.0, 6.8, 9.2]
        base_labels = ["RF", "GB", "SVM", "KNN"]
        for i, (bx, bl) in enumerate(zip(base_x[:3], base_labels[:3])):
            ax.add_patch(mpatches.FancyBboxPatch((bx-0.9, 4.5), 1.8, 0.9, boxstyle="round,pad=0.1",
                fc="#1e2130", ec=PALETTE[i], lw=1.4))
            ax.text(bx, 4.95, bl, ha="center", va="center",
                    color=PALETTE[i], fontsize=8, fontweight="bold")
            # Arrow from data to base
            ax.annotate("", xy=(bx, 5.4), xytext=(5, 6.6),
                arrowprops=dict(arrowstyle="-|>", color="#4b5563", lw=1))

        # Stacked predictions box
        ax.add_patch(mpatches.FancyBboxPatch((2.5, 2.8), 5, 0.9, boxstyle="round,pad=0.1",
            fc="#1e2130", ec="#c084fc", lw=1.5, linestyle="--"))
        ax.text(5, 3.25, "Stacked Predictions (new features)", ha="center", va="center",
                color="#c084fc", fontsize=7.5)

        # Arrows from base to stacked
        for bx in [1.2, 4.0, 6.8]:
            ax.annotate("", xy=(5, 2.8), xytext=(bx, 4.5),
                arrowprops=dict(arrowstyle="-|>", color="#4b5563", lw=1))

        # Meta learner
        ax.add_patch(mpatches.FancyBboxPatch((3.5, 1.2), 3, 0.9, boxstyle="round,pad=0.1",
            fc="#1e2130", ec="#f472b6", lw=1.5))
        ax.text(5, 1.65, "Meta Learner", ha="center", va="center",
                color="#f472b6", fontsize=8, fontweight="bold")
        ax.annotate("", xy=(5, 1.2), xytext=(5, 2.8),
            arrowprops=dict(arrowstyle="-|>", color="#c084fc", lw=1.5))

        # Final output
        ax.add_patch(mpatches.FancyBboxPatch((3.5, 0.0), 3, 0.8, boxstyle="round,pad=0.1",
            fc=(0.388, 0.4, 0.945, 0.2), ec="#818cf8", lw=1.5))
        ax.text(5, 0.4, "Final Prediction", ha="center", va="center",
                color="#818cf8", fontsize=8, fontweight="bold")
        ax.annotate("", xy=(5, 0.8), xytext=(5, 1.2),
            arrowprops=dict(arrowstyle="-|>", color="#f472b6", lw=1.5))

        st.pyplot(fig, width='stretch')
        plt.close()

        st.markdown('<div class="section-header">Key Advantages</div>', unsafe_allow_html=True)
        cols = st.columns(2)
        tags = ["Reduces Variance", "Reduces Bias", "Combines Diverse Models",
                "Better Generalisation", "Learns Optimal Weights", "Handles Non-linearity"]
        for i, tag in enumerate(tags):
            cols[i % 2].markdown(f'<span class="tag">{tag}</span>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TAB 2 – EXPERIMENT
# ─────────────────────────────────────────────
with tab_exp:
    if not run_btn:
        st.info("👈  Configure your experiment in the sidebar and click **Run Experiment**.")
    else:
        # ── DATA ──────────────────────────────────────────────────
        with st.spinner("Loading data & training models …"):
            if task == "Classification":
                dataset = load_breast_cancer()
                X, y = dataset.data, dataset.target
                feature_names = dataset.feature_names
            else:
                dataset = load_diabetes()
                X, y = dataset.data, dataset.target
                feature_names = dataset.feature_names

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42,
                stratify=y if task == "Classification" else None
            )

            # ── BUILD BASE LEARNERS ───────────────────────────────
            if task == "Classification":
                all_bases = {
                    "Random Forest":       (use_rf,  RandomForestClassifier(n_estimators=100, random_state=42)),
                    "Gradient Boosting":   (use_gb,  GradientBoostingClassifier(n_estimators=100, random_state=42)),
                    "SVM":                 (use_svm, SVC(probability=True, random_state=42)),
                    "K-Nearest Neighbors": (use_knn, KNeighborsClassifier()),
                    "Decision Tree":       (use_dt,  DecisionTreeClassifier(random_state=42)),
                }
                meta_model = (LogisticRegression(max_iter=1000)
                              if meta_choice == "Logistic Regression"
                              else RandomForestClassifier(n_estimators=50, random_state=42))
            else:
                all_bases = {
                    "Random Forest":       (use_rf,  RandomForestRegressor(n_estimators=100, random_state=42)),
                    "Gradient Boosting":   (use_gb,  GradientBoostingRegressor(n_estimators=100, random_state=42)),
                    "SVR":                 (use_svr, SVR()),
                    "K-Nearest Neighbors": (use_knn, KNeighborsRegressor()),
                    "Decision Tree":       (use_dt,  DecisionTreeRegressor(random_state=42)),
                }
                meta_model = (Ridge()
                              if meta_choice == "Ridge Regression"
                              else RandomForestRegressor(n_estimators=50, random_state=42))

            base_estimators = [(name, model)
                               for name, (enabled, model) in all_bases.items() if enabled]

            if len(base_estimators) < 2:
                st.error("Please select at least 2 base learners.")
                st.stop()

            # ── STACKING MODEL ────────────────────────────────────
            cv_obj = (StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                      if task == "Classification"
                      else KFold(n_splits=cv_folds, shuffle=True, random_state=42))

            if task == "Classification":
                stacking_model = StackingClassifier(
                    estimators=base_estimators,
                    final_estimator=meta_model,
                    cv=cv_obj, passthrough=False, n_jobs=-1
                )
            else:
                stacking_model = StackingRegressor(
                    estimators=base_estimators,
                    final_estimator=meta_model,
                    cv=cv_obj, passthrough=False, n_jobs=-1
                )

            stacking_model.fit(X_train, y_train)

            # ── INDIVIDUAL MODELS ─────────────────────────────────
            individual_results = {}
            for name, model in base_estimators:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                if task == "Classification":
                    individual_results[name] = {
                        "Accuracy":  accuracy_score(y_test, y_pred),
                        "ROC-AUC":   roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]),
                        "model": model
                    }
                else:
                    individual_results[name] = {
                        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                        "MAE":  mean_absolute_error(y_test, y_pred),
                        "R²":   r2_score(y_test, y_pred),
                        "model": model
                    }

            # ── STACKING RESULTS ──────────────────────────────────
            y_pred_stack = stacking_model.predict(X_test)
            if task == "Classification":
                stack_results = {
                    "Accuracy": accuracy_score(y_test, y_pred_stack),
                    "ROC-AUC":  roc_auc_score(y_test, stacking_model.predict_proba(X_test)[:, 1]),
                }
            else:
                stack_results = {
                    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_stack)),
                    "MAE":  mean_absolute_error(y_test, y_pred_stack),
                    "R²":   r2_score(y_test, y_pred_stack),
                }

        st.success("✅ Training complete!")

        # ── SUMMARY METRICS ────────────────────────────────────────
        st.markdown('<div class="section-header">Stacking Model — Summary</div>',
                    unsafe_allow_html=True)

        if task == "Classification":
            mcols = st.columns(4)
            mcols[0].markdown(f"""<div class="metric-card">
                <div class="metric-value">{stack_results['Accuracy']:.3f}</div>
                <div class="metric-label">Accuracy</div></div>""", unsafe_allow_html=True)
            mcols[1].markdown(f"""<div class="metric-card">
                <div class="metric-value">{stack_results['ROC-AUC']:.3f}</div>
                <div class="metric-label">ROC-AUC</div></div>""", unsafe_allow_html=True)
            mcols[2].markdown(f"""<div class="metric-card">
                <div class="metric-value">{len(base_estimators)}</div>
                <div class="metric-label">Base Learners</div></div>""", unsafe_allow_html=True)
            mcols[3].markdown(f"""<div class="metric-card">
                <div class="metric-value">{cv_folds}</div>
                <div class="metric-label">CV Folds</div></div>""", unsafe_allow_html=True)
        else:
            mcols = st.columns(4)
            mcols[0].markdown(f"""<div class="metric-card">
                <div class="metric-value">{stack_results['R²']:.3f}</div>
                <div class="metric-label">R² Score</div></div>""", unsafe_allow_html=True)
            mcols[1].markdown(f"""<div class="metric-card">
                <div class="metric-value">{stack_results['RMSE']:.1f}</div>
                <div class="metric-label">RMSE</div></div>""", unsafe_allow_html=True)
            mcols[2].markdown(f"""<div class="metric-card">
                <div class="metric-value">{stack_results['MAE']:.1f}</div>
                <div class="metric-label">MAE</div></div>""", unsafe_allow_html=True)
            mcols[3].markdown(f"""<div class="metric-card">
                <div class="metric-value">{cv_folds}</div>
                <div class="metric-label">CV Folds</div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── COMPARISON TABLE ───────────────────────────────────────
        st.markdown('<div class="section-header">Performance Comparison</div>',
                    unsafe_allow_html=True)

        rows = []
        for name, res in individual_results.items():
            row = {"Model": name, "Type": "Base Learner"}
            row.update({k: v for k, v in res.items() if k != "model"})
            rows.append(row)
        rows.append({"Model": f"Stacking ({meta_choice})", "Type": "Meta Ensemble",
                     **stack_results})

        df_comp = pd.DataFrame(rows).set_index("Model")
        numeric_cols = [c for c in df_comp.columns if c != "Type"]

        def style_df(df):
            styles = []
            for col in numeric_cols:
                if col in ("Accuracy", "ROC-AUC", "R²"):
                    styles.append(df[col].apply(
                        lambda v: f"color: {'#34d399' if v == df[col].max() else '#e8eaf0'}"))
                else:  # RMSE, MAE – lower is better
                    styles.append(df[col].apply(
                        lambda v: f"color: {'#34d399' if v == df[col].min() else '#e8eaf0'}"))
            return styles

        st.dataframe(
            df_comp.style.set_properties(**{
                "background-color": "#13161e",
                "color": "#e8eaf0",
                "border": "1px solid #2a2d3a",
                "font-family": "JetBrains Mono, monospace",
                "font-size": "0.82rem",
            }).format({col: "{:.4f}" for col in numeric_cols}),
            width='stretch'
        )

        # ── CHARTS ────────────────────────────────────────────────
        st.markdown('<div class="section-header">Visualisations</div>',
                    unsafe_allow_html=True)

        if task == "Classification":
            ch1, ch2 = st.columns(2)

            # Bar comparison
            with ch1:
                fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.6))
                for ax_i, metric in enumerate(["Accuracy", "ROC-AUC"]):
                    vals  = [individual_results[n][metric] for n in individual_results] + [stack_results[metric]]
                    names = list(individual_results.keys()) + [f"Stacking"]
                    colors= [PALETTE[i % len(PALETTE)] for i in range(len(names) - 1)] + ["#f472b6"]
                    bars  = axes[ax_i].bar(range(len(names)), vals, color=colors,
                                           edgecolor="#2a2d3a", linewidth=0.6)
                    axes[ax_i].set_xticks(range(len(names)))
                    axes[ax_i].set_xticklabels(
                        [n[:6] for n in names], rotation=35, ha="right", fontsize=7)
                    axes[ax_i].set_title(metric)
                    axes[ax_i].set_ylim(min(vals) * 0.97, 1.02)
                    axes[ax_i].yaxis.grid(True, alpha=0.4)
                    # Highlight stacking bar
                    bars[-1].set_edgecolor("#f472b6")
                    bars[-1].set_linewidth(2)
                    for bar, val in zip(bars, vals):
                        axes[ax_i].text(bar.get_x() + bar.get_width()/2,
                                        bar.get_height() + 0.002,
                                        f"{val:.3f}", ha="center", fontsize=6.5, color="#9ca3af")
                fig.suptitle("Base Learners vs Stacking", color="#c084fc", fontsize=10)
                fig.tight_layout()
                st.pyplot(fig, width='stretch')
                plt.close()

            # ROC Curves
            with ch2:
                fig, ax = plt.subplots(figsize=(5, 3.8))
                for i, (name, res) in enumerate(individual_results.items()):
                    fpr, tpr, _ = roc_curve(y_test, res["model"].predict_proba(X_test)[:, 1])
                    ax.plot(fpr, tpr, color=PALETTE[i], lw=1.2,
                            label=f"{name[:10]} ({res['ROC-AUC']:.3f})", alpha=0.7)
                fpr_s, tpr_s, _ = roc_curve(y_test, stacking_model.predict_proba(X_test)[:, 1])
                ax.plot(fpr_s, tpr_s, color="#f472b6", lw=2.2,
                        label=f"Stacking ({stack_results['ROC-AUC']:.3f})")
                ax.plot([0,1],[0,1], ":", color="#4b5563", lw=1)
                ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curves")
                ax.legend(fontsize=6.5, loc="lower right",
                          facecolor="#13161e", edgecolor="#2a2d3a", labelcolor="#9ca3af")
                fig.tight_layout()
                st.pyplot(fig, width='stretch')
                plt.close()

            # Confusion matrix for stacking
            st.markdown('<div class="section-header">Confusion Matrix — Stacking Model</div>',
                        unsafe_allow_html=True)
            cm = confusion_matrix(y_test, y_pred_stack)
            fig, ax = plt.subplots(figsize=(4, 3.2))
            sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="magma",
                        linecolor="#2a2d3a", linewidths=0.5,
                        annot_kws={"color": "#e8eaf0", "size": 11})
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix", color="#c084fc")
            fig.tight_layout()
            st.pyplot(fig, width='stretch')
            plt.close()

        else:  # Regression
            ch1, ch2 = st.columns(2)

            # Metric bar charts
            with ch1:
                fig, axes = plt.subplots(1, 3, figsize=(7, 3.4))
                for ax_i, (metric, lower_better) in enumerate(
                        [("R²", False), ("RMSE", True), ("MAE", True)]):
                    vals  = [individual_results[n][metric] for n in individual_results] + [stack_results[metric]]
                    names = list(individual_results.keys()) + ["Stacking"]
                    colors= [PALETTE[i % len(PALETTE)] for i in range(len(names) - 1)] + ["#f472b6"]
                    bars  = axes[ax_i].bar(range(len(names)), vals, color=colors,
                                           edgecolor="#2a2d3a", linewidth=0.5)
                    axes[ax_i].set_xticks(range(len(names)))
                    axes[ax_i].set_xticklabels(
                        [n[:5] for n in names], rotation=40, ha="right", fontsize=7)
                    axes[ax_i].set_title(metric)
                    axes[ax_i].yaxis.grid(True, alpha=0.4)
                    bars[-1].set_edgecolor("#f472b6"); bars[-1].set_linewidth(2)
                fig.suptitle("Metric Comparison", color="#c084fc", fontsize=10)
                fig.tight_layout()
                st.pyplot(fig, width='stretch')
                plt.close()

            # Actual vs Predicted
            with ch2:
                fig, ax = plt.subplots(figsize=(5, 3.8))
                ax.scatter(y_test, y_pred_stack, color="#818cf8", alpha=0.5, s=18, edgecolors="none")
                mn, mx = y_test.min(), y_test.max()
                ax.plot([mn, mx], [mn, mx], color="#f472b6", lw=1.5, ls="--")
                ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
                ax.set_title(f"Stacking — Actual vs Predicted (R²={stack_results['R²']:.3f})")
                fig.tight_layout()
                st.pyplot(fig, width='stretch')
                plt.close()

            # Residuals
            residuals = y_test - y_pred_stack
            fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
            axes[0].scatter(y_pred_stack, residuals, color="#c084fc", alpha=0.5, s=14)
            axes[0].axhline(0, color="#f472b6", lw=1.5, ls="--")
            axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Residual")
            axes[0].set_title("Residual Plot")
            axes[1].hist(residuals, bins=25, color="#818cf8", edgecolor="#2a2d3a", lw=0.4, alpha=0.85)
            axes[1].set_title("Residual Distribution")
            axes[1].set_xlabel("Residual"); axes[1].set_ylabel("Count")
            fig.tight_layout()
            st.pyplot(fig, width='stretch')
            plt.close()

        # ── WINNER ────────────────────────────────────────────────
        primary = "Accuracy" if task == "Classification" else "R²"
        best_base = max(individual_results, key=lambda n: individual_results[n][primary])
        best_base_val = individual_results[best_base][primary]
        stack_val     = stack_results[primary]
        improvement   = (stack_val - best_base_val) / best_base_val * 100

        st.markdown("<br>", unsafe_allow_html=True)
        if improvement > 0:
            st.markdown(f"""
            <div class="winner-badge">
              🏆 Stacking beats best base learner ({best_base}) by
              <span style='color:#34d399'>+{improvement:.2f}%</span> in {primary}
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="winner-badge" style='border-color:#fbbf24; color:#fbbf24;'>
              ⚠️ Best base learner ({best_base}) ties or beats stacking by {abs(improvement):.2f}% in {primary}
              — try adding more diverse base learners.
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TAB 3 – CODE
# ─────────────────────────────────────────────
with tab_code:
    st.markdown('<div class="section-header">Stacking Classification — Core Code</div>',
                unsafe_allow_html=True)
    st.code("""
from sklearn.ensemble import (StackingClassifier, RandomForestClassifier,
                               GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score

# 1. Load & preprocess
X, y = load_breast_cancer(return_X_y=True)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Define BASE LEARNERS
base_learners = [
    ("rf",  RandomForestClassifier(n_estimators=100, random_state=42)),
    ("gb",  GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ("svm", SVC(probability=True, random_state=42)),
]

# 3. Define META LEARNER
meta_learner = LogisticRegression(max_iter=1000)

# 4. Build Stacking model
stacking_clf = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    passthrough=False,
    n_jobs=-1,
)

# 5. Train & Evaluate
stacking_clf.fit(X_train, y_train)
y_pred = stacking_clf.predict(X_test)

print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, stacking_clf.predict_proba(X_test)[:,1]):.4f}")
""", language="python")

    st.markdown('<div class="section-header">Stacking Regression — Core Code</div>',
                unsafe_allow_html=True)
    st.code("""
from sklearn.ensemble import (StackingRegressor, RandomForestRegressor,
                               GradientBoostingRegressor)
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 1. Load & preprocess
X, y = load_diabetes(return_X_y=True)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Define BASE LEARNERS
base_learners = [
    ("rf",  RandomForestRegressor(n_estimators=100, random_state=42)),
    ("gb",  GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ("svr", SVR()),
]

# 3. Define META LEARNER
meta_learner = Ridge()

# 4. Build Stacking model
stacking_reg = StackingRegressor(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    passthrough=False,
    n_jobs=-1,
)

# 5. Train & Evaluate
stacking_reg.fit(X_train, y_train)
y_pred = stacking_reg.predict(X_test)

print(f"RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R²   : {r2_score(y_test, y_pred):.4f}")
""", language="python")

    st.markdown('<div class="section-header">How Stacking Avoids Data Leakage</div>',
                unsafe_allow_html=True)
    st.code("""
# StackingClassifier internally uses K-Fold cross-validation
# to generate out-of-fold (OOF) predictions for each base learner.
# These OOF predictions form the training set for the meta learner,
# ensuring NO data leakage — the meta learner never sees predictions
# made on data that was used to train a base learner.
#
#   ┌─────────────────────────────────────────────────┐
#   │ Fold 1: train on folds 2-5, predict on fold 1   │
#   │ Fold 2: train on folds 1,3-5, predict on fold 2 │
#   │ ...                                              │
#   │ Stacked OOF predictions → meta learner input     │
#   └─────────────────────────────────────────────────┘
""", language="python")