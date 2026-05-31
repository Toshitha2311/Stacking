"""
CNN Assignment — CIFAR-10 Image Classifier
==========================================
Dataset    : CIFAR-10 (10 classes, 60 000 images)
Model      : Custom CNN (Conv→BN→Pool blocks + Dense head)
Frontend   : Streamlit with custom dark-neon aesthetic
Backend    : TensorFlow / Keras model pipeline
Testing    : Accuracy, per-class metrics, confusion matrix,
             filter visualisation, activation maps
Deployment : `streamlit run cnn_app.py`
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image
import io, time, warnings
warnings.filterwarnings("ignore")

# ── TF quiet import ──────────────────────────────────────────────────────────
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score)

tf.get_logger().setLevel("ERROR")

# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CNN · CIFAR-10 Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

CLASSES = ["airplane","automobile","bird","cat","deer",
           "dog","frog","horse","ship","truck"]
CLASS_EMOJI = ["✈️","🚗","🐦","🐱","🦌","🐶","🐸","🐴","🚢","🚛"]

# ════════════════════════════════════════════════════════════════════════════
# CSS
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Exo+2:wght@300;400;600&display=swap');

html,body,[class*="css"]{ font-family:'Exo 2',sans-serif; }

.stApp{
  background: radial-gradient(ellipse at 20% 10%, #0a1628 0%, #060810 55%, #0d0a1a 100%);
  color:#c9d1e3;
}
[data-testid="stSidebar"]{
  background: linear-gradient(180deg,#080d1a 0%,#0c1020 100%);
  border-right:1px solid #1a2540;
}

/* HERO */
.hero{
  background:linear-gradient(135deg,#0d1b2a 0%,#050810 100%);
  border:1px solid #1a3050;
  border-radius:18px;padding:2.5rem 2.8rem;
  position:relative;overflow:hidden;margin-bottom:1.8rem;
}
.hero::after{
  content:'';position:absolute;top:-80px;right:-80px;
  width:320px;height:320px;
  background:radial-gradient(circle,rgba(0,212,255,.08) 0%,transparent 70%);
  border-radius:50%;
}
.hero-title{
  font-family:'Orbitron',monospace;font-weight:900;font-size:2.4rem;
  color:#00d4ff;text-shadow:0 0 30px rgba(0,212,255,.4);margin:0;
  letter-spacing:.04em;
}
.hero-sub{
  font-family:'Share Tech Mono',monospace;color:#4a7a9b;
  font-size:.9rem;margin-top:.5rem;
}
.badge{
  display:inline-block;background:rgba(0,212,255,.1);
  border:1px solid rgba(0,212,255,.3);border-radius:6px;
  padding:.15rem .6rem;font-size:.72rem;color:#00d4ff;
  font-family:'Share Tech Mono',monospace;margin:.15rem .1rem;
}

/* CARDS */
.card{
  background:#090e1a;border:1px solid #1a2540;border-radius:14px;
  padding:1.2rem 1.4rem;margin-bottom:1rem;
}
.card-accent{ border-left:3px solid #00d4ff; }
.metric-val{
  font-family:'Orbitron',monospace;font-size:2rem;font-weight:700;
  color:#00d4ff;text-shadow:0 0 15px rgba(0,212,255,.35);
}
.metric-lbl{
  font-size:.7rem;color:#4a7a9b;text-transform:uppercase;
  letter-spacing:.1em;margin-top:.2rem;
}
.section-hdr{
  font-family:'Orbitron',monospace;font-size:.95rem;font-weight:700;
  color:#00d4ff;letter-spacing:.08em;
  border-bottom:1px solid #1a2540;padding-bottom:.4rem;margin:1.4rem 0 .8rem;
}
.info-box{
  background:rgba(0,212,255,.04);border:1px solid rgba(0,212,255,.15);
  border-radius:10px;padding:1rem 1.2rem;color:#7a9ab8;
  font-size:.83rem;line-height:1.7;
}

/* Buttons */
.stButton>button{
  background:linear-gradient(135deg,#00576e,#003d52);
  color:#00d4ff;border:1px solid #00d4ff;border-radius:10px;
  font-family:'Orbitron',monospace;font-weight:700;font-size:.8rem;
  letter-spacing:.06em;padding:.55rem 1.2rem;width:100%;
  box-shadow:0 0 18px rgba(0,212,255,.15);transition:.2s;
}
.stButton>button:hover{
  box-shadow:0 0 28px rgba(0,212,255,.35);
  background:linear-gradient(135deg,#006d8a,#004d66);
}
label{color:#4a7a9b!important;}
.stTabs [data-baseweb="tab-list"]{
  background:#090e1a;border-radius:10px;gap:3px;padding:3px;
}
.stTabs [data-baseweb="tab"]{
  color:#4a7a9b;font-family:'Orbitron',monospace;font-size:.75rem;
}
.stTabs [aria-selected="true"]{
  background:rgba(0,212,255,.12)!important;color:#00d4ff!important;
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# MATPLOTLIB THEME
# ════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "figure.facecolor":"#060810","axes.facecolor":"#090e1a",
    "axes.edgecolor":"#1a2540","axes.labelcolor":"#7a9ab8",
    "xtick.color":"#4a7a9b","ytick.color":"#4a7a9b",
    "text.color":"#c9d1e3","grid.color":"#0f1828","grid.linewidth":.5,
    "axes.titlecolor":"#00d4ff","axes.titlesize":10,"axes.labelsize":8,
    "xtick.labelsize":7,"ytick.labelsize":7,"figure.dpi":130,
    "font.family":"monospace",
})
NEON = "#00d4ff"; ACCENT2="#7c3aed"; ACCENT3="#f59e0b"

# ════════════════════════════════════════════════════════════════════════════
# DATA & MODEL HELPERS
# ════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data(n_train=10000, n_test=2000):
    (x_train, y_train),(x_test, y_test) = cifar10.load_data()
    x_train = x_train[:n_train].astype("float32") / 255.
    x_test  = x_test[:n_test].astype("float32")  / 255.
    y_train = y_train[:n_train].flatten()
    y_test  = y_test[:n_test].flatten()
    return x_train, y_train, x_test, y_test

def build_cnn(num_filters1=32, num_filters2=64, dropout=0.4, use_bn=True):
    inp = keras.Input(shape=(32,32,3))
    # Block 1
    x = layers.Conv2D(num_filters1,3,padding="same",activation="relu")(inp)
    if use_bn: x = layers.BatchNormalization()(x)
    x = layers.Conv2D(num_filters1,3,padding="same",activation="relu")(x)
    if use_bn: x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(dropout/2)(x)
    # Block 2
    x = layers.Conv2D(num_filters2,3,padding="same",activation="relu")(x)
    if use_bn: x = layers.BatchNormalization()(x)
    x = layers.Conv2D(num_filters2,3,padding="same",activation="relu")(x)
    if use_bn: x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(dropout/2)(x)
    # Block 3
    x = layers.Conv2D(128,3,padding="same",activation="relu")(x)
    if use_bn: x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    # Head
    x = layers.Dense(256,activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(10,activation="softmax")(x)
    model = models.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def get_filter_mosaic(model, layer_name="conv2d"):
    for layer in model.layers:
        if layer_name in layer.name and len(layer.get_weights()) > 0:
            w = layer.get_weights()[0]          # (kH,kW,inC,outC)
            n = min(16, w.shape[-1])
            fig, axes = plt.subplots(2, n//2, figsize=(n//2*1.4, 3.2))
            for i, ax in enumerate(axes.flat):
                f = w[:,:,:,i]
                f = (f - f.min())/(f.max()-f.min()+1e-8)
                ax.imshow(f if f.shape[2]==3 else f[:,:,0], cmap="plasma")
                ax.axis("off")
            fig.suptitle(f"Learned Filters — {layer.name}", color=NEON, fontsize=9)
            fig.tight_layout()
            return fig
    return None

def get_activation_maps(model, img, layer_idx=2):
    sub = keras.Model(model.input, model.layers[layer_idx].output)
    acts = sub.predict(img[None], verbose=0)[0]
    n = min(16, acts.shape[-1])
    fig, axes = plt.subplots(2, n//2, figsize=(n//2*1.4, 3.2))
    for i, ax in enumerate(axes.flat):
        ax.imshow(acts[:,:,i], cmap="viridis")
        ax.axis("off")
    fig.suptitle(f"Activation Maps — layer {layer_idx}", color=NEON, fontsize=9)
    fig.tight_layout()
    return fig

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🔬 CNN Control Panel")
    st.markdown("---")
    st.markdown("**Dataset**")
    n_train = st.select_slider("Training samples",
        options=[2000,5000,10000,20000,50000], value=10000)
    n_test  = st.select_slider("Test samples",
        options=[500,1000,2000,5000,10000], value=2000)

    st.markdown("**Architecture**")
    f1 = st.select_slider("Conv Block 1 Filters", [16,32,64], value=32)
    f2 = st.select_slider("Conv Block 2 Filters", [32,64,128], value=64)
    use_bn = st.checkbox("Batch Normalisation", value=True)
    dropout = st.slider("Dropout Rate", 0.1, 0.6, 0.4, 0.05)

    st.markdown("**Training**")
    epochs    = st.slider("Max Epochs",    5, 50, 20)
    batch_sz  = st.selectbox("Batch Size", [32,64,128,256], index=1)
    use_aug   = st.checkbox("Data Augmentation", value=True)
    early_stop= st.checkbox("Early Stopping",    value=True)

    st.markdown("---")
    train_btn = st.button("⚡  Train CNN")

# ════════════════════════════════════════════════════════════════════════════
# HERO
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-title">🔬 CNN · CIFAR-10 CLASSIFIER</div>
  <div class="hero-sub">Convolutional Neural Network | Image Recognition | 10 Classes</div>
  <div style="margin-top:.8rem">
    <span class="badge">CIFAR-10</span>
    <span class="badge">Conv2D → BN → Pool</span>
    <span class="badge">GlobalAvgPool</span>
    <span class="badge">TensorFlow/Keras</span>
    <span class="badge">Streamlit</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════
tabs = st.tabs(["📐 Architecture","🏋️ Train & Results","🧪 Test Image","📊 Analysis","💻 Code"])

# ── TAB 0 : ARCHITECTURE ────────────────────────────────────────────────────
with tabs[0]:
    c1, c2 = st.columns([1.1, 1])

    with c1:
        st.markdown('<div class="section-hdr">CNN ARCHITECTURE</div>', unsafe_allow_html=True)
        for title, body in [
            ("BLOCK 1 — Feature Extraction",
             f"Conv2D({f1}) → {'BN → ' if use_bn else ''}ReLU → "
             f"Conv2D({f1}) → {'BN → ' if use_bn else ''}ReLU → MaxPool2D → Dropout({dropout/2:.2f})"),
            ("BLOCK 2 — Deep Features",
             f"Conv2D({f2}) → {'BN → ' if use_bn else ''}ReLU → "
             f"Conv2D({f2}) → {'BN → ' if use_bn else ''}ReLU → MaxPool2D → Dropout({dropout/2:.2f})"),
            ("BLOCK 3 — High-Level Features",
             "Conv2D(128) → BatchNorm → ReLU → GlobalAveragePooling2D"),
            ("CLASSIFICATION HEAD",
             f"Dense(256, ReLU) → Dropout({dropout}) → Dense(10, Softmax)"),
        ]:
            st.markdown(f"""
            <div class="card card-accent">
              <div style="font-family:'Orbitron',monospace;font-size:.8rem;
                          color:#00d4ff;margin-bottom:.4rem">{title}</div>
              <div style="font-family:'Share Tech Mono',monospace;font-size:.8rem;
                          color:#7a9ab8;line-height:1.7">{body}</div>
            </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-hdr">DATASET OVERVIEW</div>', unsafe_allow_html=True)

        # Sample grid
        try:
            x_tr, y_tr, _, _ = load_data(n_train, n_test)
            fig, axes = plt.subplots(2, 5, figsize=(7, 3.2))
            for cls_idx, ax in enumerate(axes.flat):
                idx = np.where(y_tr == cls_idx)[0][0]
                ax.imshow(x_tr[idx])
                ax.set_title(f"{CLASS_EMOJI[cls_idx]} {CLASSES[cls_idx]}", fontsize=7, color=NEON)
                ax.axis("off")
            fig.suptitle("CIFAR-10 Classes", color=NEON, fontsize=9)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
        except Exception:
            st.info("Sample images load after dataset downloads.")

        st.markdown("""
        <div class="info-box">
          <b style="color:#00d4ff">CIFAR-10 Dataset</b><br>
          • 60 000 colour images · 32×32 pixels<br>
          • 10 balanced classes, 6 000 images each<br>
          • 50 000 train / 10 000 test split<br>
          • Normalised to [0, 1] per channel
        </div>""", unsafe_allow_html=True)

# ── TAB 1 : TRAIN ────────────────────────────────────────────────────────────
with tabs[1]:
    if not train_btn:
        st.markdown("""
        <div class="info-box" style="text-align:center;padding:2rem;">
          Configure your CNN in the sidebar and click <b style="color:#00d4ff">⚡ Train CNN</b>
        </div>""", unsafe_allow_html=True)
    else:
        prog_bar = st.progress(0, text="Loading CIFAR-10 data …")
        x_train, y_train, x_test, y_test = load_data(n_train, n_test)
        prog_bar.progress(10, text="Building model …")

        model = build_cnn(f1, f2, dropout, use_bn)

        # Optional augmentation
        if use_aug:
            datagen = keras.preprocessing.image.ImageDataGenerator(
                horizontal_flip=True,
                width_shift_range=0.1,
                height_shift_range=0.1,
                rotation_range=10,
            )
            datagen.fit(x_train)
            train_gen = datagen.flow(x_train, y_train, batch_size=batch_sz)
        else:
            train_gen = None

        cbs = []
        if early_stop:
            cbs += [EarlyStopping(patience=5, restore_best_weights=True),
                    ReduceLROnPlateau(patience=3, factor=.5, verbose=0)]

        prog_bar.progress(20, text="Training … (this may take a moment)")

        history_container = st.empty()
        history = {"loss":[], "val_loss":[], "accuracy":[], "val_accuracy":[]}

        # Custom epoch callback for live chart update
        class LiveCallback(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                for k in history:
                    history[k].append(logs.get(k, 0))
                frac = 20 + int(78 * (epoch+1) / epochs)
                prog_bar.progress(min(frac,98),
                    text=f"Epoch {epoch+1}/{epochs} — loss {logs.get('loss',0):.4f} "
                         f"— val_acc {logs.get('val_accuracy',0):.4f}")

        fit_kwargs = dict(
            validation_data=(x_test, y_test),
            epochs=epochs, batch_size=batch_sz,
            callbacks=cbs+[LiveCallback()], verbose=0
        )
        if train_gen:
            model.fit(train_gen, steps_per_epoch=len(x_train)//batch_sz, **fit_kwargs)
        else:
            model.fit(x_train, y_train, **fit_kwargs)

        prog_bar.progress(100, text="Training complete ✓")
        st.session_state["cnn_model"]   = model
        st.session_state["cnn_history"] = history
        st.session_state["cnn_data"]    = (x_train, y_train, x_test, y_test)

        time.sleep(.4); prog_bar.empty()

        # ── METRICS ───────────────────────────────────────────────
        y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
        acc    = accuracy_score(y_test, y_pred)
        best_val = max(history["val_accuracy"])

        mc = st.columns(4)
        for col, (val, lbl) in zip(mc, [
            (f"{acc:.4f}",     "Test Accuracy"),
            (f"{best_val:.4f}","Best Val Acc"),
            (f"{min(history['val_loss']):.4f}","Best Val Loss"),
            (f"{len(history['loss'])}","Epochs Trained"),
        ]):
            col.markdown(f"""
            <div class="card" style="text-align:center">
              <div class="metric-val">{val}</div>
              <div class="metric-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

        # ── LEARNING CURVES ──────────────────────────────────────
        st.markdown('<div class="section-hdr">LEARNING CURVES</div>', unsafe_allow_html=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
        ep = range(1, len(history["loss"])+1)
        ax1.plot(ep, history["loss"],     color=NEON,    lw=1.8, label="Train")
        ax1.plot(ep, history["val_loss"], color=ACCENT3, lw=1.8, label="Val", ls="--")
        ax1.set_title("Loss"); ax1.legend(fontsize=7)
        ax1.yaxis.grid(True, alpha=.4)
        ax2.plot(ep, history["accuracy"],     color=NEON,    lw=1.8, label="Train")
        ax2.plot(ep, history["val_accuracy"], color=ACCENT3, lw=1.8, label="Val", ls="--")
        ax2.set_title("Accuracy"); ax2.legend(fontsize=7)
        ax2.yaxis.grid(True, alpha=.4)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # ── CONFUSION MATRIX ─────────────────────────────────────
        st.markdown('<div class="section-hdr">CONFUSION MATRIX</div>', unsafe_allow_html=True)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="YlOrRd",
                    xticklabels=CLASSES, yticklabels=CLASSES,
                    linecolor="#1a2540", linewidths=.3,
                    annot_kws={"size":7})
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix", color=NEON)
        plt.xticks(rotation=35, ha="right", fontsize=7)
        plt.yticks(rotation=0, fontsize=7)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # ── PER-CLASS REPORT ─────────────────────────────────────
        st.markdown('<div class="section-hdr">PER-CLASS REPORT</div>', unsafe_allow_html=True)
        report = classification_report(y_test, y_pred,
                    target_names=CLASSES, output_dict=True)
        import pandas as pd
        df_rep = pd.DataFrame(report).T.drop(["accuracy","macro avg","weighted avg"], errors="ignore")
        df_rep = df_rep[["precision","recall","f1-score","support"]].round(3)
        st.dataframe(df_rep.style.set_properties(**{
            "background-color":"#090e1a","color":"#c9d1e3",
            "border":"1px solid #1a2540","font-family":"monospace","font-size":".8rem"
        }).highlight_max(subset=["precision","recall","f1-score"],color="#0a2a1a"),
            use_container_width=True)

# ── TAB 2 : TEST IMAGE ───────────────────────────────────────────────────────
with tabs[2]:
    st.markdown('<div class="section-hdr">UPLOAD & CLASSIFY</div>', unsafe_allow_html=True)

    if "cnn_model" not in st.session_state:
        st.warning("Train the CNN first (Tab: Train & Results).")
    else:
        model = st.session_state["cnn_model"]
        uploaded = st.file_uploader("Upload any image (resized to 32×32)", type=["jpg","jpeg","png"])

        c1, c2 = st.columns(2)
        with c1:
            use_random = st.button("🎲 Use Random Test Image")

        img_arr = None
        if uploaded:
            pil_img = Image.open(uploaded).convert("RGB").resize((32,32))
            img_arr = np.array(pil_img, dtype="float32") / 255.
        elif use_random:
            _, _, x_test, y_test = st.session_state["cnn_data"]
            idx = np.random.randint(len(x_test))
            img_arr = x_test[idx]
            st.markdown(f"True label: **{CLASS_EMOJI[y_test[idx]]} {CLASSES[y_test[idx]]}**")

        if img_arr is not None:
            probs = model.predict(img_arr[None], verbose=0)[0]
            pred  = np.argmax(probs)

            r1, r2 = st.columns([1, 2])
            with r1:
                fig, ax = plt.subplots(figsize=(2.8, 2.8))
                ax.imshow(img_arr)
                ax.set_title(f"{CLASS_EMOJI[pred]} {CLASSES[pred]}\n{probs[pred]:.1%}",
                             color=NEON, fontsize=10)
                ax.axis("off"); fig.tight_layout()
                st.pyplot(fig, use_container_width=True); plt.close()
            with r2:
                fig, ax = plt.subplots(figsize=(5, 3))
                colors = [NEON if i==pred else "#1a3050" for i in range(10)]
                bars = ax.barh(CLASSES, probs, color=colors, edgecolor="#0a1628", linewidth=.5)
                ax.set_xlim(0,1); ax.set_title("Class Probabilities", color=NEON)
                ax.xaxis.grid(True, alpha=.3)
                for bar, p in zip(bars, probs):
                    ax.text(min(p+.01,.95), bar.get_y()+bar.get_height()/2,
                            f"{p:.1%}", va="center", fontsize=7, color="#c9d1e3")
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True); plt.close()

            # Activation maps
            st.markdown('<div class="section-hdr">ACTIVATION MAPS</div>', unsafe_allow_html=True)
            try:
                fig = get_activation_maps(model, img_arr)
                st.pyplot(fig, use_container_width=True); plt.close()
            except Exception as e:
                st.info(f"Activation map unavailable: {e}")

# ── TAB 3 : ANALYSIS ─────────────────────────────────────────────────────────
with tabs[3]:
    if "cnn_model" not in st.session_state:
        st.warning("Train the CNN first.")
    else:
        model = st.session_state["cnn_model"]
        x_train, y_train, x_test, y_test = st.session_state["cnn_data"]

        st.markdown('<div class="section-hdr">LEARNED FILTERS</div>', unsafe_allow_html=True)
        try:
            fig = get_filter_mosaic(model)
            if fig: st.pyplot(fig, use_container_width=True); plt.close()
        except Exception as e:
            st.info(f"Filter visualisation: {e}")

        st.markdown('<div class="section-hdr">PREDICTION CONFIDENCE DISTRIBUTION</div>',
                    unsafe_allow_html=True)
        y_prob = model.predict(x_test, verbose=0)
        conf   = y_prob.max(axis=1)
        pred_c = np.argmax(y_prob, axis=1)
        correct= (pred_c == y_test)

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        axes[0].hist(conf[correct],  bins=20, color=NEON,    alpha=.7, label="Correct", density=True)
        axes[0].hist(conf[~correct], bins=20, color="#f43f5e",alpha=.7, label="Wrong",   density=True)
        axes[0].set_title("Confidence — Correct vs Wrong")
        axes[0].legend(fontsize=7); axes[0].xaxis.grid(True, alpha=.3)

        per_cls_acc = [accuracy_score(y_test[y_test==c], pred_c[y_test==c])
                       for c in range(10)]
        axes[1].barh(CLASSES, per_cls_acc,
                     color=[NEON if v==max(per_cls_acc) else "#1a3050" for v in per_cls_acc],
                     edgecolor="#0a1628", linewidth=.5)
        axes[1].set_title("Per-Class Accuracy"); axes[1].set_xlim(0,1)
        axes[1].xaxis.grid(True, alpha=.3)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

        # Model summary
        st.markdown('<div class="section-hdr">MODEL SUMMARY</div>', unsafe_allow_html=True)
        summary_lines = []
        model.summary(print_fn=lambda x: summary_lines.append(x))
        st.code("\n".join(summary_lines), language="text")

# ── TAB 4 : CODE ─────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown('<div class="section-hdr">CNN MODEL CODE</div>', unsafe_allow_html=True)
    st.code("""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

# 1. LOAD & PREPROCESS DATA
(x_train, y_train),(x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.
x_test  = x_test.astype('float32')  / 255.
y_train, y_test = y_train.flatten(), y_test.flatten()

# 2. DATA AUGMENTATION
datagen = keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip=True, width_shift_range=0.1,
    height_shift_range=0.1, rotation_range=10
)
datagen.fit(x_train)

# 3. BUILD CNN
def build_cnn():
    inp = keras.Input(shape=(32, 32, 3))
    # Conv Block 1
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.2)(x)
    # Conv Block 2
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.2)(x)
    # Conv Block 3
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    # Classification Head
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(10, activation='softmax')(x)
    model = models.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = build_cnn()
model.summary()

# 4. TRAIN
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    steps_per_epoch=len(x_train)//64,
    validation_data=(x_test, y_test),
    epochs=20,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5)
    ]
)

# 5. EVALUATE
loss, acc = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {acc:.4f}')

# 6. PREDICT
import numpy as np
probs = model.predict(x_test[:1])
print('Predicted class:', np.argmax(probs))
""", language="python")