"""
RNN Assignment — IMDB Sentiment Analyser
=========================================
Dataset    : IMDB Movie Reviews (25 000 train / 25 000 test)
Model      : Embedding → LSTM → Bi-LSTM → Dense (configurable)
Frontend   : Streamlit with warm editorial aesthetic
Backend    : TensorFlow / Keras NLP pipeline
Testing    : Accuracy, AUC, confusion matrix, attention heatmap,
             word-importance visualisation, live review demo
Deployment : `streamlit run rnn_app.py`
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import re, time, warnings
warnings.filterwarnings("ignore")

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
                              classification_report, roc_curve)

tf.get_logger().setLevel("ERROR")

# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="RNN · Sentiment Analyser",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════════════════
# CSS  —  warm editorial / newsprint aesthetic
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=Inconsolata:wght@400;700&family=Source+Serif+4:wght@300;400;600&display=swap');

html,body,[class*="css"]{ font-family:'Source Serif 4',serif; }

.stApp{
  background:#faf7f2;
  color:#2c2416;
}
[data-testid="stSidebar"]{
  background:#f2ede4;
  border-right:2px solid #d4c5a9;
}

/* HERO */
.hero{
  background:linear-gradient(135deg,#2c1810 0%,#3d2614 100%);
  border-radius:16px;padding:2.6rem 3rem;margin-bottom:1.8rem;
  position:relative;overflow:hidden;
}
.hero::before{
  content:'';position:absolute;bottom:-40px;right:-40px;
  width:280px;height:280px;
  background:radial-gradient(circle,rgba(212,165,100,.15) 0%,transparent 65%);
  border-radius:50%;
}
.hero-title{
  font-family:'Playfair Display',serif;font-weight:900;font-size:2.6rem;
  color:#f5deb3;letter-spacing:.02em;margin:0;
  text-shadow:0 2px 20px rgba(0,0,0,.4);
}
.hero-sub{
  font-family:'Inconsolata',monospace;color:#a08060;font-size:.9rem;
  margin-top:.5rem;
}
.badge{
  display:inline-block;background:rgba(245,222,179,.12);
  border:1px solid rgba(245,222,179,.3);border-radius:4px;
  padding:.12rem .5rem;font-size:.7rem;color:#d4a060;
  font-family:'Inconsolata',monospace;margin:.15rem .1rem;
}

/* CARDS */
.card{
  background:#fff;border:1px solid #e0d5c5;border-radius:12px;
  padding:1.2rem 1.4rem;margin-bottom:.9rem;
  box-shadow:0 1px 4px rgba(0,0,0,.06);
}
.card-warm{ border-left:4px solid #c47a30; }
.metric-val{
  font-family:'Playfair Display',serif;font-weight:700;
  font-size:2.2rem;color:#c47a30;
}
.metric-lbl{
  font-size:.68rem;color:#8a7560;text-transform:uppercase;
  letter-spacing:.1em;margin-top:.15rem;font-family:'Inconsolata',monospace;
}
.section-hdr{
  font-family:'Playfair Display',serif;font-weight:700;font-size:1.15rem;
  color:#3d2614;border-bottom:2px solid #c47a30;padding-bottom:.3rem;
  margin:1.4rem 0 .8rem;
}
.info-box{
  background:#fdf8f0;border:1px solid #e0d5c5;border-radius:10px;
  padding:1rem 1.2rem;color:#5a4a35;font-size:.85rem;line-height:1.75;
}
.pos-badge{
  display:inline-block;background:#d4edda;border:1px solid #28a745;
  border-radius:20px;padding:.25rem 1rem;color:#155724;font-weight:700;
  font-size:1rem;font-family:'Playfair Display',serif;
}
.neg-badge{
  display:inline-block;background:#f8d7da;border:1px solid #dc3545;
  border-radius:20px;padding:.25rem 1rem;color:#721c24;font-weight:700;
  font-size:1rem;font-family:'Playfair Display',serif;
}

/* WORD HEAT */
.word-heat span{
  display:inline-block;padding:.15rem .3rem;margin:.1rem .05rem;
  border-radius:3px;font-family:'Inconsolata',monospace;font-size:.9rem;
  transition:.15s;
}

/* Buttons */
.stButton>button{
  background:linear-gradient(135deg,#c47a30,#a05820);
  color:#faf7f2;border:none;border-radius:8px;
  font-family:'Playfair Display',serif;font-weight:700;
  font-size:.85rem;padding:.55rem 1.2rem;width:100%;
  box-shadow:0 2px 10px rgba(196,122,48,.25);transition:.2s;
}
.stButton>button:hover{box-shadow:0 4px 18px rgba(196,122,48,.4);}
label{color:#5a4a35!important;}
.stTabs [data-baseweb="tab-list"]{
  background:#f2ede4;border-radius:8px;gap:2px;padding:3px;
}
.stTabs [data-baseweb="tab"]{
  color:#8a7560;font-family:'Source Serif 4',serif;font-size:.85rem;
}
.stTabs [aria-selected="true"]{
  background:#fff!important;color:#c47a30!important;font-weight:600!important;
}
textarea{font-family:'Inconsolata',monospace!important;font-size:.9rem!important;}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# MATPLOTLIB WARM THEME
# ════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "figure.facecolor":"#faf7f2","axes.facecolor":"#fff",
    "axes.edgecolor":"#d4c5a9","axes.labelcolor":"#5a4a35",
    "xtick.color":"#8a7560","ytick.color":"#8a7560",
    "text.color":"#2c2416","grid.color":"#ece5d8","grid.linewidth":.6,
    "axes.titlecolor":"#3d2614","axes.titlesize":10,"axes.labelsize":8,
    "xtick.labelsize":7,"ytick.labelsize":7,"figure.dpi":130,
    "font.family":"serif",
})
WARM="#c47a30"; COOL="#3d7ab8"; GREEN="#28a745"; RED="#dc3545"

# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════
VOCAB_SIZE   = 10000
MAX_LEN      = 256

# ════════════════════════════════════════════════════════════════════════════
# DATA
# ════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_imdb_data(n_train=10000, n_test=2000):
    (x_tr, y_tr),(x_te, y_te) = imdb.load_data(num_words=VOCAB_SIZE)
    x_tr = pad_sequences(x_tr[:n_train], maxlen=MAX_LEN, truncating="post", padding="post")
    x_te = pad_sequences(x_te[:n_test],  maxlen=MAX_LEN, truncating="post", padding="post")
    return x_tr, y_tr[:n_train], x_te, y_te[:n_test]

@st.cache_data(show_spinner=False)
def get_word_index():
    wi = imdb.get_word_index()
    return {v+3: k for k, v in wi.items()}  # index→word

# ════════════════════════════════════════════════════════════════════════════
# MODEL BUILDER
# ════════════════════════════════════════════════════════════════════════════
def build_rnn(arch="BiLSTM", embed_dim=64, lstm_units=64,
              dropout=0.3, vocab=VOCAB_SIZE, maxlen=MAX_LEN):
    inp = keras.Input(shape=(maxlen,))
    x   = layers.Embedding(vocab, embed_dim, mask_zero=True)(inp)
    x   = layers.SpatialDropout1D(dropout/2)(x)

    if arch == "SimpleRNN":
        x = layers.SimpleRNN(lstm_units, dropout=dropout)(x)
    elif arch == "LSTM":
        x = layers.LSTM(lstm_units, dropout=dropout, recurrent_dropout=.1)(x)
    elif arch == "BiLSTM":
        x = layers.Bidirectional(
            layers.LSTM(lstm_units, dropout=dropout, recurrent_dropout=.1))(x)
    elif arch == "Stacked LSTM":
        x = layers.LSTM(lstm_units, return_sequences=True,
                        dropout=dropout, recurrent_dropout=.1)(x)
        x = layers.LSTM(lstm_units//2, dropout=dropout)(x)
    elif arch == "GRU":
        x = layers.GRU(lstm_units, dropout=dropout, recurrent_dropout=.1)(x)
    elif arch == "BiGRU":
        x = layers.Bidirectional(
            layers.GRU(lstm_units, dropout=dropout, recurrent_dropout=.1))(x)

    x   = layers.Dense(32, activation="relu")(x)
    x   = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model

# ════════════════════════════════════════════════════════════════════════════
# LIVE TEXT INFERENCE HELPERS
# ════════════════════════════════════════════════════════════════════════════
def preprocess_text(text, word_index):
    """Tokenise user text → padded int array."""
    tokens = re.findall(r"[a-z']+", text.lower())
    seq    = [word_index.get(t, 2) + 3 for t in tokens]  # 2=<UNK>
    seq    = [min(v, VOCAB_SIZE-1) for v in seq]
    padded = pad_sequences([seq], maxlen=MAX_LEN,
                           truncating="post", padding="post")
    return tokens, padded

def word_importance(model, tokens, padded_seq, n_top=20):
    """Approximate word importance via input gradient magnitude."""
    emb_layer = [l for l in model.layers if isinstance(l, layers.Embedding)][0]
    sub = keras.Model(model.input,
                      [model.output, emb_layer.output])
    with tf.GradientTape() as tape:
        x_t = tf.constant(padded_seq, dtype=tf.int32)
        emb_out = emb_layer(x_t)
        tape.watch(emb_out)
        # re-run forward pass from embedding onward
        pred = model(x_t, training=False)
    grads      = tape.gradient(pred, emb_out)
    importance = tf.reduce_sum(tf.abs(grads), axis=-1).numpy()[0]
    # Align with tokens (first len(tokens) positions)
    tok_imp = importance[:len(tokens)]
    if tok_imp.max() > 0:
        tok_imp = tok_imp / tok_imp.max()
    return tokens[:n_top], tok_imp[:n_top]

def color_word(word, score, positive):
    alpha = 0.15 + 0.75 * score
    r, g, b = (0,.8,.3) if positive else (.9,.15,.15)
    bg = f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{alpha:.2f})"
    return f'<span style="background:{bg};padding:2px 5px;border-radius:3px;margin:2px">{word}</span>'

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🎬 RNN Control Panel")
    st.markdown("---")

    st.markdown("**Dataset**")
    n_train = st.select_slider("Training samples",
        [2000,5000,10000,20000,25000], value=10000)
    n_test  = st.select_slider("Test samples",
        [500,1000,2000,5000,10000], value=2000)

    st.markdown("**Architecture**")
    arch       = st.selectbox("RNN Type",
        ["BiLSTM","LSTM","Stacked LSTM","GRU","BiGRU","SimpleRNN"])
    embed_dim  = st.select_slider("Embedding Dim", [32,64,128], value=64)
    lstm_units = st.select_slider("RNN Units",     [32,64,128], value=64)
    dropout    = st.slider("Dropout", 0.1, 0.6, 0.3, 0.05)

    st.markdown("**Training**")
    epochs   = st.slider("Max Epochs", 3, 30, 10)
    batch_sz = st.selectbox("Batch Size", [32,64,128,256], index=1)
    early_st = st.checkbox("Early Stopping", value=True)

    st.markdown("---")
    train_btn = st.button("📖  Train RNN")

# ════════════════════════════════════════════════════════════════════════════
# HERO
# ════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
  <div class="hero-title">🎬 RNN · Sentiment Analyser</div>
  <div class="hero-sub">Recurrent Neural Networks | IMDB Movie Reviews | Binary Classification</div>
  <div style="margin-top:.8rem">
    <span class="badge">IMDB 25K Reviews</span>
    <span class="badge">{arch}</span>
    <span class="badge">Embedding → {arch} → Dense</span>
    <span class="badge">TensorFlow / Keras</span>
    <span class="badge">Streamlit</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════
tabs = st.tabs(["📐 Architecture","🏋️ Train & Results","🎭 Live Analysis","📊 Analysis","💻 Code"])

# ── TAB 0 : ARCHITECTURE ─────────────────────────────────────────────────────
with tabs[0]:
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown('<div class="section-hdr">RNN Architecture</div>', unsafe_allow_html=True)
        steps = [
            ("① Embedding Layer",
             f"Vocabulary: {VOCAB_SIZE:,} tokens  →  {embed_dim}-dim dense vectors.<br>"
             "Maps sparse integer IDs to learned semantic representations."),
            (f"② {arch} Layer",
             f"{lstm_units} units · Dropout {dropout} · Recurrent Dropout 0.1.<br>"
             "Processes the sequence, capturing long-range dependencies."),
            ("③ Dense Head",
             "Dense(32, ReLU) → Dropout → Dense(1, Sigmoid).<br>"
             "Outputs probability of positive sentiment in [0, 1]."),
            ("④ Training Setup",
             "Loss: Binary Cross-Entropy · Optimiser: Adam(lr=1e-3)<br>"
             "Metric: Accuracy + ROC-AUC"),
        ]
        for title, body in steps:
            st.markdown(f"""
            <div class="card card-warm">
              <div style="font-family:'Playfair Display',serif;font-weight:700;
                          color:#c47a30;margin-bottom:.35rem">{title}</div>
              <div style="font-size:.84rem;color:#5a4a35;line-height:1.7">{body}</div>
            </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-hdr">Why RNNs for Text?</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
          <b style="color:#c47a30">Sequential Memory</b><br>
          RNNs process tokens one by one, maintaining a hidden state that
          captures context from all previous tokens — critical for understanding
          negation ("not bad"), contrast ("started well but…"), and nuance.<br><br>
          <b style="color:#c47a30">LSTM / GRU Gates</b><br>
          Long Short-Term Memory gates (input, forget, output) solve the
          vanishing gradient problem, enabling the model to remember relevant
          information across hundreds of tokens.<br><br>
          <b style="color:#c47a30">Bidirectional</b><br>
          BiLSTM runs two passes — forward & backward — so each position
          receives context from both directions, significantly boosting
          sentiment understanding.
        </div>""", unsafe_allow_html=True)

        # Architecture mini-diagram
        fig, ax = plt.subplots(figsize=(5, 3.6))
        ax.set_xlim(0,10); ax.set_ylim(0,7); ax.axis("off")
        import matplotlib.patches as mpatches
        boxes = [
            (3,5.8,4,0.85,"Input Tokens","#fdf8f0","#c47a30"),
            (3,4.5,4,0.85,"Embedding Layer","#fdf8f0","#c47a30"),
            (3,3.2,4,0.85,f"{arch}","#fff8f2","#a05820"),
            (3,1.9,4,0.85,"Dense(32) → Dropout","#fdf8f0","#c47a30"),
            (3,0.6,4,0.85,"Sigmoid Output","#f8f0e0","#2c7a30"),
        ]
        for bx,by,bw,bh,lbl,fc,ec in boxes:
            ax.add_patch(mpatches.FancyBboxPatch(
                (bx,by),bw,bh,boxstyle="round,pad=0.08",fc=fc,ec=ec,lw=1.5))
            ax.text(bx+bw/2, by+bh/2, lbl, ha="center", va="center",
                    fontsize=8.5, color="#2c2416", fontweight="bold")
        for (bx,by,bw,bh,*_),(nx,ny,nw,nh,*__) in zip(boxes[:-1],boxes[1:]):
            ax.annotate("",xy=(nx+nw/2,ny+nh),xytext=(bx+bw/2,by),
                arrowprops=dict(arrowstyle="-|>",color="#c47a30",lw=1.5))
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

# ── TAB 1 : TRAIN ─────────────────────────────────────────────────────────────
with tabs[1]:
    if not train_btn:
        st.markdown("""
        <div class="info-box" style="text-align:center;padding:2rem">
          Select your architecture in the sidebar and click
          <b style="color:#c47a30">📖 Train RNN</b>
        </div>""", unsafe_allow_html=True)
    else:
        prog = st.progress(0, text="Loading IMDB dataset …")
        x_train, y_train, x_test, y_test = load_imdb_data(n_train, n_test)
        word_idx = get_word_index()
        prog.progress(12, text="Building model …")

        model = build_rnn(arch, embed_dim, lstm_units, dropout)

        cbs = []
        if early_st:
            cbs += [keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
                    keras.callbacks.ReduceLROnPlateau(patience=2, factor=.5, verbose=0)]

        history = {"loss":[],"val_loss":[],"accuracy":[],"val_accuracy":[]}

        class LiveCB(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                for k in history: history[k].append(logs.get(k,0))
                frac = 12 + int(85*(epoch+1)/epochs)
                prog.progress(min(frac,97),
                    text=f"Epoch {epoch+1}/{epochs} — "
                         f"val_acc {logs.get('val_accuracy',0):.4f} · "
                         f"loss {logs.get('loss',0):.4f}")

        model.fit(x_train, y_train, validation_data=(x_test, y_test),
                  epochs=epochs, batch_size=batch_sz,
                  callbacks=cbs+[LiveCB()], verbose=0)

        prog.progress(100, text="Training complete ✓")
        st.session_state["rnn_model"]   = model
        st.session_state["rnn_history"] = history
        st.session_state["rnn_data"]    = (x_train, y_train, x_test, y_test)
        st.session_state["word_idx"]    = word_idx
        time.sleep(.3); prog.empty()

        # ── METRICS ──────────────────────────────────────────────
        y_prob = model.predict(x_test, verbose=0).flatten()
        y_pred = (y_prob >= 0.5).astype(int)
        acc    = accuracy_score(y_test, y_pred)
        auc    = roc_auc_score(y_test, y_prob)
        best_v = max(history["val_accuracy"])

        mc = st.columns(4)
        for col,(val,lbl) in zip(mc,[
            (f"{acc:.4f}",  "Test Accuracy"),
            (f"{auc:.4f}",  "ROC-AUC"),
            (f"{best_v:.4f}","Best Val Acc"),
            (f"{len(history['loss'])}","Epochs"),
        ]):
            col.markdown(f"""
            <div class="card" style="text-align:center">
              <div class="metric-val">{val}</div>
              <div class="metric-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

        # ── LEARNING CURVES ──────────────────────────────────────
        st.markdown('<div class="section-hdr">Learning Curves</div>', unsafe_allow_html=True)
        fig, (a1,a2) = plt.subplots(1,2,figsize=(10,3.6))
        ep = range(1, len(history["loss"])+1)
        a1.plot(ep, history["loss"],     color=WARM, lw=1.8, label="Train")
        a1.plot(ep, history["val_loss"], color=COOL, lw=1.8, label="Val", ls="--")
        a1.set_title("Binary Cross-Entropy Loss"); a1.legend(fontsize=7)
        a1.yaxis.grid(True, alpha=.4)
        a2.plot(ep, history["accuracy"],     color=WARM, lw=1.8, label="Train")
        a2.plot(ep, history["val_accuracy"], color=COOL, lw=1.8, label="Val", ls="--")
        a2.set_title("Accuracy"); a2.legend(fontsize=7)
        a2.yaxis.grid(True, alpha=.4)
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

        # ── ROC + CONFUSION ──────────────────────────────────────
        st.markdown('<div class="section-hdr">ROC Curve & Confusion Matrix</div>',
                    unsafe_allow_html=True)
        fig, (a1,a2) = plt.subplots(1,2,figsize=(10,3.8))
        fpr,tpr,_ = roc_curve(y_test, y_prob)
        a1.plot(fpr,tpr,color=WARM,lw=2,label=f"AUC={auc:.4f}")
        a1.plot([0,1],[0,1],":",color="#8a7560",lw=1)
        a1.fill_between(fpr,tpr,alpha=.08,color=WARM)
        a1.set_xlabel("False Positive Rate"); a1.set_ylabel("True Positive Rate")
        a1.set_title("ROC Curve"); a1.legend(fontsize=7)

        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", ax=a2,
                    cmap="YlOrBr", xticklabels=["Neg","Pos"],
                    yticklabels=["Neg","Pos"],
                    linewidths=.5, linecolor="#e0d5c5",
                    annot_kws={"size":11,"color":"#2c2416"})
        a2.set_title("Confusion Matrix")
        a2.set_xlabel("Predicted"); a2.set_ylabel("Actual")
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

        # ── CLASSIFICATION REPORT ────────────────────────────────
        st.markdown('<div class="section-hdr">Classification Report</div>',
                    unsafe_allow_html=True)
        import pandas as pd
        rep = classification_report(y_test, y_pred,
              target_names=["Negative","Positive"], output_dict=True)
        df_rep = pd.DataFrame(rep).T.round(4)
        st.dataframe(df_rep.style.set_properties(**{
            "background-color":"#fff","color":"#2c2416",
            "border":"1px solid #e0d5c5","font-family":"monospace","font-size":".8rem"
        }), use_container_width=True)

# ── TAB 2 : LIVE ANALYSIS ────────────────────────────────────────────────────
with tabs[2]:
    st.markdown('<div class="section-hdr">Analyse Your Own Review</div>',
                unsafe_allow_html=True)

    if "rnn_model" not in st.session_state:
        st.warning("Train the RNN first (Tab: Train & Results).")
    else:
        model    = st.session_state["rnn_model"]
        word_idx = st.session_state["word_idx"]

        EXAMPLES = {
            "Glowing review 🌟":
                "This movie was absolutely fantastic! The acting was superb, "
                "the plot was gripping, and the cinematography breathtaking. "
                "One of the best films I have seen in years.",
            "Negative review 👎":
                "Terrible waste of time. The story was predictable and boring, "
                "the characters were flat and uninteresting. I nearly walked out "
                "halfway through. Completely disappointing.",
            "Mixed review 🤔":
                "The film started really well with great visuals, but the second "
                "half was confusing and the ending felt rushed. Not bad overall.",
        }
        example_choice = st.selectbox("Load example", ["(Type your own…)"] + list(EXAMPLES))
        default_text = EXAMPLES.get(example_choice, "")
        review_text  = st.text_area("Movie review text", value=default_text,
                                     height=130, placeholder="Write or paste a review here …")

        if st.button("🔍  Analyse Sentiment") and review_text.strip():
            tokens, padded = preprocess_text(review_text, word_idx)
            prob = float(model.predict(padded, verbose=0)[0][0])
            sentiment = "Positive" if prob >= 0.5 else "Negative"
            conf = prob if prob >= 0.5 else 1 - prob

            r1, r2, r3 = st.columns(3)
            badge = f'<span class="{"pos" if sentiment=="Positive" else "neg"}-badge">'\
                    f'{"😊" if sentiment=="Positive" else "😞"} {sentiment}</span>'
            r1.markdown(f"<div style='text-align:center;margin-top:.5rem'>{badge}</div>",
                        unsafe_allow_html=True)
            r2.markdown(f"""<div class="card" style="text-align:center">
                <div class="metric-val">{prob:.1%}</div>
                <div class="metric-lbl">Positive Probability</div></div>""",
                unsafe_allow_html=True)
            r3.markdown(f"""<div class="card" style="text-align:center">
                <div class="metric-val">{conf:.1%}</div>
                <div class="metric-lbl">Confidence</div></div>""",
                unsafe_allow_html=True)

            # Probability bar
            fig, ax = plt.subplots(figsize=(7, .9))
            ax.barh([0],[prob], color=GREEN, height=.6)
            ax.barh([0],[1-prob], left=[prob], color=RED, height=.6)
            ax.set_xlim(0,1); ax.axis("off")
            ax.text(prob/2,   0, f"Positive {prob:.0%}",   ha="center",va="center",
                    color="white", fontsize=9, fontweight="bold")
            ax.text(prob+(1-prob)/2, 0, f"Negative {1-prob:.0%}", ha="center",va="center",
                    color="white", fontsize=9, fontweight="bold")
            fig.tight_layout(pad=0); st.pyplot(fig, use_container_width=True); plt.close()

            # Word importance
            st.markdown('<div class="section-hdr">Word Importance Heatmap</div>',
                        unsafe_allow_html=True)
            try:
                toks, scores = word_importance(model, tokens, padded)
                html_words = " ".join(
                    color_word(w, float(s), sentiment=="Positive")
                    for w, s in zip(toks, scores))
                st.markdown(f'<div class="word-heat">{html_words}</div>',
                            unsafe_allow_html=True)
                st.caption("Colour intensity ∝ gradient magnitude (approximate importance)")
            except Exception as e:
                st.info(f"Word importance unavailable: {e}")

# ── TAB 3 : ANALYSIS ─────────────────────────────────────────────────────────
with tabs[3]:
    if "rnn_model" not in st.session_state:
        st.warning("Train the RNN first.")
    else:
        model = st.session_state["rnn_model"]
        x_train, y_train, x_test, y_test = st.session_state["rnn_data"]
        y_prob = model.predict(x_test, verbose=0).flatten()
        y_pred = (y_prob >= 0.5).astype(int)

        st.markdown('<div class="section-hdr">Confidence Distribution</div>',
                    unsafe_allow_html=True)
        correct = (y_pred == y_test)
        fig, axes = plt.subplots(1, 2, figsize=(10,3.6))
        axes[0].hist(y_prob[y_test==1], bins=25, color=GREEN, alpha=.7,
                     label="Positive reviews", density=True)
        axes[0].hist(y_prob[y_test==0], bins=25, color=RED,   alpha=.7,
                     label="Negative reviews", density=True)
        axes[0].axvline(.5, color=WARM, lw=1.5, ls="--", label="Decision boundary")
        axes[0].set_title("Prediction Probability Distribution")
        axes[0].legend(fontsize=7); axes[0].xaxis.grid(True,alpha=.4)

        axes[1].hist(y_prob[correct],  bins=25, color=COOL, alpha=.7,
                     label="Correctly classified", density=True)
        axes[1].hist(y_prob[~correct], bins=25, color="#f59e0b", alpha=.7,
                     label="Misclassified", density=True)
        axes[1].set_title("Confidence — Correct vs Misclassified")
        axes[1].legend(fontsize=7); axes[1].xaxis.grid(True,alpha=.4)
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

        # Threshold sweep
        st.markdown('<div class="section-hdr">Threshold vs Accuracy</div>',
                    unsafe_allow_html=True)
        thresholds = np.linspace(0.1, 0.9, 50)
        accs = [accuracy_score(y_test,(y_prob>=t).astype(int)) for t in thresholds]
        fig, ax = plt.subplots(figsize=(8,3))
        ax.plot(thresholds, accs, color=WARM, lw=2)
        ax.axvline(.5, color=COOL, lw=1.5, ls="--", label="Default threshold=0.5")
        ax.fill_between(thresholds, accs, alpha=.12, color=WARM)
        ax.set_xlabel("Decision Threshold"); ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy at Different Classification Thresholds")
        ax.legend(fontsize=7); ax.yaxis.grid(True,alpha=.4)
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

        # Model summary
        st.markdown('<div class="section-hdr">Model Summary</div>', unsafe_allow_html=True)
        lines = []
        model.summary(print_fn=lambda x: lines.append(x))
        st.code("\n".join(lines), language="text")

# ── TAB 4 : CODE ─────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown('<div class="section-hdr">RNN Model Code</div>', unsafe_allow_html=True)
    st.code("""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. CONSTANTS
VOCAB_SIZE = 10000
MAX_LEN    = 256

# 2. LOAD & PREPROCESS
(x_train, y_train),(x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)
x_train = pad_sequences(x_train, maxlen=MAX_LEN, padding='post', truncating='post')
x_test  = pad_sequences(x_test,  maxlen=MAX_LEN, padding='post', truncating='post')

# 3. BUILD RNN (Bidirectional LSTM)
def build_bilstm(vocab=VOCAB_SIZE, maxlen=MAX_LEN,
                 embed_dim=64, lstm_units=64, dropout=0.3):
    inp = keras.Input(shape=(maxlen,))
    x   = layers.Embedding(vocab, embed_dim, mask_zero=True)(inp)
    x   = layers.SpatialDropout1D(dropout/2)(x)
    x   = layers.Bidirectional(
            layers.LSTM(lstm_units, dropout=dropout,
                        recurrent_dropout=0.1))(x)
    x   = layers.Dense(32, activation='relu')(x)
    x   = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

model = build_bilstm()
model.summary()

# 4. TRAIN
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=10, batch_size=64,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5)
    ]
)

# 5. EVALUATE
from sklearn.metrics import accuracy_score, roc_auc_score
y_prob = model.predict(x_test).flatten()
y_pred = (y_prob >= 0.5).astype(int)
print(f'Test Accuracy : {accuracy_score(y_test, y_pred):.4f}')
print(f'ROC-AUC       : {roc_auc_score(y_test, y_prob):.4f}')

# 6. PREDICT ON NEW TEXT
from tensorflow.keras.preprocessing.sequence import pad_sequences
word_index = imdb.get_word_index()
rev_index  = {v+3: k for k, v in word_index.items()}

def predict_sentiment(text):
    import re
    tokens = re.findall(r"[a-z']+", text.lower())
    seq    = [min(word_index.get(t, 0) + 3, VOCAB_SIZE-1) for t in tokens]
    padded = pad_sequences([seq], maxlen=MAX_LEN, padding='post')
    prob   = model.predict(padded, verbose=0)[0][0]
    label  = 'POSITIVE' if prob >= 0.5 else 'NEGATIVE'
    return label, float(prob)

label, prob = predict_sentiment("This film was absolutely brilliant!")
print(f'{label}  ({prob:.2%} confidence)')
""", language="python")
