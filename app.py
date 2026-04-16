"""
app.py — Streamlit ECG Prediction App with full XAI
Run: streamlit run app.py
"""

import os
import tempfile
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import streamlit as st

from src.data.preprocessing import preprocess_signal
from src.models.patchtst import PatchTST
from src.training.trainer import Trainer
from src.data.dataset import get_dataloaders
from src.utils.config import cfg
from src.utils.label_utils import decode_scp_labels, decode_superclass_labels
from src.xai.attention_viz import attn_to_signal_importance
from src.xai.gradcam import GradCAM
from src.xai.integrated_grads import IntegratedGradients
from src.xai.occlusion import OcclusionSensitivity
from src.xai.shap_explainer import ECGShapExplainer, build_background, make_shap_radial_plot

import wfdb

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ECG Analyzer",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@400;500;600;700&display=swap');
:root {
    --bg:#050505; --surface:rgba(20, 25, 35, 0.6); --border:rgba(0, 229, 255, 0.2);
    --accent:#00e5ff; --accent-glow:0 0 10px rgba(0,229,255,0.5), 0 0 20px rgba(0,229,255,0.3);
    --accent2:#ff0055; --accent3:#a3e635;
    --text:#f0f0f0; --muted:#8a9bb0; --card:rgba(15, 20, 30, 0.8);
}
html,body,[class*="css"]{
    font-family:'Rajdhani',sans-serif;
    background-color:var(--bg);
    background-image: radial-gradient(circle at 50% 0%, rgba(0, 229, 255, 0.05) 0%, transparent 50%);
    color:var(--text);
}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:2rem;padding-bottom:2rem;}

/* Header styling */
.ecg-header{
    text-align:center;
    padding:3rem 0 2rem;
    margin-bottom:3rem;
    position: relative;
    overflow: hidden;
    border-bottom: 1px solid var(--border);
    background: linear-gradient(180deg, rgba(0,229,255,0.05) 0%, transparent 100%);
    border-radius: 0 0 30px 30px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
}
.ecg-header h1{
    font-family:'Orbitron',sans-serif;
    font-size:3rem;
    font-weight:700;
    color:var(--text);
    text-shadow: var(--accent-glow);
    letter-spacing: 2px;
    margin:0;
    text-transform: uppercase;
}
.ecg-header p{color:var(--accent);font-size:1.1rem;margin-top:0.8rem;letter-spacing: 1px; font-weight: 500;}
.pulse-dot{
    display:inline-block;width:12px;height:12px;
    background:var(--accent2);
    border-radius:50%;margin-right:12px;
    box-shadow: 0 0 10px var(--accent2);
    animation:heartbeat 1.2s infinite;
}
@keyframes heartbeat{
    0%{transform:scale(1);}
    15%{transform:scale(1.3);}
    30%{transform:scale(1);}
    45%{transform:scale(1.3);}
    60%, 100%{transform:scale(1);}
}

/* Glassmorphism Cards */
.metric-card, .xai-info{
    background:var(--card);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border:1px solid var(--border);
    border-radius:16px;
    padding:1.5rem;
    margin-bottom:1rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.metric-card:hover{
    transform: translateY(-5px);
    box-shadow: 0 12px 40px 0 rgba(0, 229, 255, 0.15);
    border-color: rgba(0, 229, 255, 0.5);
}
.metric-card h4{font-family:'Orbitron',sans-serif;font-size:0.8rem;text-transform:uppercase;letter-spacing:3px;color:var(--accent);margin:0 0 0.8rem;}

/* Cyber Badges */
.sc-badge{
    display:inline-block;padding:0.4rem 1.2rem;border-radius:4px;
    font-family:'Orbitron',sans-serif;font-size:0.8rem;font-weight:700;
    margin:0.3rem;letter-spacing:1px;
    text-transform: uppercase;
    clip-path: polygon(10px 0, 100% 0, 100% calc(100% - 10px), calc(100% - 10px) 100%, 0 100%, 0 10px);
}
.sc-NORM{background:rgba(6, 78, 59, 0.5);color:#6ee7b7;border-left: 3px solid #059669;}
.sc-MI{background:rgba(127, 29, 29, 0.5);color:#fca5a5;border-left: 3px solid #dc2626;}
.sc-STTC{background:rgba(30, 27, 75, 0.5);color:#a5b4fc;border-left: 3px solid #6366f1;}
.sc-CD{background:rgba(28, 25, 23, 0.5);color:#fcd34d;border-left: 3px solid #d97706;}
.sc-HYP{background:rgba(74, 25, 66, 0.5);color:#f0abfc;border-left: 3px solid #a21caf;}
.sc-FORM{background:rgba(12, 35, 64, 0.5);color:#7dd3fc;border-left: 3px solid #0284c7;}
.sc-RHYTHM{background:rgba(26, 46, 5, 0.5);color:#bef264;border-left: 3px solid #65a30d;}
.sc-NONE{background:var(--surface);color:var(--muted);border-left: 3px solid var(--border);}

/* Progress Bars */
.scp-bar-wrap{margin:0.5rem 0;display:flex;align-items:center;gap:1rem;}
.scp-name{font-family:'Orbitron',sans-serif;font-size:0.8rem;color:var(--text);width:100px;flex-shrink:0;}
.scp-track{flex:1;height:8px;background:rgba(255,255,255,0.05);border-radius:4px;overflow:hidden; box-shadow: inset 0 1px 3px rgba(0,0,0,0.5);}
.scp-fill{height:100%;border-radius:4px;transition:width 1s cubic-bezier(0.175, 0.885, 0.32, 1.275); position: relative; overflow: hidden;}
.scp-fill::after { content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.5), transparent); animation: shimmer 2s infinite; }
@keyframes shimmer { 0% { transform: translateX(-100%); } 100% { transform: translateX(100%); } }
.scp-pct{font-family:'Orbitron',sans-serif;font-size:0.85rem;color:var(--accent);width:50px;text-align:right;}

/* Titles and Tabs */
.section-title{
    font-family:'Orbitron',sans-serif;font-size:0.9rem;
    text-transform:uppercase;letter-spacing:4px;
    color:var(--accent);margin:2rem 0 1.2rem;
    padding-bottom:0.5rem;
    border-bottom: 2px solid transparent;
    border-image: linear-gradient(to right, var(--accent), transparent) 1;
}
.xai-info{
    border-left: 4px solid var(--accent);
    color:#e2e8f0; font-size: 1rem;
}
.stTabs [data-baseweb="tab-list"]{background:rgba(255,255,255,0.03);border-radius:12px;gap:8px;padding:8px; border: 1px solid rgba(255,255,255,0.05);}
.stTabs [data-baseweb="tab"]{border-radius:8px;color:var(--muted);font-family:'Orbitron',sans-serif;font-size:0.85rem; padding: 10px 20px; transition: all 0.3s ease;}
.stTabs [aria-selected="true"]{background:rgba(0, 229, 255, 0.15) !important;color:var(--accent) !important; box-shadow: inset 0 0 10px rgba(0, 229, 255, 0.2); border: 1px solid rgba(0, 229, 255, 0.3);}

/* File Uploader styling */
[data-testid="stFileUploader"] {
    background: var(--card) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 16px !important;
    padding: 2rem !important;
    text-align: center !important;
    transition: all 0.3s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
    box-shadow: 0 0 20px rgba(0, 229, 255, 0.2) !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="ecg-header">
    <h1><span class="pulse-dot"></span>NEURO·ECG ANALYZER</h1>
    <p>QUANTUM PATCHTST · XAI DIAGNOSTICS · MULTI-VECTOR ANALYSIS</p>
</div>
""", unsafe_allow_html=True)

LEAD_NAMES = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]


@st.cache_resource
def load_model():
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, all_codes = get_dataloaders(batch_size=32)
    model   = PatchTST()
    trainer = Trainer(model, loaders, device=device)
    trainer.load_best()
    model.to(device)
    model.eval()
    bg = build_background(loaders["train"], n_samples=50)
    return model, all_codes, device, loaders, bg


def make_ecg_fig(signal_np, importance, title, sig_color, cmap_name, cbar_label):
    fig, axes = plt.subplots(12, 1, figsize=(16, 18), sharex=True)
    fig.patch.set_facecolor("#050505")
    fig.suptitle(title, color="#f0f0f0", fontfamily='Orbitron', fontsize=12, fontweight="bold", y=0.999)
    cmap = cm.get_cmap(cmap_name)
    t = np.arange(signal_np.shape[1])
    for i, ax in enumerate(axes):
        ax.set_facecolor("#050505")
        # Add a subtle grid
        ax.grid(color="#111827", linestyle='-', linewidth=0.5, alpha=0.3)
        for j in range(len(t) - 1):
            ax.axvspan(t[j], t[j+1], color=cmap(float(importance[i][j])),
                       alpha=0.5, linewidth=0)
        # Plot signal with a subtle glow by plotting twice
        ax.plot(t, signal_np[i], color=sig_color, linewidth=1.5, alpha=0.3)
        ax.plot(t, signal_np[i], color=sig_color, linewidth=0.8)
        
        ax.set_ylabel(LEAD_NAMES[i], color="#00e5ff", fontfamily='Orbitron', fontsize=9,
                      rotation=0, labelpad=25)
        ax.tick_params(colors="#475569", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#1f2937")
    axes[-1].set_xlabel("Timestep (samples)", color="#8a9bb0", fontfamily='Rajdhani', fontsize=10)
    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=axes, location="right", fraction=0.012, pad=0.01)
    cbar.set_label(cbar_label, color="#00e5ff", fontfamily='Rajdhani', fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="#00e5ff", labelsize=8)
    plt.subplots_adjust(hspace=0.05)
    return fig


def make_lead_bar(values, title, cmap_name):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#050505")
    ax.set_facecolor("#050505")
    norm   = values / (values.max() + 1e-8)
    colors = cm.get_cmap(cmap_name)(norm)
    ax.barh(LEAD_NAMES[::-1], values[::-1], color=colors[::-1],
            edgecolor="#1f2937", height=0.6)
    
    ax.set_xlabel("Importance", color="#8a9bb0", fontfamily='Rajdhani', fontsize=10)
    ax.set_title(title, color="#f0f0f0", fontfamily='Orbitron', fontsize=11, pad=15)
    ax.tick_params(colors="#00e5ff", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#1f2937")
    plt.tight_layout()
    return fig


# ── Upload ─────────────────────────────────────────────────────────────────
col_up, col_info = st.columns([1.6, 1], gap="large")

with col_up:
    st.markdown('<p class="section-title">Upload ECG Record</p>', unsafe_allow_html=True)
    dat_file = st.file_uploader("Upload .dat file", type=["dat"])
    hea_file = st.file_uploader("Upload .hea file", type=["hea"])

with col_info:
    st.markdown('<p class="section-title">XAI Methods Included</p>', unsafe_allow_html=True)
    for icon, name, desc in [
        ("📈", "Attention",            "Transformer self-attention weights"),
        ("🔥", "GradCAM",              "Gradient-weighted class activation"),
        ("⚡", "Integrated Gradients", "Axiomatic attribution (ICML 2017)"),
        ("🎯", "SHAP",                 "Shapley values (NeurIPS 2017)"),
        ("🕶️", "Occlusion",            "Patch masking sensitivity"),
    ]:
        st.markdown(f"""
        <div class="metric-card" style="padding:0.7rem 1rem;margin-bottom:0.4rem;">
            <span style="font-family:'Space Mono',monospace;font-size:0.78rem;color:var(--accent);">
                {icon} {name}</span><br>
            <span style="font-size:0.8rem;color:#64748b;">{desc}</span>
        </div>""", unsafe_allow_html=True)


# ── Inference ──────────────────────────────────────────────────────────────
if dat_file and hea_file:
    st.divider()
    with st.spinner("Loading model..."):
        model, all_codes, device, loaders, bg = load_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        base     = dat_file.name.replace(".dat", "")
        dat_path = os.path.join(tmpdir, dat_file.name)
        hea_path = os.path.join(tmpdir, hea_file.name)
        with open(dat_path, "wb") as f: f.write(dat_file.read())
        with open(hea_path, "wb") as f: f.write(hea_file.read())

        try:
            with st.spinner("Analyzing ECG..."):
                record    = wfdb.rdrecord(os.path.join(tmpdir, base))
                signal_np = record.p_signal.astype(np.float32)
                signal_np = preprocess_signal(signal_np, cfg.SEQ_LEN)
                signal    = torch.from_numpy(signal_np).unsqueeze(0).to(device)

                with torch.no_grad():
                    class_logits, super_logits, attn_weights = model(signal)

                class_probs  = torch.sigmoid(class_logits)[0].cpu().numpy()
                super_probs  = torch.sigmoid(super_logits)[0].cpu().numpy()
                pred_classes = decode_scp_labels(
                    (class_probs >= 0.5).astype(float), all_codes)
                pred_supers  = decode_superclass_labels(
                    (super_probs >= 0.5).astype(float))
                pred_idx     = np.where(class_probs >= 0.5)[0].tolist()
                top_class    = (pred_idx[int(np.argmax(class_probs[pred_idx]))]
                                if pred_idx else int(np.argmax(class_probs)))
                top_name     = all_codes[top_class]

            # ── Results ───────────────────────────────────────────────────
            st.markdown('<p class="section-title">Diagnosis Results</p>',
                        unsafe_allow_html=True)
            
            r1, r2 = st.columns([1, 1.4], gap="large")

            with r1:
                st.markdown("**Detected Superclasses**")
                badges = "".join(f'<span class="sc-badge sc-{s}">{s}</span>'
                                 for s in pred_supers) \
                         if pred_supers \
                         else '<span class="sc-badge sc-NONE">NONE</span>'
                st.markdown(badges, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<p class="section-title">Superclass Confidence</p>',
                            unsafe_allow_html=True)
                
                # Display superclass bars in a narrower column for better aesthetics
                for i, sc in enumerate(cfg.SUPERCLASSES):
                    pct    = int(super_probs[i] * 100)
                    active = sc in pred_supers
                    grad   = ("linear-gradient(90deg,#00e5ff,#a3e635)"
                              if active else "#374151")
                    st.markdown(f"""
                    <div class="scp-bar-wrap">
                        <span class="scp-name" style="color:{'var(--accent)' if active else 'var(--muted)'}">
                            {'✦ ' if active else '  '}{sc}</span>
                        <div class="scp-track">
                            <div class="scp-fill" style="width:{pct}%;background:{grad}"></div>
                        </div>
                        <span class="scp-pct">{pct}%</span>
                    </div>""", unsafe_allow_html=True)

            with r2:
                st.markdown("**Top 15 SCP Code Probabilities**")
                for idx in np.argsort(class_probs)[::-1][:15]:
                    code   = all_codes[idx]
                    pct    = int(class_probs[idx] * 100)
                    active = class_probs[idx] >= 0.5
                    grad   = ("linear-gradient(90deg,#ff4d6d,#ff9a3c)"
                              if active else "#374151")
                    st.markdown(f"""
                    <div class="scp-bar-wrap">
                        <span class="scp-name" style="color:{'var(--accent)' if active else 'var(--muted)'}">
                            {'✦ ' if active else '  '}{code}</span>
                        <div class="scp-track">
                            <div class="scp-fill" style="width:{pct}%;background:{grad}"></div>
                        </div>
                        <span class="scp-pct">{pct}%</span>
                    </div>""", unsafe_allow_html=True)

            # ── XAI ───────────────────────────────────────────────────────
            st.markdown('<p class="section-title">Explainability (XAI)</p>',
                        unsafe_allow_html=True)
            st.markdown(f"**XAI target:** `{top_name}` "
                        f"*(confidence: {class_probs[top_class]*100:.1f}%)*")

            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📈 Attention",
                "🔥 GradCAM",
                "⚡ Integrated Gradients",
                "🎯 SHAP",
                "🕶️ Occlusion",
            ])

            with tab1:
                st.markdown('<div class="xai-info"><b style="color:#00e5ff;">Attention Visualization</b> — Self-attention weights from last transformer layer. Shows which time patches the model focused on. Fast, intrinsic to PatchTST architecture.</div>', unsafe_allow_html=True)
                imp  = attn_to_signal_importance(attn_weights)[0]
                c1, c2 = st.columns([2.5, 1])
                with c1:
                    f = make_ecg_fig(signal_np, imp, "Attention Heatmap",
                                     "#00e5ff", "plasma", "Attention")
                    st.pyplot(f); plt.close(f)
                with c2:
                    f = make_lead_bar(imp.mean(axis=-1), "Lead Importance", "plasma")
                    st.pyplot(f); plt.close(f)

            with tab2:
                st.markdown('<div class="xai-info"><b style="color:#ff4d6d;">GradCAM</b> — Hooks into last transformer block. Weights activations by gradient magnitude w.r.t target class. Class-specific, post-hoc saliency method.</div>', unsafe_allow_html=True)
                with st.spinner("Computing GradCAM..."):
                    gc  = GradCAM(model)
                    cam = gc.compute(signal, top_class)
                    gc.remove_hooks()
                c1, c2 = st.columns([2.5, 1])
                with c1:
                    f = make_ecg_fig(signal_np, cam, f"GradCAM → {top_name}",
                                     "#a3e635", "hot", "GradCAM")
                    st.pyplot(f); plt.close(f)
                with c2:
                    f = make_lead_bar(cam.mean(axis=-1), "Lead Importance", "hot")
                    st.pyplot(f); plt.close(f)

            with tab3:
                st.markdown('<div class="xai-info"><b style="color:#a3e635;">Integrated Gradients</b> (Sundararajan et al., ICML 2017) — Integrates gradients from zero baseline to input. Satisfies Sensitivity + Implementation Invariance axioms. Most rigorous gradient-based method.</div>', unsafe_allow_html=True)
                steps = st.slider("Integration steps", 20, 100, 50, 10)
                with st.spinner("Computing Integrated Gradients..."):
                    ig   = IntegratedGradients(model)
                    attr = ig.compute(signal, top_class, n_steps=steps)
                    chk  = ig.completeness_check(signal, attr, top_class)
                st.markdown(f"""
                <div style="background:#0a1628;border:1px solid #1f2937;border-radius:6px;
                     padding:0.6rem 1rem;margin-bottom:0.8rem;font-family:'Space Mono',monospace;
                     font-size:0.7rem;color:#64748b;">
                    Completeness: F(x)={chk['f_input']:.4f} | F(x')={chk['f_baseline']:.4f} |
                    Δ={chk['delta']:.4f} | ΣAttr={chk['attr_sum']:.4f} | Error={chk['error_%']:.1f}%
                </div>""", unsafe_allow_html=True)
                c1, c2 = st.columns([2.5, 1])
                with c1:
                    f = make_ecg_fig(signal_np, attr, f"Integrated Gradients → {top_name}",
                                     "#a3e635", "coolwarm", "IG Attribution")
                    st.pyplot(f); plt.close(f)
                with c2:
                    f = make_lead_bar(attr.mean(axis=-1), "Lead Attribution", "coolwarm")
                    st.pyplot(f); plt.close(f)

            with tab4:
                st.markdown('<div class="xai-info"><b style="color:#fcd34d;">Quantum SHAP</b> — Game-theoretic attribution. Blue/Cyan regions push the prediction <b>DOWN</b>, while Red/Orange regions push it <b>UP</b> for the target class. This provides a bidirectional view of model logic.</div>', unsafe_allow_html=True)
                with st.spinner("Computing SHAP values (~30s)..."):
                    se  = ECGShapExplainer(model, bg)
                    # Compute signed values for bidirectional creativity
                    shv_signed = se.compute(signal, top_class, absolute=False)
                
                c1, c2 = st.columns([2.5, 1])
                with c1:
                    # Using coolwarm for signed SHAP (Red for positive, Blue for negative)
                    f = make_ecg_fig(signal_np, (shv_signed + 1) / 2, f"SHAP (Bidirectional) → {top_name}",
                                     "#f0f0f0", "coolwarm", "Push (Blue: - | Red: +)")
                    st.pyplot(f); plt.close(f)
                with c2:
                    # Creative Radial Plot for Lead Importance
                    f = make_shap_radial_plot(shv_signed, "Lead Impact Radar")
                    st.pyplot(f); plt.close(f)

            with tab5:
                st.markdown('<div class="xai-info"><b style="color:#f0abfc;">Occlusion Sensitivity</b> — Masks each patch with zeros, measures prediction score drop. Most intuitive and faithful method: directly tests what happens when each region is removed. No gradients needed.</div>', unsafe_allow_html=True)
                with st.spinner("Running occlusion sensitivity..."):
                    oc      = OcclusionSensitivity(model)
                    occ_map = oc.compute(signal, top_class, per_lead=True)
                    drops   = oc.compute_lead_sensitivity(signal, top_class)
                c1, c2 = st.columns([2.5, 1])
                with c1:
                    f = make_ecg_fig(signal_np, occ_map, f"Occlusion → {top_name}",
                                     "#f0abfc", "YlOrRd", "Score Drop")
                    st.pyplot(f); plt.close(f)
                with c2:
                    f = make_lead_bar(drops, "Lead Sensitivity", "YlOrRd")
                    st.pyplot(f); plt.close(f)

        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)

else:
    st.markdown("""
    <div style="text-align:center;padding:4rem;">
        <div style="font-size:3.5rem;">🫀</div>
        <div style="font-family:'Space Mono',monospace;font-size:0.8rem;
             letter-spacing:3px;color:#4b5563;margin-top:1rem;">
            UPLOAD .DAT + .HEA FILES TO BEGIN
        </div>
        <div style="font-size:0.82rem;color:#374151;margin-top:0.6rem;">
            Files are in your PTB-XL records100/ or records500/ folders
        </div>
    </div>""", unsafe_allow_html=True)
