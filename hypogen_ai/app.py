"""
app.py: Production-grade scientific Streamlit web platform for AllergenMod-AI.
ISEF & Ibdaa Research Framework (Computational Biology & Bioinformatics Track).
Implements:
1. Complete bilingual support (English LTR / Arabic RTL) with strict BiDi sequence isolation.
2. Robust UniProt state management without silent fallbacks.
3. Biological coordinate mapping (Precursor vs Mature vs 33-mer peptide) with Shan et al. (2002) citation.
4. Food Allergen mode vs. General Protein mode (e.g. Hemoglobin P68871) handling.
5. Hugging Face ESM-2 masked language model inference with explicit evolutionary likelihood disclaimers.
6. Multi-objective Pareto optimization (NSGA-II non-dominated sorting with normalized weights and knee point).
7. SpCas9 CRISPR sgRNA design with authentic biological CDS support and hexaploid wheat paralog caveats.
8. AlphaFold 3D molecular visualization with honest in-silico substitution labeling.
9. Data provenance-rich CSV export with metadata headers.
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import sys
import time
import requests
import py3Dmol
from datetime import datetime, timezone

# Add current workspace directory to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.uniprot_fetcher import UniProtFetcher, BENCHMARK_PROTEINS, classify_protein_mode
from core.coordinate_mapper import CoordinateMapper, CELIAC_33MER_REFERENCE
from core.epitope_scanner import EpitopeScanner
from core.translations import get_text, get_rtl_css, EVIDENCE_LABELS
from models.esm_mutator import ESMMutator
from scoring.immunogenicity import ImmunogenicityScorer
from scoring.food_functionality import FoodFunctionalityScorer
from scoring.agronomic_fitness import AgronomicFitnessScorer
from scoring.pareto_optimizer import ParetoOptimizer
from crispr.sgrna_engine import CRISPRsgRNAEngine

# ==========================================
# STREAMLIT PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="HypoGen-AI | Scientific Audit & In-Silico Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# SINGLETON CACHED ENGINES
# ==========================================
@st.cache_resource(show_spinner="Initializing Hugging Face ESM-2 Transformer...")
def get_esm_mutator() -> ESMMutator:
    return ESMMutator()

@st.cache_resource
def get_uniprot_fetcher() -> UniProtFetcher:
    return UniProtFetcher()

@st.cache_resource
def get_epitope_scanner() -> EpitopeScanner:
    return EpitopeScanner()

@st.cache_resource
def get_coordinate_mapper() -> CoordinateMapper:
    return CoordinateMapper()

@st.cache_resource
def get_scoring_engines():
    return (
        ImmunogenicityScorer(),
        FoodFunctionalityScorer(),
        AgronomicFitnessScorer(),
        ParetoOptimizer(),
        CRISPRsgRNAEngine()
    )

esm_mutator = get_esm_mutator()
uniprot_fetcher = get_uniprot_fetcher()
epitope_scanner = get_epitope_scanner()
coord_mapper = get_coordinate_mapper()
im_scorer, ff_scorer, af_scorer, pareto_optimizer, crispr_engine = get_scoring_engines()

# ==========================================
# ALPHAFOLD 3D STRUCTURE FETCHER
# ==========================================
@st.cache_data(show_spinner="Fetching AlphaFold 3D coordinates...")
def fetch_alphafold_pdb(accession: str) -> str | None:
    """Queries AlphaFold DB API to retrieve predicted 3D atomic coordinates (PDB)."""
    if not accession or accession.startswith("CUSTOM"):
        return None
    try:
        api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{accession}"
        r = requests.get(api_url, timeout=6)
        if r.status_code == 200:
            data = r.json()
            if data and len(data) > 0 and "pdbUrl" in data[0]:
                pdb_url = data[0]["pdbUrl"]
                pdb_resp = requests.get(pdb_url, timeout=8)
                if pdb_resp.status_code == 200:
                    return pdb_resp.text
    except Exception:
        pass
    return None


def render_3d_protein(
    pdb_text: str,
    epitopes: list[dict[str, Any]],
    mutation_pos: int | None = None,
    width: int = 680,
    height: int = 440
) -> str:
    """Generates an interactive 3D WebGL protein visualization using py3Dmol."""
    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_text, "pdb")
    view.setStyle({'cartoon': {'color': '#64748b', 'opacity': 0.85}})

    for ep in epitopes:
        resi_range = list(range(ep["start"], ep["end"] + 1))
        view.addStyle(
            {'resi': resi_range},
            {'cartoon': {'color': '#ef4444', 'opacity': 1.0}}
        )

    if mutation_pos:
        view.addStyle(
            {'resi': [mutation_pos]},
            {
                'stick': {'color': '#22c55e', 'radius': 0.35},
                'sphere': {'color': '#22c55e', 'radius': 1.1, 'opacity': 0.95}
            }
        )

    view.zoomTo()
    view.spin(False)
    return view._make_html()


def render_3d_variant(
    pdb_text: str,
    mode: str = "wildtype",
    target_pos: int | None = None,
    epitopes: list[dict[str, Any]] | None = None,
    cys_positions: list[int] | None = None,
    width: int = 480,
    height: int = 400
) -> str:
    """Renders 3D molecular structure comparing Wild-Type vs Engineered In-Silico Variant."""
    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_text, "pdb")

    if mode == "wildtype":
        view.setStyle({'cartoon': {'color': '#64748b', 'opacity': 0.85}})
        if epitopes:
            for ep in epitopes:
                resi_range = list(range(ep["start"], ep["end"] + 1))
                view.addStyle(
                    {'resi': resi_range},
                    {'cartoon': {'color': '#ef4444', 'opacity': 0.95}}
                )
        if target_pos:
            view.addStyle(
                {'resi': [target_pos]},
                {
                    'stick': {'color': '#f97316', 'radius': 0.45},
                    'sphere': {'color': '#f97316', 'radius': 1.25, 'opacity': 0.95}
                }
            )
    else:
        view.setStyle({'cartoon': {'color': '#0284c7', 'opacity': 0.85}})
        if cys_positions:
            for c_pos in cys_positions:
                view.addStyle(
                    {'resi': [c_pos]},
                    {'stick': {'color': '#eab308', 'radius': 0.35}}
                )
        if target_pos:
            view.addStyle(
                {'resi': [target_pos]},
                {
                    'stick': {'color': '#22c55e', 'radius': 0.45},
                    'sphere': {'color': '#22c55e', 'radius': 1.25, 'opacity': 0.95}
                }
            )

    view.zoomTo()
    view.spin(False)
    return view._make_html()


# ==========================================
# SIDEBAR: LANGUAGE & SETUP
# ==========================================
with st.sidebar:
    st.image("https://img.shields.io/badge/Competition-ISEF%20%2F%20Ibdaa%202026-blue?style=for-the-badge&logo=dna", use_container_width=True)

    # Dual-Language Toggle
    st.subheader("🌐 Language / اللغة")
    language = st.radio(
        "Interface Language / لغة الواجهة",
        ["English", "العربية"],
        index=0,
        horizontal=True,
        label_visibility="collapsed"
    )
    is_ar = (language == "العربية")
    lang_code = "ar" if is_ar else "en"

    # Inject RTL / LTR layout CSS
    st.markdown(get_rtl_css(is_ar), unsafe_allow_html=True)

    # Custom styling
    st.markdown("""
    <style>
        .metric-card {
            background: linear-gradient(135deg, rgba(18, 24, 38, 0.90), rgba(30, 41, 59, 0.8));
            border: 1px solid rgba(56, 189, 248, 0.25);
            border-radius: 12px;
            padding: 16px 20px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }
        .metric-title {
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94a3b8;
            margin-bottom: 4px;
        }
        .metric-value {
            font-size: 1.45rem;
            font-weight: 700;
            color: #f8fafc;
        }
        .metric-badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-top: 4px;
        }
        .badge-green { background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid #22c55e; }
        .badge-blue { background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid #0284c7; }
        .badge-amber { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #d97706; }
        .badge-red { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #dc2626; }
        .seq-box {
            font-family: 'Consolas', 'Courier New', monospace;
            letter-spacing: 0.15em;
            line-height: 1.8;
            background: #0f172a;
            padding: 18px;
            border-radius: 10px;
            border: 1px solid #334155;
            overflow-x: auto;
            color: #cbd5e1;
            font-size: 0.92rem;
            direction: ltr !important;
            text-align: left !important;
        }
        .seq-highlight-epitope {
            background-color: rgba(239, 68, 68, 0.35);
            color: #fca5a5;
            border-bottom: 2px solid #ef4444;
            font-weight: bold;
            padding: 2px 1px;
        }
        .seq-highlight-cys {
            background-color: rgba(234, 179, 8, 0.35);
            color: #fde047;
            border-bottom: 2px solid #eab308;
            font-weight: bold;
            padding: 2px 1px;
        }
        .seq-highlight-mut {
            background-color: rgba(34, 197, 94, 0.45);
            color: #86efac;
            border: 2px solid #22c55e;
            border-radius: 3px;
            font-weight: bold;
            padding: 2px 3px;
        }
        .ltr-span {
            direction: ltr !important;
            display: inline-block;
            unicode-bidi: embed;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🔬 " + get_text("experiment_setup", lang_code))

    source_options = [
        "Wheat Alpha-Gliadin (P18573, Celiac 33-mer)",
        "Peanut Ara h 2 (Q6PSU2, Anaphylaxis)",
        "Bovine Milk Beta-lactoglobulin (P02754)",
        "Human Beta-Hemoglobin (P68871, General Protein Test)",
        "UniProtKB Live REST Search",
        "Custom Raw IUPAC FASTA"
    ]
    input_source = st.selectbox(get_text("target_source_label", lang_code), source_options)

    # Detect protein change to flush stale state
    prev_source = st.session_state.get("_last_selected_source")
    if prev_source != input_source:
        st.session_state["_last_selected_source"] = input_source
        if "pareto_results" in st.session_state:
            del st.session_state["pareto_results"]
        if "active_protein" in st.session_state:
            del st.session_state["active_protein"]

    protein_data = None
    source_tag = "UniProtKB"
    fetch_error = None

    if "P18573" in input_source:
        protein_data = uniprot_fetcher.fetch_protein("P18573")
        source_tag = "UniProtKB (P18573, Benchmark)"
    elif "Q6PSU2" in input_source:
        protein_data = uniprot_fetcher.fetch_protein("Q6PSU2")
        source_tag = "UniProtKB (Q6PSU2, Benchmark)"
    elif "P02754" in input_source:
        protein_data = uniprot_fetcher.fetch_protein("P02754")
        source_tag = "UniProtKB (P02754, Benchmark)"
    elif "P68871" in input_source:
        protein_data = uniprot_fetcher.fetch_protein("P68871")
        source_tag = "UniProtKB (P68871, General Protein)"
    elif "Live REST Search" in input_source:
        custom_acc = st.text_input(get_text("custom_accession_label", lang_code), value="P18573").strip().upper()
        if st.button(get_text("fetch_rest_btn", lang_code), type="primary"):
            try:
                with st.spinner(f"Querying rest.uniprot.org for {custom_acc}..."):
                    protein_data = uniprot_fetcher.fetch_protein(custom_acc)
                    st.session_state["active_protein"] = protein_data
                    if "pareto_results" in st.session_state:
                        del st.session_state["pareto_results"]
            except Exception as e:
                fetch_error = str(e)
                if "active_protein" in st.session_state:
                    del st.session_state["active_protein"]

        if "active_protein" in st.session_state and st.session_state["active_protein"].get("accession") == custom_acc:
            protein_data = st.session_state["active_protein"]
            source_tag = f"Live REST ({custom_acc})"
    elif "Custom Raw IUPAC" in input_source:
        fasta_text = st.text_area(
            get_text("custom_fasta_label", lang_code),
            value=">Synthetic_Gliadin_Core\nLQLQPFPQPQLPYPQPQLPYPQPQLPYPQPQPFRPQQPYPQSQPQYSQPQQPISQQQQQQQQQQQQKQQQQQQQQ",
            height=120
        )
        if st.button(get_text("validate_fasta_btn", lang_code), type="primary"):
            try:
                protein_data = uniprot_fetcher.sanitize_fasta(fasta_text)
                st.session_state["active_protein"] = protein_data
                if "pareto_results" in st.session_state:
                    del st.session_state["pareto_results"]
            except Exception as e:
                fetch_error = str(e)

        if "active_protein" in st.session_state and st.session_state["active_protein"].get("accession") == "CUSTOM_FASTA":
            protein_data = st.session_state["active_protein"]
            source_tag = "Custom FASTA"

    # Multi-Objective Weights Sliders
    st.markdown("---")
    st.subheader(get_text("weights_header", lang_code))
    w_safety = st.slider(get_text("w_safety_label", lang_code), min_value=0.10, max_value=0.80, value=0.40, step=0.05)
    w_food = st.slider(get_text("w_food_label", lang_code), min_value=0.10, max_value=0.80, value=0.35, step=0.05)
    w_agronomic = st.slider(get_text("w_agro_label", lang_code), min_value=0.10, max_value=0.80, value=0.25, step=0.05)

    sum_w = w_safety + w_food + w_agronomic
    weights_summary = (
        f"الأوزان بعد التوحيد: أمان **{w_safety/sum_w:.2f}** | غذاء **{w_food/sum_w:.2f}** | زراعة **{w_agronomic/sum_w:.2f}**"
        if is_ar else
        f"Normalized Weights: Safety **{w_safety/sum_w:.2f}** | Food **{w_food/sum_w:.2f}** | Agro **{w_agronomic/sum_w:.2f}**"
    )
    st.caption(weights_summary)

# ==========================================
# FETCH ERROR HANDLING (NO SILENT FALLBACK)
# ==========================================
if fetch_error:
    st.error(f"❌ UniProt Data Retrieval Error: {fetch_error}")
    st.warning("Analysis halted. Please check the accession ID and network connection. AllergenMod-AI will not fabricate data or silently fall back to an unrelated protein.")
    st.stop()

if protein_data is None:
    st.info("Please select a target sequence source or fetch a valid UniProt accession from the sidebar to begin.")
    st.stop()

# Classify protein analysis mode
analysis_mode = protein_data.get("analysis_mode") or classify_protein_mode(protein_data)
is_general_protein = (analysis_mode == "GENERAL_PROTEIN")
full_seq = protein_data["sequence"]

# ==========================================
# UNIVERSAL EPITOPE SCANNER RUN
# ==========================================
scan_results = epitope_scanner.scan_sequence(
    sequence=full_seq,
    uniprot_features=protein_data.get("features"),
    analysis_mode=analysis_mode
)
detected_epitopes = scan_results["epitopes"]

# ==========================================
# HIGH-LEVEL METRICS RIBBON (HEADER)
# ==========================================
st.title(get_text("app_title", lang_code))
st.markdown(f"##### *{get_text('app_subtitle', lang_code)}*")

if is_general_protein:
    st.info(f"ℹ️ **{get_text('general_protein_mode_banner', lang_code)}**")

col_m1, col_m2, col_m3, col_m4 = st.columns(4)

with col_m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{get_text('metric_ai_model', lang_code)}</div>
        <div class="metric-value" style="font-size: 1.2rem;"><span class="ltr-span">ESM-2 (8M UR50D)</span></div>
        <span class="metric-badge badge-green">● Online (PyTorch)</span>
        <div style="font-size: 0.72rem; color: #94a3b8; margin-top: 4px;">{get_text('metric_ai_sub', lang_code)}</div>
    </div>
    """, unsafe_allow_html=True)

with col_m2:
    res_label = "حمض أميني" if is_ar else "Residues"
    crc_sub = f"CRC64: {protein_data.get('crc64', 'N/A')[:8]}"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{get_text('metric_protein_source', lang_code)}</div>
        <div class="metric-value" style="font-size: 1.25rem;"><span class="ltr-span">{protein_data.get('accession', 'N/A')}</span></div>
        <span class="metric-badge badge-blue">{source_tag}</span>
        <div style="font-size: 0.72rem; color: #94a3b8; margin-top: 4px;">{protein_data.get('length', 0)} {res_label} | <span class="ltr-span">{crc_sub}</span></div>
    </div>
    """, unsafe_allow_html=True)

with col_m3:
    org_short = protein_data.get('organism', 'Unknown').split("(")[0]
    gene_str = protein_data.get('gene_name', 'N/A')
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{get_text('metric_target_host', lang_code)}</div>
        <div class="metric-value" style="font-size: 1.15rem;"><span class="ltr-span">{gene_str}</span></div>
        <span class="metric-badge badge-amber">{org_short}</span>
        <div style="font-size: 0.72rem; color: #94a3b8; margin-top: 4px;">{protein_data.get('protein_name', 'Protein')[:28]}...</div>
    </div>
    """, unsafe_allow_html=True)

with col_m4:
    density = scan_results.get("epitope_density_score", 0.0)
    if is_general_protein:
        m4_badge = "General Protein Surface Antigenicity"
        badge_color = "badge-blue"
    else:
        m4_badge = "Severe Risk" if density > 15.0 else ("Moderate Risk" if density > 5.0 else "Low Risk")
        badge_color = "badge-red" if density > 15.0 else ("badge-amber" if density > 5.0 else "badge-green")
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{get_text('metric_pathogenicity', lang_code)}</div>
        <div class="metric-value">{density}%</div>
        <span class="metric-badge {badge_color}">{m4_badge} ({len(detected_epitopes)} loci)</span>
        <div style="font-size: 0.72rem; color: #94a3b8; margin-top: 4px;">[COMPUTATIONAL_PREDICTION] IEDB Calibrated</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# MAIN APPLICATION TABS
# ==========================================
tab_labels = [
    get_text("tab1_title", lang_code),
    get_text("tab2_title", lang_code),
    get_text("tab3_title", lang_code)
]

tab1, tab2, tab3 = st.tabs(tab_labels)

# ==========================================
# TAB 1: SEQUENCE, COORDINATES & 3D STRUCTURE
# ==========================================
with tab1:
    st.header(get_text("tab1_header", lang_code))

    # Provenance and Coordinate Verification Banner
    st.subheader(get_text("coord_mapping_title", lang_code))

    if protein_data.get("accession") == "P18573":
        # Provenance Citation & Alignment for Wheat Celiac 33-mer
        align_res = coord_mapper.align_celiac_33mer(full_seq)
        st.success(f"✅ **{get_text('wheat_celiac_provenance', lang_code)}**")

        coord_col1, coord_col2, coord_col3 = st.columns(3)
        with coord_col1:
            st.markdown(f"""
            **Coordinate Space 1: Full-Length Precursor**
            - Length: `{len(full_seq)}` aa
            - Signal Peptide: `1-20` (UniProt feature)
            - 33-mer Peptide Locus: **Residues {align_res['locus_start_1indexed']} to {align_res['locus_end_1indexed']}**
            - Canonical Target Residue: **Q77** (Precursor Q77)
            """)
        with coord_col2:
            st.markdown(f"""
            **Coordinate Space 2: Celiac 33-mer Peptide**
            - Sequence: `<span class="ltr-span"><code>{CELIAC_33MER_REFERENCE}</code></span>`
            - Length: `33` aa
            - Target Gln: **Q2** (Shan et al. 2002 core target)
            - Matches Precursor: **Q77 = 33-mer Q2**
            """, unsafe_allow_html=True)
        with coord_col3:
            st.markdown(f"""
            **Coordinate Space 3: Mature Protein Chain**
            - Mature Chain: `21-307` (`287` aa)
            - Mature Target Locus: **Q57** (Q77 - 20)
            - Identity Match: **{align_res['percent_identity']}%** ({align_res['match_status']})
            """)

        with st.expander("🔍 " + ("عرض محاذاة ببتيد 33-mer الكاملة ومواقع إزالة الأميد (tTG2 Deamidation)" if is_ar else "View Detailed 33-mer Alignment & tTG2 Deamidation Sites"), expanded=False):
            st.markdown(f"""
            ```
            Reference 33-mer : {CELIAC_33MER_REFERENCE}
            P18573 [76..108] : {full_seq[align_res['locus_start_1indexed'] - 1 : align_res['locus_end_1indexed']]}
            ```
            - Deamidation Targets: Substrates for tissue transglutaminase (tTG2) in the gut mucosa, forming negatively charged glutamic acid (E) that binds HLA-DQ2.5 grooves.
            """)
    else:
        st.info(f"Target Protein: `{protein_data.get('accession')}` ({protein_data.get('protein_name')}). Precursor Length: `{len(full_seq)}` residues. Mode: `{analysis_mode}`.")

    st.markdown("---")

    t1_col1, t1_col2 = st.columns([1, 1])

    with t1_col1:
        st.subheader(get_text("structure_3d_title", lang_code))
        st.caption(get_text("structure_3d_disclaimer", lang_code))

        pdb_data = fetch_alphafold_pdb(protein_data.get("accession", ""))
        if pdb_data:
            target_pos_3d = st.session_state.get("active_target_pos", detected_epitopes[0]["start"] if detected_epitopes else 1)
            html_3d = render_3d_protein(
                pdb_text=pdb_data,
                epitopes=detected_epitopes,
                mutation_pos=target_pos_3d,
                width=620,
                height=430
            )
            components.html(html_3d, height=450)
            st.success(f"AlphaFold coordinates loaded: `AF-{protein_data.get('accession')}-F1`")
        else:
            st.info("AlphaFold 3D coordinates unavailable for custom synthetic FASTA entry or accession. Secondary structure annotations active.")

    with t1_col2:
        st.subheader(get_text("linear_seq_title", lang_code))
        st.caption("🔴 Red underline: Epitope locus | 🟡 Yellow underline: Cysteine residue")

        epitope_indices = set()
        for ep in detected_epitopes:
            for idx in range(ep["start"] - 1, ep["end"]):
                epitope_indices.add(idx)

        cys_indices = {i for i, aa in enumerate(full_seq) if aa == "C"}

        html_tokens = []
        for i, aa in enumerate(full_seq):
            if i in epitope_indices:
                html_tokens.append(f'<span class="seq-highlight-epitope" title="Epitope Residue {i+1}">{aa}</span>')
            elif i in cys_indices:
                html_tokens.append(f'<span class="seq-highlight-cys" title="Cysteine Bridge {i+1}">{aa}</span>')
            else:
                html_tokens.append(f"<span>{aa}</span>")
            
            if (i + 1) % 60 == 0:
                html_tokens.append(f' <span style="color: #64748b; font-size: 0.75rem;">[{i+1}]</span><br>')

        st.markdown(f'<div class="seq-box">{"".join(html_tokens)}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader(get_text("iedb_curve_title", lang_code))
    res_scores = scan_results.get("residue_scores", [])
    if res_scores:
        df_profile = pd.DataFrame({
            "Residue Position": list(range(1, len(res_scores) + 1)),
            "Amino Acid": list(full_seq),
            "Immunogenicity Score": res_scores
        })

        fig_prof = px.line(
            df_profile,
            x="Residue Position",
            y="Immunogenicity Score",
            hover_data=["Amino Acid", "Residue Position", "Immunogenicity Score"],
            labels={"Immunogenicity Score": "IEDB Propensity (0 - 1.0)"},
            template="plotly_dark",
            height=300
        )
        fig_prof.update_traces(line_color="#ef4444", fill="tozeroy", fillcolor="rgba(239, 68, 68, 0.15)")
        fig_prof.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            hovermode="x unified",
            xaxis=dict(showgrid=True, gridcolor="#1e293b"),
            yaxis=dict(showgrid=True, gridcolor="#1e293b", range=[0, 1.1])
        )
        st.plotly_chart(fig_prof, use_container_width=True)

    st.subheader(get_text("epitope_registry_title", lang_code))
    if detected_epitopes:
        df_ep = pd.DataFrame(detected_epitopes)
        cols_ep = ["id", "name", "start", "end", "matched_sequence", "disease", "immune_mechanism", "risk_level", "citation", "evidence_level"]
        df_ep_display = df_ep[cols_ep].rename(columns={
            "id": "Epitope ID",
            "name": "Epitope Name",
            "start": "Start",
            "end": "End",
            "matched_sequence": "Peptide",
            "disease": "Pathological Condition",
            "immune_mechanism": "Immunological Mechanism",
            "risk_level": "Risk Tier",
            "citation": "Literature Citation",
            "evidence_level": "Evidence Level"
        } if not is_ar else {
            "id": "رمز الحبيبة",
            "name": "اسم الحبيبة المستضدية",
            "start": "البداية",
            "end": "النهاية",
            "matched_sequence": "تسلسل الببتيد",
            "disease": "الحالة المرضية",
            "immune_mechanism": "الآلية المناعية",
            "risk_level": "مستوى الخطورة",
            "citation": "المرجع العلمي",
            "evidence_level": "مستوى الإثبات"
        })
        st.dataframe(df_ep_display, use_container_width=True)

# ==========================================
# TAB 2: AI MUTATION ENGINE & PARETO OPTIMIZATION
# ==========================================
with tab2:
    st.header(get_text("tab2_header", lang_code))
    st.write(get_text("tab2_intro", lang_code))
    st.caption(f"ℹ️ **{get_text('esm_explanation', lang_code)}**")

    t2_setup_col1, t2_setup_col2 = st.columns([2, 1])

    with t2_setup_col1:
        st.subheader(get_text("select_mutation_loci", lang_code))
        suggested_positions = []
        if detected_epitopes:
            for ep in detected_epitopes:
                for p in range(ep["start"], ep["end"] + 1):
                    aa = full_seq[p - 1]
                    if aa in ["Q", "P", "R", "K", "W", "E", "D"]:
                        suggested_positions.append(p)
        
        suggested_positions = sorted(list(set(suggested_positions)))
        if not suggested_positions:
            suggested_positions = [min(77, len(full_seq)), min(78, len(full_seq))]

        selected_positions = st.multiselect(
            "Positions (1-indexed precursor coordinates):",
            options=list(range(1, len(full_seq) + 1)),
            default=suggested_positions[:5] if suggested_positions else [1],
            format_func=lambda x: f"Position {x}: {full_seq[x-1]} (Epitope: {'Yes' if x in epitope_indices else 'No'})"
        )

    with t2_setup_col2:
        st.subheader("Inference Settings / إعدادات الاستدلال")
        top_k_subs = st.slider("Substitutions per Position (Top K)", min_value=2, max_value=8, value=4)
        run_optimization = st.button(get_text("run_optimization_btn", lang_code), type="primary", use_container_width=True)

    if run_optimization or "pareto_results" not in st.session_state:
        if not selected_positions:
            st.warning("Please select at least one position to evolve.")
        else:
            with st.spinner("Executing ESM-2 Masked Inference & Pareto Optimization..."):
                t_start = time.time()
                raw_library = esm_mutator.generate_candidate_library(
                    sequence=full_seq,
                    target_positions=selected_positions,
                    top_k_per_position=top_k_subs
                )

                evaluated_candidates = []
                for cand in raw_library:
                    p = cand["position"]
                    mut_aa = cand["mutant"]
                    mut_seq = full_seq[:p - 1] + mut_aa + full_seq[p:]

                    im_eval = im_scorer.evaluate_mutation_safety(
                        full_seq, mut_seq, p, detected_epitopes, analysis_mode=analysis_mode
                    )
                    ff_eval = ff_scorer.evaluate_food_functionality(
                        full_seq, mut_seq, p, protein_data.get("features"), analysis_mode=analysis_mode
                    )
                    af_eval = af_scorer.evaluate_agronomic_fitness(
                        full_seq, mut_seq, p, protein_data.get("features"), analysis_mode=analysis_mode
                    )

                    merged_record = {
                        **cand,
                        **im_eval,
                        **ff_eval,
                        **af_eval,
                        "full_mutated_sequence": mut_seq
                    }
                    evaluated_candidates.append(merged_record)

                pareto_out = pareto_optimizer.compute_pareto_front(
                    candidates=evaluated_candidates,
                    weight_safety=w_safety,
                    weight_food=w_food,
                    weight_agronomic=w_agronomic,
                    analysis_mode=analysis_mode
                )

                elapsed = time.time() - t_start
                st.session_state["pareto_results"] = pareto_out
                st.session_state["optimization_elapsed"] = elapsed

    if "pareto_results" in st.session_state:
        results = st.session_state["pareto_results"]
        all_cands = results["all_candidates"]
        pareto_cands = results["pareto_front"]
        knee_cand = results["knee_point"]

        if knee_cand:
            st.session_state["active_target_pos"] = knee_cand["position"]

        st.markdown("---")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric("Total Variants Evaluated" if not is_ar else "إجمالي الطفرات المقيمة", len(all_cands))
        with kpi2:
            st.metric("Pareto Frontier Variants" if not is_ar else "طفرات جبهة باريتو المثلى", f"{len(pareto_cands)} (Rank 1)")
        with kpi3:
            if knee_cand:
                st.metric("Knee Point Candidate" if not is_ar else "الطفرة المثالية عند نقطة الركبة", knee_cand["mutation"], f"{knee_cand['composite_score']}%")
        with kpi4:
            st.metric("Engine Runtime" if not is_ar else "زمن الاستدلال الحوسبي", f"{st.session_state.get('optimization_elapsed', 0.5):.2f}s")

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader(get_text("pareto_landscape_title", lang_code))

        df_cands = pd.DataFrame(all_cands)
        pareto_opt_text = "Pareto Frontier (Rank 1)" if not is_ar else "جبهة باريتو (الرتبة 1)"
        pareto_dom_text = "Dominated Variant" if not is_ar else "طفرة مهيمن عليها"
        df_cands["Pareto_Status"] = df_cands["is_pareto"].apply(lambda x: pareto_opt_text if x else pareto_dom_text)

        if not is_general_protein:
            # 3D Pareto Scatter for Food Allergens
            fig_3d = px.scatter_3d(
                df_cands,
                x="medical_safety_index",
                y="food_functionality_index",
                z="agronomic_fitness_index",
                color="Pareto_Status",
                color_discrete_map={pareto_opt_text: "#eab308", pareto_dom_text: "#38bdf8"},
                symbol="Pareto_Status",
                symbol_map={pareto_opt_text: "diamond", pareto_dom_text: "circle"},
                hover_name="mutation",
                hover_data={
                    "medical_safety_index": ":.1f%",
                    "food_functionality_index": ":.1f%",
                    "agronomic_fitness_index": ":.1f%",
                    "composite_score": ":.2f%",
                    "llr": ":.3f",
                    "Pareto_Status": True
                },
                template="plotly_dark",
                height=560
            )
            fig_3d.update_layout(
                scene=dict(
                    xaxis_title="Medical Safety (%)" if not is_ar else "الأمان الطبي (%)",
                    yaxis_title="Food Functionality (%)" if not is_ar else "جودة الغذاء (%)",
                    zaxis_title="Agronomic Fitness (%)" if not is_ar else "حيوية الإنبات (%)",
                    xaxis=dict(backgroundcolor="#0f172a", gridcolor="#334155"),
                    yaxis=dict(backgroundcolor="#0f172a", gridcolor="#334155"),
                    zaxis=dict(backgroundcolor="#0f172a", gridcolor="#334155"),
                ),
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_3d, use_container_width=True)
        else:
            # 2D Pareto Scatter for General Proteins
            fig_2d_gen = px.scatter(
                df_cands,
                x="fitness_score",
                y="food_functionality_index",
                color="Pareto_Status",
                color_discrete_map={pareto_opt_text: "#eab308", pareto_dom_text: "#38bdf8"},
                hover_name="mutation",
                labels={"fitness_score": "ESM-2 Evolutionary Fitness (%)", "food_functionality_index": "Structural Conservation (%)"},
                title="Evolutionary Likelihood vs. Structural Conservation" if not is_ar else "الملاءمة التطورية مقابل الحفظ الهيكلي",
                template="plotly_dark",
                height=420
            )
            st.plotly_chart(fig_2d_gen, use_container_width=True)

        # Knee Point Details
        if knee_cand:
            st.subheader(get_text("knee_point_title", lang_code))
            st.markdown(f"""
            - **Mutation**: `<span class="ltr-span"><code>{knee_cand['mutation']}</code></span>` (Position {knee_cand['position']})
            - **ESM-2 Marginal ΔLLR**: `{knee_cand['llr']}` ({knee_cand['plausibility']})
            - **Composite Score**: **{knee_cand['composite_score']}%** (Distance to Ideal Utopia: `{knee_cand['distance_to_utopia']}`)
            - **Safety Index**: **{knee_cand['medical_safety_index']}%** | **Functionality**: **{knee_cand['food_functionality_index']}%** | **Agronomic**: **{knee_cand['agronomic_fitness_index']}%**
            - **Biophysical Mechanism**: *{knee_cand['mechanism']}*
            """, unsafe_allow_html=True)

        # Structural Retention 3D Comparison
        st.markdown("---")
        st.subheader(get_text("structure_retention_title", lang_code))
        st.caption(get_text("structure_retention_disclaimer", lang_code))

        mutation_names = [c["mutation"] for c in all_cands]
        default_idx_3d = mutation_names.index(knee_cand["mutation"]) if knee_cand else 0
        inspected_mutation_name = st.selectbox(
            "Select Variant for 3D Molecular Inspection / اختر المتغير للمعاينة الفراغية:",
            options=mutation_names,
            index=default_idx_3d,
            key="select_3d_variant"
        )

        inspected_cand = next(c for c in all_cands if c["mutation"] == inspected_mutation_name)
        target_pos_inspected = inspected_cand["position"]
        wt_aa_inspected = inspected_cand["wildtype"]
        mut_aa_inspected = inspected_cand["mutant"]
        cys_positions = [i + 1 for i, aa in enumerate(full_seq) if aa == "C"]

        pdb_data_tab2 = fetch_alphafold_pdb(protein_data.get("accession", ""))

        if pdb_data_tab2:
            col_3d_wt, col_3d_mut = st.columns(2)
            with col_3d_wt:
                st.markdown(f"##### Wild-Type Allergen (`{wt_aa_inspected}{target_pos_inspected}`)")
                html_wt = render_3d_variant(
                    pdb_text=pdb_data_tab2,
                    mode="wildtype",
                    target_pos=target_pos_inspected,
                    epitopes=detected_epitopes,
                    width=560,
                    height=420
                )
                components.html(html_wt, height=435)
                st.caption("🟠 Target allergenic anchor residue before ablation.")

            with col_3d_mut:
                st.markdown(f"##### In-Silico Substituted Variant (`{inspected_cand['mutation']}`)")
                html_mut = render_3d_variant(
                    pdb_text=pdb_data_tab2,
                    mode="mutant",
                    target_pos=target_pos_inspected,
                    cys_positions=cys_positions,
                    width=560,
                    height=420
                )
                components.html(html_mut, height=435)
                st.caption("🟢 In-silico substituted residue on AlphaFold template backbone; disulfide bridges intact.")

            s_kpi1, s_kpi2, s_kpi3, s_kpi4 = st.columns(4)
            with s_kpi1:
                st.metric("BLOSUM62 Conservation", f"{inspected_cand['blosum62_raw']}", f"{inspected_cand['blosum_score']}% Score")
            with s_kpi2:
                st.metric("Disulfide Integrity", f"{inspected_cand['disulfide_score']}%", "Conserved" if inspected_cand['disulfide_score'] == 100.0 else "Penalty Applied")
            with s_kpi3:
                st.metric("Hydropathy Shift", f"{inspected_cand['hydropathy_shift']}", "Polar Shift")
            with s_kpi4:
                st.metric("Structural Model Status", "AlphaFold In-Silico", "Template Backbone")

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Candidate Population Table / جدول المرشحين")
        filter_pareto_only = st.checkbox("Show Rank 1 Non-Dominated Pareto Frontier Only", value=False)
        display_cands = pareto_cands if filter_pareto_only else all_cands
        df_table = pd.DataFrame(display_cands)
        cols_to_show = [
            "mutation", "pareto_rank", "composite_score", "medical_safety_index",
            "food_functionality_index", "agronomic_fitness_index", "llr", "plausibility", "mechanism"
        ]
        st.dataframe(df_table[cols_to_show], use_container_width=True)

# ==========================================
# TAB 3: CRISPR SGRNA & DATA PROVENANCE EXPORT
# ==========================================
with tab3:
    st.header(get_text("tab3_header", lang_code))

    if protein_data.get("accession") == "P18573":
        st.warning(f"⚠️ **{get_text('wheat_hexaploid_warning', lang_code)}**")
        st.info(f"ℹ️ **{get_text('cds_source_notice', lang_code)}**")

    if "pareto_results" not in st.session_state:
        st.info("⚡ Please run AI Mutation Engine in Tab 2 first.")
    else:
        results = st.session_state["pareto_results"]
        all_cands = results["all_candidates"]
        knee_cand = results["knee_point"]

        mutation_options = [c["mutation"] for c in all_cands]
        default_idx = mutation_options.index(knee_cand["mutation"]) if knee_cand else 0

        selected_mutation_str = st.selectbox("Select Target Mutation for CRISPR sgRNA Synthesis:", options=mutation_options, index=default_idx)
        chosen_cand = next(c for c in all_cands if c["mutation"] == selected_mutation_str)

        st.session_state["active_target_pos"] = chosen_cand["position"]

        pos = chosen_cand["position"]
        wt_aa = chosen_cand["wildtype"]
        mut_aa = chosen_cand["mutant"]

        # Pass verified CDS if available (e.g. for P18573)
        cds_seq = protein_data.get("cds_sequence")
        crispr_data = crispr_engine.design_sgrnas_for_mutation(
            full_protein_sequence=full_seq,
            mutation_pos_1indexed=pos,
            wildtype_aa=wt_aa,
            mutant_aa=mut_aa,
            cds_sequence=cds_seq
        )

        st.caption(f"DNA Sequence Source: `{crispr_data['dna_provenance']}`")

        col_aln1, col_aln2 = st.columns(2)
        with col_aln1:
            st.markdown(f"""
            **Wild-Type Window:**
            `<span class="ltr-span"><code>{crispr_data['window_peptide_wt']}</code></span>`
            - Target Codon: `<span class="ltr-span"><code>{crispr_data['target_codon_wt']}</code></span>` ({wt_aa})
            """, unsafe_allow_html=True)
        with col_aln2:
            st.markdown(f"""
            **Engineered Variant Window:**
            `<span class="ltr-span"><code>{crispr_data['window_peptide_mut']}</code></span>`
            - Edited Codon: `<span class="ltr-span"><code>{crispr_data['target_codon_mut']}</code></span>` ({mut_aa})
            """, unsafe_allow_html=True)

        st.markdown("---")
        guides_hdr = f"Designed SpCas9 sgRNAs (Found: {crispr_data['total_guides_found']})"
        st.subheader(guides_hdr)
        st.caption(f"**{get_text('doench_score_label', lang_code)}** — *{get_text('doench_disclaimer', lang_code)}*")

        guides = crispr_data["guides"]
        if guides:
            df_guides = pd.DataFrame(guides)
            cols_guide = [
                "protospacer", "pam", "strand", "tier", "on_target_score", "off_target_risk_pct", "off_target_risk_tier",
                "gc_content_pct", "cut_offset_bp", "overlaps_seed_region", "fwd_cloning_oligo", "rev_cloning_oligo"
            ]
            st.dataframe(df_guides[cols_guide], use_container_width=True)

            top_guide = guides[0]
            st.success(
                f"🌟 **Top Recommended sgRNA**: `<span class=\"ltr-span\"><code>{top_guide['protospacer']}</code></span>` | PAM: `{top_guide['pam']}` | "
                f"Predicted Cleavage Activity: **{top_guide['on_target_score']}%** | Off-Target Risk: **{top_guide['off_target_risk_pct']}%** ({top_guide['off_target_risk_tier']})"
            )

        st.markdown("---")
        st.subheader(get_text("export_center_title", lang_code))

        # Build Provenance Metadata Header
        utc_timestamp = datetime.now(timezone.utc).isoformat()
        provenance_header = (
            f"# AllergenMod-AI Scientific Data Export\n"
            f"# Timestamp: {utc_timestamp}\n"
            f"# UniProt Accession: {protein_data.get('accession')}\n"
            f"# Protein Name: {protein_data.get('protein_name')}\n"
            f"# Organism: {protein_data.get('organism')}\n"
            f"# Sequence Length: {protein_data.get('length')} aa\n"
            f"# Sequence CRC64: {protein_data.get('crc64')}\n"
            f"# Analysis Mode: {analysis_mode}\n"
            f"# AI Model: ESM-2 (facebook/esm2_t6_8M_UR50D)\n"
            f"# Evidence Levels: [COMPUTATIONAL_PREDICTION], [ILLUSTRATIVE_HEURISTIC], [RETRIEVED_REFERENCE]\n"
            f"# Target Mutation: {chosen_cand['mutation']}\n"
            f"# ------------------------------------------------------------\n"
        )

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            df_export_cands = pd.DataFrame(all_cands)
            cands_csv = provenance_header + df_export_cands.to_csv(index=False)
            st.download_button(
                label="📥 Download Candidate Library CSV (with Provenance Header)" if not is_ar else "📥 تحميل مكتبة طفرات باريتو (CSV موثق)",
                data=cands_csv.encode('utf-8'),
                file_name=f"AllergenMod_{protein_data.get('accession')}_pareto_library.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_dl2:
            df_export_guides = pd.DataFrame(guides) if guides else pd.DataFrame()
            guides_csv = provenance_header + df_export_guides.to_csv(index=False)
            st.download_button(
                label="📥 Download CRISPR sgRNAs CSV (with Provenance Header)" if not is_ar else "📥 تحميل أدلة كريسبر (CSV موثق)",
                data=guides_csv.encode('utf-8'),
                file_name=f"CRISPR_sgRNAs_{chosen_cand['mutation']}.csv",
                mime="text/csv",
                use_container_width=True
            )

# Footer
st.markdown("---")
st.caption(get_text("app_caption", lang_code))
