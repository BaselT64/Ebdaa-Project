"""
run_hypogen_analysis.py
يشغّل نفس مكوّنات مشروعك (ESM-2 + المؤشرات الثلاثة + باريتو + كريسبر) ويصدّر النتائج الحقيقية في مجلد hypogen_results.

طريقة التشغيل (من داخل مجلد المشروع الذي فيه app.py):
    python run_hypogen_analysis.py

لا يعدّل هذا الملف أي ملف من ملفات مشروعك.
"""
import os
import sys
import json
import time
import argparse
import platform

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.uniprot_fetcher import UniProtFetcher, BENCHMARK_PROTEINS
from core.epitope_scanner import EpitopeScanner
from models.esm_mutator import ESMMutator
from scoring.immunogenicity import ImmunogenicityScorer
from scoring.food_functionality import FoodFunctionalityScorer
from scoring.agronomic_fitness import AgronomicFitnessScorer
from scoring.pareto_optimizer import ParetoOptimizer
from crispr.sgrna_engine import CRISPRsgRNAEngine

ACCESSION = "P18573"
OUT_DIR = "hypogen_results"
DEFAULT_W = (0.40, 0.35, 0.25)  # القيم الافتراضية في app.py
WEIGHT_SETS = {
    "default_0.40_0.35_0.25": (0.40, 0.35, 0.25),
    "equal_1:1:1": (1 / 3, 1 / 3, 1 / 3),
    "safety_first_0.50_0.25_0.25": (0.50, 0.25, 0.25),
    "food_agro_0.20_0.40_0.40": (0.20, 0.40, 0.40),
    "safety_heavy_0.60_0.20_0.20": (0.60, 0.20, 0.20),
}

ap = argparse.ArgumentParser()
ap.add_argument("--allow-fallback", action="store_true",
                help="للاختبار فقط: يسمح بالتشغيل بدون ESM-2 (لا تستخدم نتائجه في الاستمارة)")
args = ap.parse_args()

os.makedirs(OUT_DIR, exist_ok=True)
log_lines = []


def log(msg=""):
    print(msg)
    log_lines.append(str(msg))


# ---------------------------------------------------------------- 1) التسلسل
cache = BENCHMARK_PROTEINS[ACCESSION]
fetcher = UniProtFetcher()
seq_source = "cache"
try:
    live = fetcher._query_rest_api(ACCESSION)
    if live and live.get("sequence"):
        protein = live
        seq_source = "live UniProt"
    else:
        protein = cache
except Exception as e:  # بدون إنترنت
    protein = cache
    log(f"[تنبيه] تعذر الجلب الحي من UniProt ({e}); استُخدم التسلسل المخزن في الكود.")

seq = protein["sequence"]
features = protein.get("features", [])
log(f"مصدر التسلسل: {seq_source} | الطول: {len(seq)} | يطابق المخزن في الكود: {seq == cache['sequence']}")

scanner = EpitopeScanner()
scan = scanner.scan_sequence(seq, features)
epitopes = scan["epitopes"]
motif33 = "LQLQPFPQPQLPYPQPQLPYPQPQLPYPQPQPF"
i33 = seq.find(motif33)
if i33 >= 0:
    s33, e33 = i33 + 1, i33 + len(motif33)
    log(f"ببتيد 33-mer موجود في المواضع {s33} إلى {e33}")
    scan_positions = list(range(s33, e33 + 1))
else:
    log("[تنبيه] ببتيد 33-mer غير موجود في هذا التسلسل؛ سيُمسح كل ما يغطيه الحواتم المكتشفة.")
    scan_positions = sorted({p for e in epitopes for p in range(e["start"], e["end"] + 1)})
log(f"عدد الحواتم المكتشفة: {len(epitopes)}")

# ---------------------------------------------------------------- 2) نموذج ESM-2
t0 = time.time()
esm = ESMMutator()
info = esm.get_model_info()
log(f"حالة المحرك: {info['status']} | {info['checkpoint']} | الجهاز: {info['device']} | المعاملات: {info['parameter_count']}")
if not esm.is_model_ready and not args.allow_fallback:
    log("\n[توقف] لم يُحمَّل نموذج ESM-2 (يحتاج اتصالاً بالإنترنت أول مرة لتنزيل facebook/esm2_t6_8M_UR50D).")
    log("اتصل بالإنترنت ثم أعد التشغيل. لم تُكتب نتائج لأن المحرك الاحتياطي ليس ESM-2.")
    with open(os.path.join(OUT_DIR, "run_info.txt"), "w", encoding="utf8") as f:
        f.write("\n".join(log_lines))
    sys.exit(1)

im, ff, af = ImmunogenicityScorer(), FoodFunctionalityScorer(), AgronomicFitnessScorer()
pareto = ParetoOptimizer()
crispr = CRISPRsgRNAEngine()


def evaluate(cand):
    p, a = cand["position"], cand["mutant"]
    mut_seq = seq[: p - 1] + a + seq[p:]
    rec = dict(cand)
    rec.update(im.evaluate_mutation_safety(seq, mut_seq, p, epitopes))
    rec.update(ff.evaluate_food_functionality(seq, mut_seq, p, features))
    rec.update(af.evaluate_agronomic_fitness(seq, mut_seq, p, features))
    return rec


COLS = ["mutation", "position", "llr", "marginal_prob", "fitness_score",
        "medical_safety_index", "binding_affinity_reduction_pct",
        "food_functionality_index", "blosum62_raw", "disulfide_disrupted",
        "agronomic_fitness_index", "nitrogen_atoms_wt", "nitrogen_atoms_mut",
        "composite_score", "distance_to_utopia", "is_pareto"]

# ---------------------------------------------------------------- 3) مسح كامل: كل البدائل في كل موضع من 33-mer
all_cands = []
for pos in scan_positions:
    for m in esm.compute_masked_marginals(seq, pos):
        if not m["is_wildtype"]:
            all_cands.append(evaluate(m))
full = pareto.compute_pareto_front(all_cands, *DEFAULT_W)
df_full = pd.DataFrame(full["all_candidates"])
df_full[[c for c in COLS if c in df_full.columns]].to_csv(
    os.path.join(OUT_DIR, "1_full_scan_33mer.csv"), index=False, encoding="utf-8-sig")
log(f"\n[مسح كامل] {full['total_count']} بديلاً | على جبهة باريتو: {full['pareto_count']}")
log(f"[مسح كامل] نقطة الركبة: {full['knee_point']['mutation']}")

# ---------------------------------------------------------------- 4) تشغيل مطابق للتطبيق (أول 5 مواضع مقترحة، Top-K = 4)
suggested = sorted({p for e in epitopes for p in range(e["start"], e["end"] + 1)
                    if seq[p - 1] in ["Q", "P", "R", "K", "W", "E", "D"]})
app_positions = suggested[:5]
lib = esm.generate_candidate_library(seq, app_positions, top_k_per_position=4)
app_cands = [evaluate(c) for c in lib]
app = pareto.compute_pareto_front(app_cands, *DEFAULT_W)
df_app = pd.DataFrame(app["all_candidates"])
df_app[[c for c in COLS if c in df_app.columns]].to_csv(
    os.path.join(OUT_DIR, "2_app_default_run.csv"), index=False, encoding="utf-8-sig")
knee = app["knee_point"]
log(f"\n[تشغيل التطبيق الافتراضي] المواضع: {app_positions} | مرشحون: {app['total_count']} | جبهة باريتو: {app['pareto_count']}")
log(f"[تشغيل التطبيق الافتراضي] نقطة الركبة: {knee['mutation']} | أمان {knee['medical_safety_index']} | غذاء {knee['food_functionality_index']} | زراعة {knee['agronomic_fitness_index']} | LLR {knee['llr']}")

# ---------------------------------------------------------------- 5) كل بدائل الموضع المختار (جدول الترتيب)
sel_pos = knee["position"]
sel_mut = knee["mutation"]
rows_sel = [c for c in full["all_candidates"] if c["position"] == sel_pos]
pd.DataFrame(rows_sel)[[c for c in COLS if c in rows_sel[0]]].sort_values(
    "composite_score", ascending=False).to_csv(
    os.path.join(OUT_DIR, "3_selected_position_all_19.csv"), index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- 6) حساسية الأوزان
sens = []
for name, w in WEIGHT_SETS.items():
    out = pareto.compute_pareto_front(all_cands, *w)
    ranked = [c["mutation"] for c in out["all_candidates"]]
    sens.append({
        "weights": name,
        "top1_by_composite": ranked[0],
        "rank_of_selected": ranked.index(sel_mut) + 1 if sel_mut in ranked else None,
        "selected_on_pareto_front": any(c["mutation"] == sel_mut for c in out["pareto_front"]),
        "knee_point": out["knee_point"]["mutation"],
    })
pd.DataFrame(sens).to_csv(os.path.join(OUT_DIR, "4_weight_sensitivity.csv"), index=False, encoding="utf-8-sig")
log(f"\n[حساسية الأوزان] المختارة: {sel_mut}")
for r in sens:
    log(f"  {r['weights']}: الأولى = {r['top1_by_composite']} | ترتيب المختارة = {r['rank_of_selected']} | على الجبهة = {r['selected_on_pareto_front']}")

# ---------------------------------------------------------------- 7) خطوط الأساس
idx = ["medical_safety_index", "food_functionality_index", "agronomic_fitness_index"]
dfa = pd.DataFrame(all_cands)
base_rows = [{
    "baseline": "الطفرة المختارة " + sel_mut,
    **{k: knee[k] for k in idx},
}]
base_rows.append({"baseline": "متوسط كل البدائل في 33-mer (طفرة عشوائية)", **{k: round(dfa[k].mean(), 2) for k in idx}})
base_rows.append({"baseline": "الانحراف المعياري لكل البدائل", **{k: round(dfa[k].std(), 2) for k in idx}})
at_pos = dfa[dfa["position"] == sel_pos]
b62 = at_pos.sort_values("blosum62_raw", ascending=False).iloc[0]
base_rows.append({"baseline": f"أعلى BLOSUM62 فقط في الموضع نفسه ({b62['mutation']})", **{k: b62[k] for k in idx}})
llr_top = at_pos.sort_values("llr", ascending=False).iloc[0]
base_rows.append({"baseline": f"أعلى LLR من ESM-2 فقط في الموضع نفسه ({llr_top['mutation']})", **{k: llr_top[k] for k in idx}})
pd.DataFrame(base_rows).to_csv(os.path.join(OUT_DIR, "5_baselines.csv"), index=False, encoding="utf-8-sig")
log("\n[خطوط الأساس]")
for r in base_rows:
    log(f"  {r['baseline']}: أمان {r[idx[0]]} | غذاء {r[idx[1]]} | زراعة {r[idx[2]]}")

# ---------------------------------------------------------------- 8) كريسبر للطفرة المختارة
cr = crispr.design_sgrnas_for_mutation(seq, sel_pos, knee["wildtype"], knee["mutant"])
pd.DataFrame(cr["guides"]).to_csv(os.path.join(OUT_DIR, "6_sgrna_guides.csv"), index=False, encoding="utf-8-sig")
if cr["guides"]:
    g = cr["guides"][0]
    log(f"\n[كريسبر] {cr['locus_mutation']} | عدد الأدلة: {cr['total_guides_found']} | الأفضل: {g['protospacer']} {g['pam']} | الدرجة {g['on_target_score']} | الخطر {g['off_target_risk_pct']}%")

# ---------------------------------------------------------------- 9) معلومات التشغيل
run_info = {
    "python": platform.python_version(),
    "engine": info,
    "sequence_source": seq_source,
    "sequence_length": len(seq),
    "sequence_equals_code_cache": seq == cache["sequence"],
    "scan_positions": [scan_positions[0], scan_positions[-1]],
    "selected_mutation": sel_mut,
    "seconds": round(time.time() - t0, 1),
}
with open(os.path.join(OUT_DIR, "run_info.json"), "w", encoding="utf8") as f:
    json.dump(run_info, f, ensure_ascii=False, indent=2)
with open(os.path.join(OUT_DIR, "run_info.txt"), "w", encoding="utf8") as f:
    f.write("\n".join(log_lines))
log(f"\nتم. الملفات في المجلد: {OUT_DIR}")
