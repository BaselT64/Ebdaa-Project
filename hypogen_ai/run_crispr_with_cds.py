"""
run_crispr_with_cds.py
إعادة تصميم أدلة SpCas9 sgRNAs باستخدام تسلسل CDS البيولوجي الموثق (EMBL:X17361.1)
ومقارنتها بنتيجة الترجمة العكسية الافتراضية لطفرتي Q77S و Q79S.
"""

import os
import sys
import pandas as pd

# ضمان مسار المشروع
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.uniprot_fetcher import BENCHMARK_PROTEINS
from crispr.sgrna_engine import CRISPRsgRNAEngine

def main():
    engine = CRISPRsgRNAEngine()
    p = BENCHMARK_PROTEINS["P18573"]
    seq = p["sequence"]
    cds = p["cds_sequence"]

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hypogen_results")
    os.makedirs(out_dir, exist_ok=True)

    mutations = [
        {"name": "Q77S", "pos": 77, "wt": "Q", "mut": "S"},
        {"name": "Q79S", "pos": 79, "wt": "Q", "mut": "S"}
    ]

    fields = [
        "protospacer", "pam", "strand", "cut_offset_bp",
        "gc_content_pct", "on_target_score", "off_target_risk_pct", "off_target_risk_tier"
    ]

    for m in mutations:
        name = m["name"]
        pos = m["pos"]
        wt = m["wt"]
        mut = m["mut"]

        print("=" * 80)
        print(f"تصميم كريسبر للطفرة: {name} (الموضع {pos}: {wt} -> {mut})")
        print("=" * 80)

        # 1. التصميم باستخدام تسلسل CDS الموثق
        res_cds = engine.design_sgrnas_for_mutation(
            full_protein_sequence=seq,
            mutation_pos_1indexed=pos,
            wildtype_aa=wt,
            mutant_aa=mut,
            cds_sequence=cds
        )

        print("[أولاً: التصميم باستخدام تسلسل CDS الموثق (EMBL:X17361.1)]")
        print(f"  has_verified_cds     : {res_cds['has_verified_cds']}")
        print(f"  dna_provenance       : {res_cds['dna_provenance']}")
        print(f"  total_guides_found   : {res_cds['total_guides_found']}")
        print(f"  target_codon_wt      : {res_cds['target_codon_wt']}")
        print(f"  target_codon_mut     : {res_cds['target_codon_mut']}")

        guides_cds = res_cds["guides"]
        print(f"\n  أفضل 5 أدلة SpCas9 (بناءً على درجة كفاءة القطع على الهدف):")
        top_5 = guides_cds[:5]
        for i, g in enumerate(top_5, 1):
            print(f"    {i}. protospacer: {g['protospacer']} | pam: {g['pam']} | strand: {g['strand']:14s} | "
                  f"cut_offset_bp: {g['cut_offset_bp']:4.1f} | gc_content_pct: {g['gc_content_pct']}% | "
                  f"on_target_score: {g['on_target_score']:4.1f}% | "
                  f"off_target_risk_pct: {g['off_target_risk_pct']:4.1f}% | off_target_risk_tier: {g['off_target_risk_tier']}")

        # حفظ كامل الجدول في hypogen_results
        csv_filename = f"7_sgrna_cds_{name}.csv"
        csv_path = os.path.join(out_dir, csv_filename)
        df_cds = pd.DataFrame(guides_cds)
        df_cds.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\n  [حفظ] تم حفظ جدول الأدلة الكامل ({len(guides_cds)} دليلاً) في: {csv_path}")

        # 2. المقارنة مع التصميم بدون CDS (الترجمة العكسية الافتراضية)
        res_nocds = engine.design_sgrnas_for_mutation(
            full_protein_sequence=seq,
            mutation_pos_1indexed=pos,
            wildtype_aa=wt,
            mutant_aa=mut,
            cds_sequence=None
        )

        print(f"\n[ثانياً: المقارنة مع التصميم الافتراضي بدون CDS (In-Silico Reverse Translation)]")
        print(f"  has_verified_cds     : {res_nocds['has_verified_cds']}")
        print(f"  dna_provenance       : {res_nocds['dna_provenance']}")
        print(f"  total_guides_found   : {res_nocds['total_guides_found']}")
        print(f"  target_codon_wt      : {res_nocds['target_codon_wt']}")
        print(f"  target_codon_mut     : {res_nocds['target_codon_mut']}")

        guides_nocds = res_nocds["guides"]
        if guides_nocds:
            top_nocds = guides_nocds[0]
            print(f"  أفضل دليل بدون CDS  : {top_nocds['protospacer']} | pam: {top_nocds['pam']} | "
                  f"strand: {top_nocds['strand']} | on_target_score: {top_nocds['on_target_score']}% | "
                  f"off_target_risk_pct: {top_nocds['off_target_risk_pct']}%")
        else:
            print("  لم يُعثر على أدلة بدون CDS.")

        print("\n")

if __name__ == "__main__":
    main()
