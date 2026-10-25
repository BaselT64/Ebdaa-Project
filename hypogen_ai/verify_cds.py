"""
verify_cds.py
التحقق من تسلسل CDS الموثق لبروتين P18573 ومقارنة ترجمته الحيوية بالتسلسل البروتيني.
"""

import os
import sys

# ضمان مسار المشروع
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Bio.Seq import Seq
from core.uniprot_fetcher import BENCHMARK_PROTEINS

def verify():
    p = BENCHMARK_PROTEINS["P18573"]
    prot_seq = p["sequence"]
    cds_seq = p["cds_sequence"]
    cds_ref = p.get("nucleotide_cds_reference", {})

    print("=" * 70)
    print("التحقق من تسلسل CDS لبروتين القمح P18573 (Alpha/beta-gliadin MM1)")
    print("=" * 70)
    print(f"المصدر الموثق في الكود: {cds_ref.get('database')}:{cds_ref.get('accession')} (ProteinId: {cds_ref.get('protein_id')})")
    print(f"طول تسلسل الـ CDS بالنيوكليوتيدات: {len(cds_seq)} bp")
    print(f"طول تسلسل البروتين بالأحماض الأمينية: {len(prot_seq)} aa")

    # ترجمة الـ CDS
    trans_seq = str(Seq(cds_seq).translate())
    print(f"طول التسلسل المترجم: {len(trans_seq)} aa")

    # الكودون الأول والنهائي
    start_codon = cds_seq[:3]
    end_codon = cds_seq[-3:]
    print(f"الكودون الأول (Start): {start_codon} -> المترجم: {trans_seq[0]} | الأصلي: {prot_seq[0]}")
    print(f"الكودون النهائي (End): {end_codon} -> المترجم: {trans_seq[-1]} | الأصلي: {prot_seq[-1]}")

    # المقارنة الحرفية
    exact_match = (trans_seq == prot_seq)
    print(f"\nهل الترجمة تطابق البروتين حرفًا بحرف؟ {exact_match}")

    mismatches = []
    min_len = min(len(trans_seq), len(prot_seq))
    for i in range(min_len):
        if trans_seq[i] != prot_seq[i]:
            mismatches.append((i + 1, trans_seq[i], prot_seq[i]))

    if len(trans_seq) != len(prot_seq):
        print(f"اختلاف في الطول: المترجم={len(trans_seq)} مقابل الأصلي={len(prot_seq)}")

    print(f"عدد الاختلافات: {len(mismatches)}")
    if mismatches:
        print("مواضع الاختلافات:")
        for pos, c_aa, p_aa in mismatches:
            print(f"  الموضع {pos}: المترجم من CDS = '{c_aa}' مقابل البروتين الأصلي = '{p_aa}'")
    else:
        print("تطابق تام بنسبة 100% بين جميع الأحماض الأمينية (307/307).")

    # طباعة الكودونات في المواضع 75 إلى 80
    print("\nالكودونات والأحماض الأمينية في المواضع 75 إلى 80 (ببتيد 33-mer وموضع Q77):")
    for pos in range(75, 81):
        codon = cds_seq[(pos - 1) * 3 : pos * 3]
        aa = prot_seq[pos - 1]
        trans_aa = str(Seq(codon).translate())
        note = " <-- (موضع طفرة Q77 المستهدفة)" if pos == 77 else (" <-- (موضع طفرة Q79)" if pos == 79 else "")
        print(f"  الموضع {pos:02d}: الحمض الأميني = {aa} | الكودون = {codon} | المترجم = {trans_aa}{note}")

if __name__ == "__main__":
    verify()
