"""
scoring/immunogenicity.py: HLA-DQ2/DQ8 & IgE peptide binding affinity reduction metrics.
Evidence Level: [ILLUSTRATIVE_HEURISTIC] Sequence-based binding affinity reduction proxy.
Separates Celiac Disease (HLA-DQ2.5/DQ8 T-cell enteropathy via TG2 deamidation) from IgE allergies.
"""

from typing import Dict, Any, List, Optional
import numpy as np

# HLA-DQ2.5 (DQA1*05:01/DQB1*02:01) Position-Weight Matrix for 9-mer core binding grooves
# Reference: Sollid LM, et al. (2012) Immunogenetics 64:455-460.
HLA_DQ2_ANCHOR_WEIGHTS = {
    1: {"F": 2.2, "Y": 2.0, "W": 1.8, "L": 1.5, "I": 1.4, "M": 1.2, "V": 0.8, "A": 0.2, "Q": 0.0, "P": -1.5},
    4: {"E": 3.0, "D": 1.8, "Q": 1.2, "N": 0.3, "A": -1.0, "S": -0.8, "T": -0.7, "R": -2.5, "K": -2.5, "P": -1.8},
    6: {"E": 2.5, "D": 2.0, "Q": 1.4, "N": 0.8, "S": 0.5, "T": 0.5, "A": -0.5, "P": -1.2, "R": -2.0, "K": -2.0},
    7: {"E": 2.0, "L": 1.8, "F": 1.6, "Y": 1.5, "Q": 1.0, "I": 1.2, "P": 0.4, "A": 0.0},
    9: {"L": 2.2, "F": 2.0, "Y": 1.8, "I": 1.6, "E": 1.5, "V": 1.0, "A": 0.2, "P": -2.0}
}

# IgE paratope binding contribution for surface residues in linear allergens (Calis et al. / IEDB)
IGE_CONTACT_PROBABILITY: Dict[str, float] = {
    "R": 2.4, "K": 2.2, "W": 2.6, "Y": 2.1, "F": 2.0, "H": 1.8,
    "E": 1.6, "D": 1.4, "Q": 1.2, "N": 1.0, "M": 0.9, "L": 0.8,
    "I": 0.8, "V": 0.6, "P": 1.1, "T": 0.5, "S": 0.4, "A": 0.2, "G": 0.1, "C": 0.2
}


class ImmunogenicityScorer:
    """
    Evaluates allergenic binding affinity reduction across HLA-DQ2/8 and IgE epitopes.
    Labeled as: [ILLUSTRATIVE_HEURISTIC] Sequence-based binding affinity reduction proxy.
    """

    def __init__(self):
        self.evidence_level = "[ILLUSTRATIVE_HEURISTIC]"

    def evaluate_mutation_safety(
        self,
        wt_sequence: str,
        mut_sequence: str,
        mutation_pos_1indexed: int,
        detected_epitopes: List[Dict[str, Any]],
        analysis_mode: str = "FOOD_ALLERGEN"
    ) -> Dict[str, Any]:
        """
        Calculates the Medical Safety Index (0.0 to 100.0%) for a proposed point mutation.
        In GENERAL_PROTEIN mode, allergen safety is marked as non-applicable.
        """
        if analysis_mode == "GENERAL_PROTEIN":
            return {
                "medical_safety_index": 50.0,
                "binding_affinity_reduction_pct": 0.0,
                "hla_anchor_disrupted": False,
                "ige_epitope_disrupted": False,
                "is_applicable": False,
                "evidence_level": self.evidence_level,
                "mechanism": "Non-food protein: Allergen safety index not applicable (Neutral baseline 50.0%).",
                "wt_local_affinity": 1.0,
                "mut_local_affinity": 1.0,
                "overlapping_epitopes_count": 0
            }

        wt_seq = wt_sequence.upper()
        mut_seq = mut_sequence.upper()
        pos = mutation_pos_1indexed
        wt_aa = wt_seq[pos - 1]
        mut_aa = mut_seq[pos - 1]

        overlapping_epitopes = [
            ep for ep in detected_epitopes
            if ep["start"] <= pos <= ep["end"]
        ]

        if not overlapping_epitopes:
            return {
                "medical_safety_index": 50.0,
                "binding_affinity_reduction_pct": 0.0,
                "hla_anchor_disrupted": False,
                "ige_epitope_disrupted": False,
                "is_applicable": True,
                "evidence_level": self.evidence_level,
                "mechanism": "Mutation in non-epitope scaffold (Neutral Safety Profile).",
                "wt_local_affinity": 1.0,
                "mut_local_affinity": 1.0,
                "overlapping_epitopes_count": 0
            }

        is_celiac_locus = any("CELIAC" in ep["id"] or "DQ2" in ep["id"] or "DQ8" in ep["id"] for ep in overlapping_epitopes)
        hla_disrupted = False
        hla_score_wt = 0.0
        hla_score_mut = 0.0

        if is_celiac_locus:
            if wt_aa == "Q":
                hla_disrupted = True
                hla_score_wt += 4.5
                if mut_aa in ["A", "S", "T", "N", "V", "L"]:
                    hla_score_mut += 0.5
                elif mut_aa in ["R", "K"]:
                    hla_score_mut += 0.2
                else:
                    hla_score_mut += 1.0
            elif wt_aa == "P":
                hla_disrupted = True
                hla_score_wt += 3.5
                hla_score_mut += 0.8
            else:
                hla_score_wt += 2.0
                hla_score_mut += 1.2
        else:
            hla_score_wt = 1.0
            hla_score_mut = 1.0

        ige_disrupted = False
        ige_wt_contact = IGE_CONTACT_PROBABILITY.get(wt_aa, 0.5)
        ige_mut_contact = IGE_CONTACT_PROBABILITY.get(mut_aa, 0.5)

        win_start = max(0, pos - 4)
        win_end = min(len(wt_seq), pos + 3)
        wt_window = wt_seq[win_start:win_end]
        mut_window = mut_seq[win_start:win_end]

        local_ige_wt = sum(IGE_CONTACT_PROBABILITY.get(aa, 0.5) for aa in wt_window)
        local_ige_mut = sum(IGE_CONTACT_PROBABILITY.get(aa, 0.5) for aa in mut_window)

        if ige_mut_contact < ige_wt_contact:
            ige_disrupted = True

        total_affinity_wt = hla_score_wt + local_ige_wt
        total_affinity_mut = hla_score_mut + local_ige_mut

        reduction_ratio = max(0.0, (total_affinity_wt - total_affinity_mut) / max(0.1, total_affinity_wt))
        affinity_reduction_pct = round(reduction_ratio * 100.0, 2)

        base_safety = 50.0 + (reduction_ratio * 45.0)

        if hla_disrupted:
            base_safety += 5.0
        if ige_disrupted and (wt_aa in ["R", "K", "W", "Y"]):
            base_safety += 4.0

        if total_affinity_mut > total_affinity_wt:
            base_safety = max(5.0, 50.0 - ((total_affinity_mut - total_affinity_wt) / total_affinity_wt) * 40.0)

        medical_safety_index = min(100.0, max(0.0, round(base_safety, 2)))

        mechanisms = []
        if hla_disrupted and wt_aa == "Q":
            mechanisms.append(f"Abolished tTG2 deamidation substrate anchor (Q{pos} -> {mut_aa}).")
        if hla_disrupted and wt_aa == "P":
            mechanisms.append(f"Perturbed polyproline II backbone conformation for HLA-DQ2 groove (P{pos} -> {mut_aa}).")
        if ige_disrupted:
            mechanisms.append(f"Depleted high-affinity IgE contact paratope ({wt_aa}{pos} -> {mut_aa}).")
        if not mechanisms:
            mechanisms.append("Neutral substitution within epitope boundary.")

        return {
            "medical_safety_index": medical_safety_index,
            "binding_affinity_reduction_pct": affinity_reduction_pct,
            "hla_anchor_disrupted": hla_disrupted,
            "ige_epitope_disrupted": ige_disrupted,
            "is_applicable": True,
            "evidence_level": self.evidence_level,
            "mechanism": " ".join(mechanisms),
            "wt_local_affinity": round(total_affinity_wt, 3),
            "mut_local_affinity": round(total_affinity_mut, 3),
            "overlapping_epitopes_count": len(overlapping_epitopes)
        }
