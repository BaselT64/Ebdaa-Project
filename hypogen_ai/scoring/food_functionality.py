"""
scoring/food_functionality.py: BLOSUM-approximated stability & rheological gluten index.
Evidence Level: [ILLUSTRATIVE_HEURISTIC] Sequence-based structural stability & rheology proxy.
Mandate: Loss of conserved cysteine residues required for disulfide bridging incurs a -50% penalty.
Mode-aware: Disables wheat dough rheological claims for non-crop / general proteins.
"""

from typing import Dict, Any, List, Optional
import numpy as np

# Standard BLOSUM62 Substitution Matrix
BLOSUM62_MATRIX: Dict[str, Dict[str, int]] = {
    'A': {'A':  4, 'R': -1, 'N': -2, 'D': -2, 'C':  0, 'Q': -1, 'E': -1, 'G':  0, 'H': -2, 'I': -1, 'L': -1, 'K': -1, 'M': -1, 'F': -2, 'P': -1, 'S':  1, 'T':  0, 'W': -3, 'Y': -2, 'V':  0},
    'R': {'A': -1, 'R':  5, 'N':  0, 'D': -2, 'C': -3, 'Q':  1, 'E':  0, 'G': -2, 'H':  0, 'I': -3, 'L': -2, 'K':  2, 'M': -1, 'F': -3, 'P': -2, 'S': -1, 'T': -1, 'W': -3, 'Y': -2, 'V': -3},
    'N': {'A': -2, 'R':  0, 'N':  6, 'D':  1, 'C': -3, 'Q':  0, 'E':  0, 'G':  0, 'H':  1, 'I': -3, 'L': -3, 'K':  0, 'M': -2, 'F': -3, 'P': -2, 'S':  1, 'T':  0, 'W': -4, 'Y': -2, 'V': -3},
    'D': {'A': -2, 'R': -2, 'N':  1, 'D':  6, 'C': -3, 'Q':  0, 'E':  2, 'G': -1, 'H': -1, 'I': -3, 'L': -4, 'K': -1, 'M': -3, 'F': -3, 'P': -1, 'S':  0, 'T': -1, 'W': -4, 'Y': -3, 'V': -3},
    'C': {'A':  0, 'R': -3, 'N': -3, 'D': -3, 'C':  9, 'Q': -3, 'E': -4, 'G': -3, 'H': -3, 'I': -1, 'L': -1, 'K': -3, 'M': -1, 'F': -2, 'P': -3, 'S': -1, 'T': -1, 'W': -2, 'Y': -2, 'V': -1},
    'Q': {'A': -1, 'R':  1, 'N':  0, 'D':  0, 'C': -3, 'Q':  5, 'E':  2, 'G': -2, 'H':  0, 'I': -3, 'L': -2, 'K':  1, 'M':  0, 'F': -3, 'P': -1, 'S':  0, 'T': -1, 'W': -2, 'Y': -1, 'V': -2},
    'E': {'A': -1, 'R':  0, 'N':  0, 'D':  2, 'C': -4, 'Q':  2, 'E':  5, 'G': -2, 'H':  0, 'I': -3, 'L': -3, 'K':  1, 'M': -2, 'F': -3, 'P': -1, 'S':  0, 'T': -1, 'W': -3, 'Y': -2, 'V': -2},
    'G': {'A':  0, 'R': -2, 'N':  0, 'D': -1, 'C': -3, 'Q': -2, 'E': -2, 'G':  6, 'H': -2, 'I': -4, 'L': -4, 'K': -2, 'M': -3, 'F': -3, 'P': -2, 'S':  0, 'T': -2, 'W': -2, 'Y': -3, 'V': -3},
    'H': {'A': -2, 'R':  0, 'N':  1, 'D': -1, 'C': -3, 'Q':  0, 'E':  0, 'G': -2, 'H':  8, 'I': -3, 'L': -3, 'K': -1, 'M': -2, 'F': -1, 'P': -2, 'S': -1, 'T': -2, 'W': -2, 'Y':  2, 'V': -3},
    'I': {'A': -1, 'R': -3, 'N': -3, 'D': -3, 'C': -1, 'Q': -3, 'E': -3, 'G': -4, 'H': -3, 'I':  4, 'L':  2, 'K': -3, 'M':  1, 'F':  0, 'P': -3, 'S': -2, 'T': -1, 'W': -3, 'Y': -1, 'V':  3},
    'L': {'A': -1, 'R': -2, 'N': -3, 'D': -4, 'C': -1, 'Q': -2, 'E': -3, 'G': -4, 'H': -3, 'I':  2, 'L':  4, 'K': -2, 'M':  2, 'F':  0, 'P': -3, 'S': -2, 'T': -1, 'W': -2, 'Y': -1, 'V':  1},
    'K': {'A': -1, 'R':  2, 'N':  0, 'D': -1, 'C': -3, 'Q':  1, 'E':  1, 'G': -2, 'H': -1, 'I': -3, 'L': -2, 'K':  5, 'M': -1, 'F': -3, 'P': -1, 'S':  0, 'T': -1, 'W': -3, 'Y': -2, 'V': -2},
    'M': {'A': -1, 'R': -1, 'N': -2, 'D': -3, 'C': -1, 'Q':  0, 'E': -2, 'G': -3, 'H': -2, 'I':  1, 'L':  2, 'K': -1, 'M':  5, 'F':  0, 'P': -2, 'S': -1, 'T': -1, 'W': -1, 'Y': -1, 'V':  1},
    'F': {'A': -2, 'R': -3, 'N': -3, 'D': -3, 'C': -2, 'Q': -3, 'E': -3, 'G': -3, 'H': -1, 'I':  0, 'L':  0, 'K': -3, 'M':  0, 'F':  6, 'P': -4, 'S': -2, 'T': -2, 'W':  1, 'Y':  3, 'V': -1},
    'P': {'A': -1, 'R': -2, 'N': -2, 'D': -1, 'C': -3, 'Q': -1, 'E': -1, 'G': -2, 'H': -2, 'I': -3, 'L': -3, 'K': -1, 'M': -2, 'F': -4, 'P':  7, 'S': -1, 'T': -1, 'W': -4, 'Y': -3, 'V': -2},
    'S': {'A':  1, 'R': -1, 'N':  1, 'D':  0, 'C': -1, 'Q':  0, 'E':  0, 'G':  0, 'H': -1, 'I': -2, 'L': -2, 'K':  0, 'M': -1, 'F': -2, 'P': -1, 'S':  4, 'T':  1, 'W': -3, 'Y': -2, 'V': -2},
    'T': {'A':  0, 'R': -1, 'N':  0, 'D': -1, 'C': -1, 'Q': -1, 'E': -1, 'G': -2, 'H': -2, 'I': -1, 'L': -1, 'K': -1, 'M': -1, 'F': -2, 'P': -1, 'S':  1, 'T':  5, 'W': -2, 'Y': -2, 'V':  0},
    'W': {'A': -3, 'R': -3, 'N': -4, 'D': -4, 'C': -2, 'Q': -2, 'E': -3, 'G': -2, 'H': -2, 'I': -3, 'L': -2, 'K': -3, 'M': -1, 'F':  1, 'P': -4, 'S': -3, 'T': -2, 'W': 11, 'Y':  2, 'V': -3},
    'Y': {'A': -2, 'R': -2, 'N': -2, 'D': -3, 'C': -2, 'Q': -1, 'E': -2, 'G': -3, 'H':  2, 'I': -1, 'L': -1, 'K': -2, 'M': -1, 'F':  3, 'P': -3, 'S': -2, 'T': -2, 'W':  2, 'Y':  7, 'V': -1},
    'V': {'A':  0, 'R': -3, 'N': -3, 'D': -3, 'C': -1, 'Q': -2, 'E': -2, 'G': -3, 'H': -3, 'I':  3, 'L':  1, 'K': -2, 'M':  1, 'F': -1, 'P': -2, 'S': -2, 'T':  0, 'W': -3, 'Y': -1, 'V':  4}
}

KYTE_DOOLITTLE = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5,
    'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9,
    'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9,
    'Y': -1.3, 'V': 4.2
}


class FoodFunctionalityScorer:
    """Evaluates stability, disulfide structural integrity, and dough rheological capacity."""

    def __init__(self):
        self.evidence_level = "[ILLUSTRATIVE_HEURISTIC]"

    def evaluate_food_functionality(
        self,
        wt_sequence: str,
        mut_sequence: str,
        mutation_pos_1indexed: int,
        curated_features: Optional[List[Dict[str, Any]]] = None,
        analysis_mode: str = "FOOD_ALLERGEN"
    ) -> Dict[str, Any]:
        """Calculates Food Functionality Index (0.0 to 100.0%). Disables dough metrics for general proteins."""
        wt_seq = wt_sequence.upper()
        mut_seq = mut_sequence.upper()
        pos = mutation_pos_1indexed
        wt_aa = wt_seq[pos - 1]
        mut_aa = mut_seq[pos - 1]

        # 1. Disulfide Bond Integrity
        is_disulfide_cys = False
        disulfide_penalty = 0.0

        if wt_aa == "C":
            if curated_features:
                is_disulfide_cys = any(
                    f.get("type") == "Disulfide bond" and (f.get("start") == pos or f.get("end") == pos)
                    for f in curated_features
                )
            else:
                is_disulfide_cys = True

            if mut_aa != "C":
                disulfide_penalty = 50.0

        disulfide_score = max(0.0, 100.0 - disulfide_penalty)

        # 2. BLOSUM62 Conservation Score
        blosum_val = BLOSUM62_MATRIX.get(wt_aa, {}).get(mut_aa, -2)
        if wt_aa == mut_aa:
            blosum_score = 100.0
        else:
            blosum_score = min(100.0, max(10.0, ((blosum_val + 4.0) / 7.5) * 100.0))

        # 3. Rheological Gluten Index (Wheat / Food Allergen Mode Only)
        hydro_wt = KYTE_DOOLITTLE.get(wt_aa, 0.0)
        hydro_mut = KYTE_DOOLITTLE.get(mut_aa, 0.0)
        hydro_shift = abs(hydro_wt - hydro_mut)

        if analysis_mode == "FOOD_ALLERGEN":
            rheo_retention = 90.0
            if wt_aa == "P":
                if mut_aa in ["A", "S", "T"]:
                    rheo_retention = 82.0
                else:
                    rheo_retention = 68.0
            elif wt_aa == "Q":
                if mut_aa in ["N", "S", "T"]:
                    rheo_retention = 88.0
                elif mut_aa in ["A"]:
                    rheo_retention = 80.0
                else:
                    rheo_retention = 65.0

            if hydro_shift > 4.0:
                rheo_retention -= 15.0

            rheo_score = min(100.0, max(10.0, rheo_retention))
            
            if disulfide_penalty > 0.0:
                raw_functionality = (0.50 * blosum_score + 0.50 * rheo_score) * 0.50
            else:
                raw_functionality = (0.35 * disulfide_score + 0.35 * blosum_score + 0.30 * rheo_score)
        else:
            # GENERAL_PROTEIN mode: structural integrity proxy without dough rheology claims
            rheo_score = None
            if disulfide_penalty > 0.0:
                raw_functionality = blosum_score * 0.50
            else:
                raw_functionality = (0.50 * disulfide_score + 0.50 * blosum_score)

        food_functionality_index = min(100.0, max(0.0, round(raw_functionality, 2)))

        return {
            "food_functionality_index": food_functionality_index,
            "disulfide_score": round(disulfide_score, 1),
            "disulfide_disrupted": (disulfide_penalty > 0.0),
            "blosum_score": round(blosum_score, 1),
            "blosum62_raw": blosum_val,
            "rheological_index": round(rheo_score, 1) if rheo_score is not None else None,
            "hydropathy_shift": round(hydro_shift, 2),
            "cysteine_penalty_applied": (disulfide_penalty > 0.0),
            "evidence_level": self.evidence_level,
            "analysis_mode": analysis_mode
        }
