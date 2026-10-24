"""
scoring/agronomic_fitness.py: Seed germination nitrogen reserves & plant defense domain conservation.
Evidence Level: [ILLUSTRATIVE_HEURISTIC] Sequence-based seed storage & plant defense proxy.
Mode-aware: Disables crop agronomic claims for non-plant / general proteins.
"""

from typing import Dict, Any, List, Optional
import numpy as np

# Nitrogen atom counts per amino acid residue
NITROGEN_ATOMS: Dict[str, int] = {
    'R': 4,  # Arginine: highest N content
    'H': 3,  # Histidine
    'N': 2,  # Asparagine
    'Q': 2,  # Glutamine: primary seed nitrogen transporter
    'K': 2,  # Lysine
    'W': 2,  # Tryptophan
    'A': 1, 'C': 1, 'D': 1, 'E': 1, 'F': 1,
    'G': 1, 'I': 1, 'L': 1, 'M': 1, 'P': 1,
    'S': 1, 'T': 1, 'V': 1, 'Y': 1
}


class AgronomicFitnessScorer:
    """Assesses seed germination viability, nitrogen storage retention, and plant defense integrity."""

    def __init__(self):
        self.evidence_level = "[ILLUSTRATIVE_HEURISTIC]"

    def evaluate_agronomic_fitness(
        self,
        wt_sequence: str,
        mut_sequence: str,
        mutation_pos_1indexed: int,
        curated_features: Optional[List[Dict[str, Any]]] = None,
        analysis_mode: str = "FOOD_ALLERGEN"
    ) -> Dict[str, Any]:
        """Calculates Agronomic Fitness Index (0.0 to 100.0%). Disables crop metrics for general proteins."""
        if analysis_mode == "GENERAL_PROTEIN":
            return {
                "agronomic_fitness_index": 50.0,
                "nitrogen_score": 50.0,
                "nitrogen_atoms_wt": 1,
                "nitrogen_atoms_mut": 1,
                "nitrogen_status": "Non-crop protein: Seed agronomic fitness not applicable (Neutral baseline 50.0%).",
                "defense_score": 50.0,
                "in_defense_domain": False,
                "defense_role": "N/A",
                "metabolic_feasibility": 50.0,
                "is_applicable": False,
                "evidence_level": self.evidence_level,
                "analysis_mode": analysis_mode
            }

        wt_seq = wt_sequence.upper()
        mut_seq = mut_sequence.upper()
        pos = mutation_pos_1indexed
        wt_aa = wt_seq[pos - 1]
        mut_aa = mut_seq[pos - 1]

        # 1. Seed Nitrogen Storage Conservation
        n_wt = NITROGEN_ATOMS.get(wt_aa, 1)
        n_mut = NITROGEN_ATOMS.get(mut_aa, 1)

        if n_mut >= n_wt:
            nitrogen_score = 100.0
            n_status = "Maintained or enriched nitrogen reserve"
        else:
            loss_ratio = (n_wt - n_mut) / n_wt
            nitrogen_score = max(55.0, 100.0 - (loss_ratio * 35.0))
            n_status = f"Minor nitrogen depot reduction (-{n_wt - n_mut} N atom)"

        # 2. Plant Defense Domain Conservation
        in_defense_domain = False
        defense_penalty = 0.0
        defense_role = "Structural / Storage Region"

        if curated_features:
            for feat in curated_features:
                desc = feat.get("description", "").lower()
                f_type = feat.get("type", "").lower()
                if any(k in desc or k in f_type for k in ["defense", "antimicrobial", "chitin", "inhibitor", "calyx"]):
                    if feat.get("start", 0) <= pos <= feat.get("end", 0):
                        in_defense_domain = True
                        defense_role = feat.get("description", "Plant Defense Domain")
                        break

        if in_defense_domain:
            if wt_aa == "C" and mut_aa != "C":
                defense_penalty = 40.0
            elif wt_aa in ["W", "Y", "F"] and mut_aa in ["G", "P"]:
                defense_penalty = 25.0
            else:
                defense_penalty = 10.0

        defense_score = max(20.0, 100.0 - defense_penalty)

        # 3. Metabolic Sprouting Feasibility
        metabolic_feasibility = 95.0
        if mut_aa in ["W", "Y", "H"]:
            metabolic_feasibility = 88.0
        elif mut_aa in ["A", "S", "N", "T", "G"]:
            metabolic_feasibility = 98.0

        # Composite Agronomic Fitness Index
        composite_score = (
            0.45 * nitrogen_score +
            0.35 * defense_score +
            0.20 * metabolic_feasibility
        )

        agronomic_fitness_index = min(100.0, max(0.0, round(composite_score, 2)))

        return {
            "agronomic_fitness_index": agronomic_fitness_index,
            "nitrogen_score": round(nitrogen_score, 1),
            "nitrogen_atoms_wt": n_wt,
            "nitrogen_atoms_mut": n_mut,
            "nitrogen_status": n_status,
            "defense_score": round(defense_score, 1),
            "in_defense_domain": in_defense_domain,
            "defense_role": defense_role,
            "metabolic_feasibility": round(metabolic_feasibility, 1),
            "is_applicable": True,
            "evidence_level": self.evidence_level,
            "analysis_mode": analysis_mode
        }
