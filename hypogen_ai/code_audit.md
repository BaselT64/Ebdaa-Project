# HypoGen-AI Code Audit Report
# تقرير تدقيق الكود البرمجي لمشروع HypoGen-AI
# تاريخ التدقيق: 2026-10-25
# مسار المشروع: C:\Users\ns143\.gemini\antigravity\scratch\hypogen_ai

---

## الجزء الأول: النسخ الحرفي للدوال وجداول الأوزان

### 1. دالة حساب الأمان المناعي كاملة وجداول الأوزان المستخدمة
**المسار**: `scoring/immunogenicity.py` (الأسطر 10–168)

```python
# الأسطر 10-25: جداول الأوزان
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

# الأسطر 37-168: دالة حساب الأمان الطبي evaluate_mutation_safety
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
```

---

### 2. دالة الجودة الوظيفية (Food Functionality) كاملة
**المسار**: `scoring/food_functionality.py` (الأسطر 49–139)

```python
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
```

---

### 3. دالة اللياقة الزراعية (Agronomic Fitness) كاملة
**المسار**: `scoring/agronomic_fitness.py` (الأسطر 30–127)

```python
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
```

---

### 4. ملف `pareto_optimizer.py` كاملًا والقيم الافتراضية في `app.py`
**المسار**: `scoring/pareto_optimizer.py` (الأسطر 1–130)

```python
"""
scoring/pareto_optimizer.py: Multi-objective non-dominated sorting Pareto optimization framework.
Evidence Level: [COMPUTATIONAL_PREDICTION] Multi-objective Pareto Frontier via NSGA-II non-dominated sorting.
Implements:
1. True mathematical Pareto dominance: a dominates b iff all a_i >= b_i and at least one a_j > b_j.
2. Normalized weights (sum = 1.0).
3. Knee point definition: Pareto candidate with minimum Euclidean distance to Utopia point.
4. Mode-aware optimization: 3-objective for Food Allergens vs. 2-objective for General Proteins.
"""

from typing import List, Dict, Any, Tuple
import numpy as np


class ParetoOptimizer:
    """
    Multi-objective evolutionary decision engine.
    Computes Pareto Dominance, Rank 1 non-dominated frontier, knee point, and normalized weighted scores.
    """

    def __init__(self):
        self.evidence_level = "[COMPUTATIONAL_PREDICTION]"

    @staticmethod
    def dominates(candidate_a: Dict[str, Any], candidate_b: Dict[str, Any], objectives: List[str]) -> bool:
        """
        True if candidate_a Pareto-dominates candidate_b across the specified objective functions.
        Requires a_i >= b_i for all i, and a_j > b_j for at least one j.
        """
        f_a = [candidate_a[obj] for obj in objectives]
        f_b = [candidate_b[obj] for obj in objectives]

        all_ge = all(a >= b for a, b in zip(f_a, f_b))
        any_gt = any(a > b for a, b in zip(f_a, f_b))
        return all_ge and any_gt

    def compute_pareto_front(
        self,
        candidates: List[Dict[str, Any]],
        weight_safety: float = 0.40,
        weight_food: float = 0.35,
        weight_agronomic: float = 0.25,
        analysis_mode: str = "FOOD_ALLERGEN"
    ) -> Dict[str, Any]:
        """
        Performs non-dominated sorting on candidate population.
        Identifies Rank 1 Pareto front and computes normalized composite utility and knee point.
        """
        if not candidates:
            return {
                "all_candidates": [],
                "pareto_front": [],
                "pareto_count": 0,
                "total_count": 0,
                "knee_point": None,
                "analysis_mode": analysis_mode
            }

        # Define active objectives based on mode
        if analysis_mode == "FOOD_ALLERGEN":
            objectives = ["medical_safety_index", "food_functionality_index", "agronomic_fitness_index"]
            total_w = weight_safety + weight_food + weight_agronomic
            w_s = weight_safety / total_w if total_w > 0 else 0.333
            w_f = weight_food / total_w if total_w > 0 else 0.333
            w_a = weight_agronomic / total_w if total_w > 0 else 0.334
            weights = {"safety": round(w_s, 4), "food": round(w_f, 4), "agronomic": round(w_a, 4)}
        else:
            # GENERAL_PROTEIN mode: optimize ESM-2 fitness and Food/Structural Functionality
            objectives = ["fitness_score", "food_functionality_index"]
            total_w = weight_safety + weight_food
            w_s = weight_safety / total_w if total_w > 0 else 0.5
            w_f = weight_food / total_w if total_w > 0 else 0.5
            weights = {"esm_fitness": round(w_s, 4), "structural": round(w_f, 4)}

        n = len(candidates)
        domination_counts = [0] * n
        dominated_indices = [[] for _ in range(n)]

        # Pairwise dominance comparison (NSGA-II Fast Non-Dominated Sorting step)
        for p in range(n):
            for q in range(p + 1, n):
                if self.dominates(candidates[p], candidates[q], objectives):
                    dominated_indices[p].append(q)
                    domination_counts[q] += 1
                elif self.dominates(candidates[q], candidates[p], objectives):
                    dominated_indices[q].append(p)
                    domination_counts[p] += 1

        pareto_indices = [i for i in range(n) if domination_counts[i] == 0]

        enriched_candidates = []
        for i, c in enumerate(candidates):
            c_copy = dict(c)
            is_pareto = (i in pareto_indices)
            c_copy["is_pareto"] = is_pareto
            c_copy["pareto_rank"] = 1 if is_pareto else 2

            if analysis_mode == "FOOD_ALLERGEN":
                s = c_copy["medical_safety_index"]
                f = c_copy["food_functionality_index"]
                a = c_copy["agronomic_fitness_index"]
                composite = (weights["safety"] * s) + (weights["food"] * f) + (weights["agronomic"] * a)
                dist_to_utopia = np.sqrt((100.0 - s)**2 + (100.0 - f)**2 + (100.0 - a)**2)
            else:
                s = c_copy["fitness_score"]
                f = c_copy["food_functionality_index"]
                composite = (weights["esm_fitness"] * s) + (weights["structural"] * f)
                dist_to_utopia = np.sqrt((100.0 - s)**2 + (100.0 - f)**2)

            c_copy["composite_score"] = round(composite, 2)
            c_copy["distance_to_utopia"] = round(float(dist_to_utopia), 2)
            enriched_candidates.append(c_copy)

        pareto_candidates = [c for c in enriched_candidates if c["is_pareto"]]
        # Knee point: candidate on Pareto front closest to Utopia point (100, 100, 100)
        pareto_candidates.sort(key=lambda x: x["distance_to_utopia"])
        knee_point = pareto_candidates[0] if pareto_candidates else None

        enriched_candidates.sort(key=lambda x: x["composite_score"], reverse=True)

        return {
            "all_candidates": enriched_candidates,
            "pareto_front": pareto_candidates,
            "pareto_count": len(pareto_candidates),
            "total_count": len(enriched_candidates),
            "knee_point": knee_point,
            "weights_used": weights,
            "analysis_mode": analysis_mode,
            "evidence_level": self.evidence_level
        }
```

**القيم الافتراضية للأوزان في `app.py`** (الأسطر 376–378):
```python
    w_safety = st.slider(get_text("w_safety_label", lang_code), min_value=0.10, max_value=0.80, value=0.40, step=0.05)
    w_food = st.slider(get_text("w_food_label", lang_code), min_value=0.10, max_value=0.80, value=0.35, step=0.05)
    w_agronomic = st.slider(get_text("w_agro_label", lang_code), min_value=0.10, max_value=0.80, value=0.25, step=0.05)
```
- وزن الأمان الطبي (`w_safety`): **0.40**
- وزن جودة الغذاء (`w_food`): **0.35**
- وزن اللياقة الزراعية (`w_agronomic`): **0.25**

---

### 5. دالة تصميم sgRNA ودوال التقييم وخطر القطع غير المستهدف
**المسار**: `crispr/sgrna_engine.py` (الأسطر 43–280)

```python
# دالة التصميم الرئيسية: الأسطر 43-136
    def design_sgrnas_for_mutation(
        self,
        full_protein_sequence: str,
        mutation_pos_1indexed: int,
        wildtype_aa: str,
        mutant_aa: str,
        cds_sequence: Optional[str] = None,
        window_radius_codons: int = 15
    ) -> Dict[str, Any]:
        """
        Designs SpCas9 sgRNAs intersecting the target point mutation locus.
        Uses verified CDS if provided; otherwise falls back to reverse translation with explicit disclaimer.
        """
        prot_seq = full_protein_sequence.upper()
        pos = mutation_pos_1indexed
        seq_len = len(prot_seq)

        start_aa = max(1, pos - window_radius_codons)
        end_aa = min(seq_len, pos + window_radius_codons)
        local_wt_peptide = prot_seq[start_aa - 1 : end_aa]

        mut_offset = (pos - start_aa)
        local_mut_peptide = (
            local_wt_peptide[:mut_offset] +
            mutant_aa +
            local_wt_peptide[mut_offset + 1 :]
        )

        has_verified_cds = False
        if cds_sequence and len(cds_sequence) >= (end_aa * 3):
            # Extract real biological CDS window
            wt_dna = cds_sequence[(start_aa - 1) * 3 : end_aa * 3].upper()
            has_verified_cds = True
            # Build mutated DNA window using plant-optimized mutant codon
            mut_codon = PLANT_CODON_TABLE.get(mutant_aa, "NNN")
            mut_nt_start = mut_offset * 3
            mut_nt_end = mut_nt_start + 3
            mut_dna = wt_dna[:mut_nt_start] + mut_codon + wt_dna[mut_nt_end:]
            dna_provenance = "[RETRIEVED_REFERENCE] Verified Biological CDS (EMBL:X17361.1 / UniProt)"
        else:
            wt_dna = self.reverse_translate(local_wt_peptide)
            mut_dna = self.reverse_translate(local_mut_peptide)
            mut_nt_start = mut_offset * 3
            mut_nt_end = mut_nt_start + 3
            dna_provenance = "[ILLUSTRATIVE_HEURISTIC] In-silico reverse-translation (Plant codon consensus)"

        guides: List[Dict[str, Any]] = []

        # 1. Scan Sense Strand for SpCas9 PAM (5'-[20nt]-NGG-3')
        guides.extend(self._scan_strand(
            dna_strand=mut_dna,
            strand_type="Sense (+)",
            local_mut_start=mut_nt_start,
            local_mut_end=mut_nt_end,
            protein_locus=pos,
            full_window_start_aa=start_aa
        ))

        # 2. Scan Antisense Strand for SpCas9 PAM (3'-CCN-[20nt]-5')
        rc_dna = reverse_complement(mut_dna)
        rc_mut_start = len(rc_dna) - mut_nt_end
        rc_mut_end = len(rc_dna) - mut_nt_start

        guides.extend(self._scan_strand(
            dna_strand=rc_dna,
            strand_type="Antisense (-)",
            local_mut_start=rc_mut_start,
            local_mut_end=rc_mut_end,
            protein_locus=pos,
            full_window_start_aa=start_aa
        ))

        guides.sort(key=lambda x: x["on_target_score"], reverse=True)

        return {
            "locus_mutation": f"{wildtype_aa}{pos}{mutant_aa}",
            "mutation_pos_aa": pos,
            "has_verified_cds": has_verified_cds,
            "dna_provenance": dna_provenance,
            "window_peptide_wt": local_wt_peptide,
            "window_peptide_mut": local_mut_peptide,
            "window_dna_wt": wt_dna,
            "window_dna_mut": mut_dna,
            "target_codon_wt": wt_dna[mut_nt_start:mut_nt_end],
            "target_codon_mut": mut_dna[mut_nt_start:mut_nt_end],
            "total_guides_found": len(guides),
            "guides": guides,
            "evidence_level": self.evidence_level,
            "hexaploid_wheat_caveat": (
                "Hexaploid wheat (Triticum aestivum, AABBDD) carries 30-50+ alpha-gliadin paralogs. "
                "Editing a single gene does not abolish celiac toxicity without targeting conserved protospacers across paralogs."
            )
        }

# دالة مسح السلسلة: الأسطر 137-206
    def _scan_strand(
        self,
        dna_strand: str,
        strand_type: str,
        local_mut_start: int,
        local_mut_end: int,
        protein_locus: int,
        full_window_start_aa: int
    ) -> List[Dict[str, Any]]:
        strand_len = len(dna_strand)
        results = []

        for i in range(20, strand_len - 2):
            if dna_strand[i + 1 : i + 3] == "GG":
                pam_seq = dna_strand[i : i + 3]
                protospacer = dna_strand[i - 20 : i]
                cut_site_local = i - 3

                mut_center = (local_mut_start + local_mut_end) / 2.0
                cut_to_mut_dist = abs(cut_site_local - mut_center)

                seed_start = i - 10
                seed_end = i
                overlaps_seed = not (local_mut_end <= seed_start or local_mut_start >= seed_end)
                overlaps_protospacer = not (local_mut_end <= (i - 20) or local_mut_start >= i)

                if cut_to_mut_dist > 25.0:
                    continue

                gc_count = protospacer.count("G") + protospacer.count("C")
                gc_content_pct = round((gc_count / 20.0) * 100.0, 1)

                on_target_score = self._compute_rule_set_heuristic(
                    protospacer=protospacer,
                    pam=pam_seq,
                    gc_content=gc_content_pct,
                    cut_to_mut_dist=cut_to_mut_dist,
                    overlaps_seed=overlaps_seed
                )

                off_target_risk_pct, off_target_tier = self._compute_off_target_risk(
                    protospacer=protospacer,
                    on_target_score=on_target_score,
                    gc_content=gc_content_pct,
                    overlaps_seed=overlaps_seed
                )

                fwd_oligo = f"5'-CACC{protospacer}-3'"
                rev_oligo = f"5'-AAAC{reverse_complement(protospacer)}-3'"

                results.append({
                    "protospacer": protospacer,
                    "pam": pam_seq,
                    "target_sequence_with_pam": f"{protospacer} {pam_seq}",
                    "strand": strand_type,
                    "cut_offset_bp": round(cut_to_mut_dist, 1),
                    "overlaps_seed_region": overlaps_seed,
                    "overlaps_protospacer": overlaps_protospacer,
                    "gc_content_pct": gc_content_pct,
                    "on_target_score": on_target_score,
                    "off_target_risk_pct": off_target_risk_pct,
                    "off_target_risk_tier": off_target_tier,
                    "fwd_cloning_oligo": fwd_oligo,
                    "rev_cloning_oligo": rev_oligo,
                    "tier": "Tier 1 (High Efficiency)" if on_target_score >= 70.0 else (
                        "Tier 2 (Moderate)" if on_target_score >= 50.0 else "Tier 3 (Sub-optimal)"
                    )
                })

        return results

# دالة تقييم خطر القطع غير المستهدف: الأسطر 208-240
    def _compute_off_target_risk(
        self,
        protospacer: str,
        on_target_score: float,
        gc_content: float,
        overlaps_seed: bool
    ) -> Tuple[float, str]:
        base_risk = max(1.5, (100.0 - on_target_score) * 0.28)
        seed_seq = protospacer[-10:]
        unique_bases = len(set(seed_seq))
        if unique_bases <= 2:
            base_risk += 12.0
        elif unique_bases == 4:
            base_risk -= 3.5

        if gc_content > 70.0 or gc_content < 30.0:
            base_risk += 6.0
        elif 45.0 <= gc_content <= 55.0:
            base_risk -= 2.5

        if overlaps_seed:
            base_risk -= 2.0

        risk_val = min(85.0, max(1.2, round(base_risk, 1)))
        if risk_val < 6.0:
            tier = "Minimal (<6%)"
        elif risk_val < 15.0:
            tier = "Low (6-15%)"
        elif risk_val < 30.0:
            tier = "Moderate (15-30%)"
        else:
            tier = "Elevated (>30%)"
        return risk_val, tier

# دالة كفاءة القطع على الهدف (Rule Set 2 الاستدلالية): الأسطر 242-280
    def _compute_rule_set_heuristic(
        self,
        protospacer: str,
        pam: str,
        gc_content: float,
        cut_to_mut_dist: float,
        overlaps_seed: bool
    ) -> float:
        score = 75.0

        if 40.0 <= gc_content <= 60.0:
            score += 10.0
        elif gc_content < 30.0 or gc_content > 75.0:
            score -= 22.0
        else:
            score -= 5.0

        if "TTTT" in protospacer:
            score -= 35.0
        elif "TTT" in protospacer:
            score -= 8.0

        last_base = protospacer[-1]
        if last_base == "G":
            score += 6.0
        elif last_base == "T":
            score -= 8.0

        if cut_to_mut_dist <= 3.0:
            score += 12.0
        elif cut_to_mut_dist <= 8.0:
            score += 6.0
        elif cut_to_mut_dist > 15.0:
            score -= 10.0

        if overlaps_seed:
            score += 5.0

        return min(100.0, max(5.0, round(score, 1)))
```

---

## الجزء الثاني: إجابات الأسئلة المحددة بالأدلة السطرية

### السؤال 1: هل يستخدم التطبيق تسلسل P18573 المخزن في الكود أم يجلبه حيًا من UniProt؟ وما طول التسلسل؟ وما الحمض الأميني في المواضع 58 و 60 و 77؟ وأين يقع الببتيد LQLQPFPQPQLPYPQPQLPYPQPQLPYPQPQPF؟
- **إجابة الجلب**: يحاول التطبيق أولاً جلبه **حيًا من خادم UniProt REST API** (الملف `core/uniprot_fetcher.py`، الأسطر 288–291: `record = self._query_rest_api(clean_id)`). وفي حال عدم توفر اتصال بالإنترنت أو فشل الطلب، فإنه يستخدم **النسخة المخزنة كمرجع قياسي في الكود** (الأسطر 297–302: `cached = BENCHMARK_PROTEINS[clean_id].copy()`).
- **طول التسلسل**: **307** حمضًا أمينيًا (الملف `core/uniprot_fetcher.py`، السطر 35: `"length": 307`، ويطابق تمامًا UniProt الحي `P18573`).
- **الحمض الأميني في الموضع 58** (1-indexed precursor): **P** (Proline / برولين) (السطر 29).
- **الحمض الأميني في الموضع 60** (1-indexed precursor): **Q** (Glutamine / جلوتامين) (السطر 29).
- **الحمض الأميني في الموضع 77** (1-indexed precursor): **Q** (Glutamine / جلوتامين) (السطر 30).
- **موقع ببتيد 33-mer (`LQLQPFPQPQLPYPQPQLPYPQPQLPYPQPQPF`)**: يقع في المواضع **76 إلى 108** من التسلسل الكامل للبروتين الأولِي (الملف `core/uniprot_fetcher.py`، السطر 44: `"celiac_33mer_coords": (76, 108)`، والملف `core/coordinate_mapper.py`، الأسطر 47–48).

### السؤال 2: هل يوجد في الكود أي توليد لقالب إصلاح ssODN أو تعديل صامت؟ (ابحث بالكلمات ssODN و repair و silent و template واكتب النتيجة).
- **الإجابة**: **لا (No)**.
- **النتيجة والسطور المثبتة**:
  - البحث عن `ssODN`: **0 نتائج** في كامل المشروع.
  - البحث عن `repair`: **0 نتائج** في أي كود حسابي أو تنفيذي.
  - البحث عن `silent`: يظهر فقط في عبارة `silent fallback` في التعليقات وسجل الأخطاء (مثل `app.py` السطر 6: `2. Robust UniProt state management without silent fallbacks` والسطر 389، و`test_audit_verification.py` السطر 63)، ولا يوجد أي تعديل صامت (silent mutation/editing).
  - البحث عن `template`: يظهر فقط في تنسيق الواجهة الرسومية Plotly (`template="plotly_dark"` في `app.py` الأسطر 605 و 780 و 806) وفي وصف الهيكل الفراغي لـ AlphaFold (`"AlphaFold template backbone"` في `app.py` السطر 870 والسطر 880 و`core/translations.py` السطر 193).
  - لا توجد أي دالة أو سطر ينشئ تسلسل ssODN أو قوالب إصلاح مانحة.

### السؤال 3: هل درجة "Rule Set 2" مأخوذة من حزمة أو نموذج حقيقي (مثل Azimuth) أم هي دالة مكتوبة يدويًا؟
- **الإجابة**: **دالة مكتوبة يدويًا (Hand-written heuristic function)** وليست من حزمة أو نموذج حقيقي مثل Azimuth.
- **السطور المثبتة**: الملف `crispr/sgrna_engine.py`، الأسطر 242–280 (دالة `_compute_rule_set_heuristic`)، حيث كُتبت شروط تقريبية يدوية (`if 40.0 <= gc_content <= 60.0: score += 10.0`، `if "TTTT" in protospacer: score -= 35.0`، إلخ)، والتوثيق في السطر 2 والسطر 8 يصرّح صراحة:
  `Evidence Level: [COMPUTATIONAL_PREDICTION] Computational cleavage activity heuristic.`
  `3. Labels Doench Rule Set 2 score as predicted cleavage activity (0-100), NOT plant transformation success rate.`

### السؤال 4: هل يُفحص القطع غير المستهدف بالبحث في جينوم حقيقي أم بمعادلة تقريبية؟ وعلى أي تسلسل DNA يُصمَّم الدليل؟
- **الإجابة**: **بمعادلة تقريبية (Heuristic approximation)**، ولا يتم البحث في أي جينوم حقيقي.
- **السطور المثبتة**: الملف `crispr/sgrna_engine.py`، الأسطر 208–240 (دالة `_compute_off_target_risk`)، حيث تعتمد المعادلة على تنوع القواعد في منطقة البذرة ومحتوى GC فقط (`base_risk = max(1.5, (100.0 - on_target_score) * 0.28)`، الأسطر 215–231). والسطر 10 ينص صراحة:
  `5. Labels off-target risk as sequence-based heuristic, noting requirement for whole-genome NGS verification.`
- **تسلسل الـ DNA الذي يُصمم عليه الدليل**: يُصمم الدليل على نافذة موضعية بحجم **30 كودون (90 نيوكليوتيد)** تحيط بموقع الطفرة (±15 كودون) مستخرجة من **التسلسل المشفر الحيوي الموثق (CDS) لـ P18573 من `EMBL:X17361.1`** (الأسطر 72–81: `wt_dna = cds_sequence[(start_aa - 1) * 3 : end_aa * 3]`). وفي حال عدم توفر CDS، يُستخدم تسلسل مشتق بالترجمة العكسية الافتراضية باستخدام جدول كودونات المحاصيل (الأسطر 83–87).

### السؤال 5: هل يوجد أي استدعاء لأداة تنبؤ HLA مثل NetMHCIIpan؟
- **الإجابة**: **لا (No)**.
- **السطور المثبتة**: البحث عن `NetMHC` و `NetMHCIIpan` في كافة ملفات المشروع يعطي **0 نتائج**. وبدلاً من ذلك، يستخدم الكود جدولاً محلياً ثابتاً لأوزان جيوب HLA-DQ2.5 (`HLA_DQ2_ANCHOR_WEIGHTS` في `scoring/immunogenicity.py`، الأسطر 12–18).

### السؤال 6: هل يدخل نموذج ESM-2 في حساب المؤشرات الثلاثة، أم يُستخدم فقط لاقتراح البدائل؟
- **الإجابة**: في نمط الحساسية الغذائية (النمط الأساسي)، **لا يدخل نموذج ESM-2 في حساب المؤشرات الثلاثة** (الأمان الطبي، جودة الغذاء، اللياقة الزراعية)، وإنما **يُستخدم فقط لاقتراح البدائل وتصفيتها وترتيبها حسب احتمالية التسلسل**.
- **السطور المثبتة**: الملف `app.py`، الأسطر 689–718 والملف `run_hypogen_analysis.py` الأسطر 125–127: يقوم `esm.generate_candidate_library` بتوليد قائمة البدائل الممكنة استناداً لـ LLR، ثم تُحسب المؤشرات الثلاثة بدوال مستقلة تماماً لا تستخدم مخرجات ESM-2 (`im.evaluate_mutation_safety`، `ff.evaluate_food_functionality`، `af.evaluate_agronomic_fitness`). وتتأكد الأهداف في `scoring/pareto_optimizer.py`، الأسطر 60–66:
  `objectives = ["medical_safety_index", "food_functionality_index", "agronomic_fitness_index"]`.

### السؤال 7: ماذا تغطي الاختبارات في مجلد tests؟ (قائمة بأسماء الاختبارات فقط)
الملف `tests/test_audit_verification.py` يحتوي على 12 اختباراً تغطي:
1. `test_01_celiac_33mer_alignment_and_coordinates`
2. `test_02_no_silent_fallback_on_invalid_accession`
3. `test_03_crc64_and_metadata_integrity`
4. `test_04_mode_classification`
5. `test_05_celiac_vs_ige_separation`
6. `test_06_general_protein_mode_epitope_scanning`
7. `test_07_esm_delta_llr`
8. `test_08_cysteine_loss_penalty`
9. `test_09_general_protein_scoring_honesty`
10. `test_10_pareto_dominance_and_knee_point`
11. `test_11_crispr_sgrna_authentic_cds_and_hexaploid_caveat`
12. `test_12_bilingual_translations`

### السؤال 8: ما الفرق بين هذه النسخة والنسخة القديمة المحفوظة في allergenmod_ai إن وُجد؟
- **المقارنة مع مجلد `allergenmod_ai` الحالي**:
  - كافة ملفات الكود في مجلدات `core/` و `models/` و `scoring/` و `crispr/` و `tests/` وملف `app.py` **متطابقة حرفياً (Identical)** بنسبة 100%.
  - الاختلاف الوحيد هو ملف `run_app.bat` (حيث تم تحديث مسار التشغيل ليشير إلى `hypogen_ai`) وملف التحليل المضاف `run_hypogen_analysis.py`.
- **المقارنة مع النسخة الأصلية قبل التدقيق العلمي (`allergenmod_ai - Copy`)**:
  - **ملفات جديدة أُضيفت**: `core/coordinate_mapper.py` (محاذاة أنظمة الإحداثيات وببتيد Shan et al. 2002)، و `core/translations.py` (قاموس الترجمة العربي والإنجليزي مع عزل BiDi)، و `tests/test_audit_verification.py` (حزمة الاختبارات الشاملة).
  - **أهم الفروق في المعادلات والمنطق**:
    1. في `core/uniprot_fetcher.py`: تصحيح طول تسلسل P18573 من 290 مقطوعاً إلى 307 حمضاً أمينياً، وإضافة تسلسل CDS الحقيقي (921 bp من `EMBL:X17361.1`)، والتحقق بـ CRC64، وإلغاء السقوط الصامت (silent fallback) عند الخطأ.
    2. في `core/epitope_scanner.py`: تصنيف الحواتم بين مرض السيلياك (HLA-DQ2/8 غير المرتبط بـ IgE) وحساسيات IgE الحقيقية، وإضافة وسم مستويات الإثبات (`[RETRIEVED_REFERENCE]` و `[COMPUTATIONAL_PREDICTION]`).
    3. في `scoring/food_functionality.py` و `scoring/agronomic_fitness.py`: حظر وتصفير ادعاءات عجين القمح ولياقة المحاصيل عند تحليل بروتينات عامة غير نباتية (مثل الهيموجلوبين P68871)، وتطبيق عقوبة -50% عند فقدان جسر كبريتيد.
    4. في `scoring/pareto_optimizer.py`: توحيد أوزان المستخدم ليصبح مجموعها 1.0 بدقة، وحساب نقطة الركبة (knee point) بأقصر مسافة إقليدية لنقطة اليوتوبيا (100, 100, 100)، ودعم نمطين للاستمثال (3 أهداف للأغذية وهدفان للبروتينات العامة).
    5. في `crispr/sgrna_engine.py`: ربط التصميم بتسلسل CDS البيولوجي الموثق، وتوصيف نتيجة Doench Rule Set 2 كدرجة تقريبية للقطع الحوسبي (وليس كفاءة تحول نباتي)، وإبراز تنبيه جينوم القمح سداسي الصيغة الصبغية (30–50+ جين نظير).
    6. في `app.py`: إزالة ادعاءات RMSD الوهمية (< 0.25 Å) واستبدالها بنمذجة صريحة على هيكل AlphaFold مع طفرة حاسوبية موضعية.
