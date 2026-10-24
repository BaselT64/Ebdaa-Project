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
