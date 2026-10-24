"""
models/esm_mutator.py: Hugging Face ESM-2 (facebook/esm2_t6_8M_UR50D) masked inference engine.
Evidence Level: [COMPUTATIONAL_PREDICTION] Deep learning transformer masked marginal log-likelihood ratio.
Calculates:
    ΔLLR = log P(mutant | masked_sequence) - log P(wildtype | masked_sequence)

Scientific Interpretation:
- Reflects evolutionary sequence conservation and family-level fitness across homologous proteins.
- Does NOT constitute direct experimental thermodynamic ΔΔG stability or guarantee absence of aggregation.
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import torch
import numpy as np

logger = logging.getLogger(__name__)

CANONICAL_AMINO_ACIDS = [
    "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
    "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"
]

BLOSUM62_SUBSTITUTION_LOGODDS: Dict[Tuple[str, str], int] = {
    ("C", "C"): 9, ("C", "S"): -1, ("C", "A"): 0, ("C", "V"): -1,
    ("Q", "E"): 2, ("Q", "K"): 1, ("Q", "R"): 1, ("Q", "N"): 0, ("Q", "H"): 0, ("Q", "A"): -1, ("Q", "P"): -1,
    ("P", "A"): -1, ("P", "S"): -1, ("P", "T"): -1, ("P", "Q"): -1, ("P", "G"): -2,
    ("R", "K"): 2, ("R", "Q"): 1, ("R", "H"): 0, ("R", "A"): -1,
    ("K", "R"): 2, ("K", "Q"): 1, ("K", "E"): 1, ("K", "N"): 0, ("K", "A"): -1,
    ("W", "Y"): 2, ("W", "F"): 1, ("W", "L"): -2, ("W", "A"): -3,
    ("F", "Y"): 3, ("F", "W"): 1, ("F", "L"): 0, ("F", "I"): 0, ("F", "M"): 0, ("F", "A"): -2,
    ("Y", "F"): 3, ("Y", "W"): 2, ("Y", "H"): 2, ("Y", "A"): -2,
}


class ESMMutator:
    """Protein Language Model Mutator powered by ESM-2 (facebook/esm2_t6_8M_UR50D)."""

    MODEL_CHECKPOINT = "facebook/esm2_t6_8M_UR50D"

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self._is_loaded = False
        self._load_error = None
        self.evidence_level = "[COMPUTATIONAL_PREDICTION]"
        self._init_model()

    def _init_model(self):
        try:
            from transformers import AutoTokenizer, AutoModelForMaskedLM
            logger.info(f"Loading ESM-2 ({self.MODEL_CHECKPOINT}) on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_CHECKPOINT)
            self.model = AutoModelForMaskedLM.from_pretrained(self.MODEL_CHECKPOINT)
            self.model.to(self.device)
            self.model.eval()
            self._is_loaded = True
            logger.info("ESM-2 loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load ESM-2 from Hugging Face ({e}). Utilizing biophysical fallback engine.")
            self._load_error = str(e)
            self._is_loaded = False

    @property
    def is_model_ready(self) -> bool:
        return self._is_loaded and self.model is not None

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "checkpoint": self.MODEL_CHECKPOINT,
            "device": str(self.device),
            "status": "Online (PyTorch ESM-2)" if self.is_model_ready else "Biophysical Fallback Engine",
            "parameter_count": sum(p.numel() for p in self.model.parameters()) if self.is_model_ready else 0,
            "architecture": "ESM-2 Masked LM (Transformer encoder)" if self.is_model_ready else "Physicochemical Log-Odds",
            "evidence_level": self.evidence_level
        }

    def compute_masked_marginals(
        self,
        sequence: str,
        target_pos_1indexed: int
    ) -> List[Dict[str, Any]]:
        seq = sequence.upper().strip()
        seq_len = len(seq)

        if target_pos_1indexed < 1 or target_pos_1indexed > seq_len:
            raise ValueError(f"Position {target_pos_1indexed} out of sequence range [1, {seq_len}]")

        wt_residue = seq[target_pos_1indexed - 1]

        if self.is_model_ready:
            return self._esm_inference(seq, target_pos_1indexed, wt_residue)
        else:
            return self._fallback_inference(seq, target_pos_1indexed, wt_residue)

    def _esm_inference(
        self,
        sequence: str,
        target_pos: int,
        wt_residue: str
    ) -> List[Dict[str, Any]]:
        inputs = self.tokenizer(sequence, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)

        token_pos = target_pos
        mask_token_id = self.tokenizer.mask_token_id

        masked_input_ids = input_ids.clone()
        masked_input_ids[0, token_pos] = mask_token_id

        with torch.no_grad():
            outputs = self.model(masked_input_ids)
            logits = outputs.logits[0, token_pos]
            log_probs = torch.log_softmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)

        wt_token_id = self.tokenizer.convert_tokens_to_ids(wt_residue)
        wt_log_prob = log_probs[wt_token_id].item() if wt_token_id is not None else -10.0

        candidates: List[Dict[str, Any]] = []
        for aa in CANONICAL_AMINO_ACIDS:
            aa_token_id = self.tokenizer.convert_tokens_to_ids(aa)
            if aa_token_id is None:
                continue

            mut_log_prob = log_probs[aa_token_id].item()
            mut_prob = probs[aa_token_id].item()
            llr = mut_log_prob - wt_log_prob

            fitness_score = min(100.0, max(0.0, ((llr + 5.0) / 7.0) * 100.0))

            if llr >= 0.0:
                plausibility = "Evolutionarily Favored"
            elif llr >= -1.5:
                plausibility = "Conservative Neutral"
            elif llr >= -3.5:
                plausibility = "Moderate Tolerance"
            else:
                plausibility = "Deleterious / Destabilizing"

            candidates.append({
                "position": target_pos,
                "wildtype": wt_residue,
                "mutant": aa,
                "mutation": f"{wt_residue}{target_pos}{aa}",
                "llr": round(llr, 4),
                "marginal_prob": round(mut_prob, 5),
                "fitness_score": round(fitness_score, 2),
                "plausibility": plausibility,
                "is_conservative": llr >= -2.0,
                "is_wildtype": (aa == wt_residue),
                "evidence_level": self.evidence_level,
                "model_source": "ESM-2 facebook/esm2_t6_8M_UR50D"
            })

        candidates.sort(key=lambda x: x["llr"], reverse=True)
        return candidates

    def _fallback_inference(
        self,
        sequence: str,
        target_pos: int,
        wt_residue: str
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for aa in CANONICAL_AMINO_ACIDS:
            if aa == wt_residue:
                llr = 0.0
                prob = 0.25
            else:
                pair = (wt_residue, aa)
                rev_pair = (aa, wt_residue)
                blosum_val = BLOSUM62_SUBSTITUTION_LOGODDS.get(
                    pair, BLOSUM62_SUBSTITUTION_LOGODDS.get(rev_pair, -2)
                )
                llr = blosum_val * 0.45
                prob = max(0.005, 0.05 + (blosum_val * 0.02))

            fitness_score = min(100.0, max(0.0, ((llr + 4.0) / 6.0) * 100.0))
            if llr >= 0.0:
                plausibility = "Evolutionarily Favored"
            elif llr >= -1.5:
                plausibility = "Conservative Neutral"
            elif llr >= -3.0:
                plausibility = "Moderate Tolerance"
            else:
                plausibility = "Deleterious / Destabilizing"

            candidates.append({
                "position": target_pos,
                "wildtype": wt_residue,
                "mutant": aa,
                "mutation": f"{wt_residue}{target_pos}{aa}",
                "llr": round(llr, 4),
                "marginal_prob": round(prob, 5),
                "fitness_score": round(fitness_score, 2),
                "plausibility": plausibility,
                "is_conservative": llr >= -1.5,
                "is_wildtype": (aa == wt_residue),
                "evidence_level": "[ILLUSTRATIVE_HEURISTIC]",
                "model_source": "BLOSUM62 Log-Odds Fallback"
            })

        candidates.sort(key=lambda x: x["llr"], reverse=True)
        return candidates

    def generate_candidate_library(
        self,
        sequence: str,
        target_positions: List[int],
        top_k_per_position: int = 3
    ) -> List[Dict[str, Any]]:
        library: List[Dict[str, Any]] = []
        for pos in target_positions:
            all_muts = self.compute_masked_marginals(sequence, pos)
            muts_only = [m for m in all_muts if not m["is_wildtype"]]
            top_candidates = muts_only[:top_k_per_position]
            library.extend(top_candidates)
        return library
