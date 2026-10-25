"""
tests/test_audit_verification.py: Comprehensive Scientific & Functional Audit Test Suite.
Verifies all 15 audit criteria across coordinate systems, UniProt fetcher, epitope scanner,
ESM-2 mutator, biophysical scorers, Pareto optimizer, CRISPR engine, and bilingual localization.
"""

import unittest
import os
import sys

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.coordinate_mapper import CoordinateMapper, CELIAC_33MER_REFERENCE
from core.uniprot_fetcher import UniProtFetcher, BENCHMARK_PROTEINS, classify_protein_mode
from core.epitope_scanner import EpitopeScanner
from core.translations import get_text, get_rtl_css, EVIDENCE_LABELS
from models.esm_mutator import ESMMutator
from scoring.immunogenicity import ImmunogenicityScorer
from scoring.food_functionality import FoodFunctionalityScorer
from scoring.agronomic_fitness import AgronomicFitnessScorer
from scoring.pareto_optimizer import ParetoOptimizer
from crispr.sgrna_engine import CRISPRsgRNAEngine


class TestScientificAudit(unittest.TestCase):

    def setUp(self):
        self.coord_mapper = CoordinateMapper()
        self.uniprot_fetcher = UniProtFetcher()
        self.epitope_scanner = EpitopeScanner()
        self.im_scorer = ImmunogenicityScorer()
        self.ff_scorer = FoodFunctionalityScorer()
        self.af_scorer = AgronomicFitnessScorer()
        self.pareto_opt = ParetoOptimizer()
        self.crispr_engine = CRISPRsgRNAEngine()

    # -------------------------------------------------------------
    # 1. Coordinate systems & Shan et al. 2002 33-mer alignment
    # -------------------------------------------------------------
    def test_01_celiac_33mer_alignment_and_coordinates(self):
        p18573_data = self.uniprot_fetcher.fetch_protein("P18573")
        seq = p18573_data["sequence"]
        self.assertEqual(len(seq), 307, "P18573 canonical precursor length must be 307 aa.")

        align = self.coord_mapper.align_celiac_33mer(seq)
        self.assertEqual(align["match_status"], "Exact Match", "33-mer must match P18573 exactly.")
        self.assertEqual(align["locus_start_1indexed"], 76, "33-mer locus must start at precursor residue 76.")
        self.assertEqual(align["locus_end_1indexed"], 108, "33-mer locus must end at precursor residue 108.")

        # Test coordinate mapping: Precursor Q77 = 33-mer Q2 = Mature Q57
        map_77 = self.coord_mapper.map_precursor_to_mature(77, signal_peptide_length=20)
        self.assertEqual(map_77["precursor_pos"], 77)
        self.assertEqual(map_77["mature_pos"], 57)
        self.assertEqual(map_77["peptide_33mer_pos"], 2)
        self.assertEqual(map_77["residue"], "Q")

    # -------------------------------------------------------------
    # 2. UniProt live search state & no silent fallback
    # -------------------------------------------------------------
    def test_02_no_silent_fallback_on_invalid_accession(self):
        with self.assertRaises(ValueError):
            self.uniprot_fetcher.fetch_protein("INVALID_NONEXISTENT_XYZ999")

    def test_03_crc64_and_metadata_integrity(self):
        p18573 = self.uniprot_fetcher.fetch_protein("P18573")
        self.assertTrue("crc64" in p18573)
        self.assertEqual(p18573["crc64"], "06C1858BD96F1E08")
        self.assertEqual(p18573["analysis_mode"], "FOOD_ALLERGEN")

    # -------------------------------------------------------------
    # 3. Protein Mode Classification (Food Allergen vs. General Protein)
    # -------------------------------------------------------------
    def test_04_mode_classification(self):
        p18573 = self.uniprot_fetcher.fetch_protein("P18573")
        p68871 = self.uniprot_fetcher.fetch_protein("P68871")  # Hemoglobin
        self.assertEqual(classify_protein_mode(p18573), "FOOD_ALLERGEN")
        self.assertEqual(classify_protein_mode(p68871), "GENERAL_PROTEIN")

    # -------------------------------------------------------------
    # 4. Epitope Scanner: Separation of Celiac T-Cell from IgE
    # -------------------------------------------------------------
    def test_05_celiac_vs_ige_separation(self):
        p18573 = self.uniprot_fetcher.fetch_protein("P18573")
        scan = self.epitope_scanner.scan_sequence(p18573["sequence"], p18573.get("features"), analysis_mode="FOOD_ALLERGEN")
        epitopes = scan["epitopes"]

        # Ensure Celiac 33-mer is found and labeled with its distinct immune mechanism
        celiac_hits = [e for e in epitopes if "CELIAC" in e["id"]]
        self.assertTrue(len(celiac_hits) > 0)
        self.assertIn("HLA-DQ2.5", celiac_hits[0]["immune_mechanism"])
        self.assertEqual(celiac_hits[0]["evidence_level"], "[RETRIEVED_REFERENCE]")

        # Test Peanut Ara h 2 linear IgE
        q6psu2 = self.uniprot_fetcher.fetch_protein("Q6PSU2")
        scan_peanut = self.epitope_scanner.scan_sequence(q6psu2["sequence"], q6psu2.get("features"), analysis_mode="FOOD_ALLERGEN")
        peanut_hits = [e for e in scan_peanut["epitopes"] if "ARAH2" in e["id"]]
        self.assertTrue(len(peanut_hits) > 0)
        self.assertIn("IgE", peanut_hits[0]["immune_mechanism"])

    def test_06_general_protein_mode_epitope_scanning(self):
        p68871 = self.uniprot_fetcher.fetch_protein("P68871")
        scan_gen = self.epitope_scanner.scan_sequence(p68871["sequence"], p68871.get("features"), analysis_mode="GENERAL_PROTEIN")
        # In GENERAL_PROTEIN mode, no food allergen epitopes should be assigned
        for ep in scan_gen["epitopes"]:
            self.assertNotIn("Gluten Prolamin", ep.get("allergen_type", ""))

    # -------------------------------------------------------------
    # 5. ESM-2 ΔLLR formula & disclaimers
    # -------------------------------------------------------------
    def test_07_esm_delta_llr(self):
        esm = ESMMutator()
        seq = "LQLQPFPQPQLPYPQPQLPYPQPQLPYPQPQPF"
        marginals = esm.compute_masked_marginals(seq, target_pos_1indexed=2)  # Q2
        self.assertEqual(len(marginals), 20)
        # Wild-type Q marginal has ΔLLR == 0
        wt_entry = next(m for m in marginals if m["is_wildtype"])
        self.assertAlmostEqual(wt_entry["llr"], 0.0, places=2)

    # -------------------------------------------------------------
    # 6. Biophysical Scorers & Cysteine Penalty
    # -------------------------------------------------------------
    def test_08_cysteine_loss_penalty(self):
        wt_seq = "ACDEFGHIKLMNPQRSTVWY"
        mut_seq = "ASDEFGHIKLMNPQRSTVWY"  # C2 mutated to S2
        res = self.ff_scorer.evaluate_food_functionality(wt_seq, mut_seq, mutation_pos_1indexed=2)
        self.assertTrue(res["cysteine_penalty_applied"], "Mutating a cysteine must incur the -50% structural penalty.")
        self.assertEqual(res["disulfide_score"], 50.0)

    def test_09_general_protein_scoring_honesty(self):
        wt_seq = "MVHLTPEEKSAVTALWGKVN"
        mut_seq = "MVHLTPEEKSAVTALWAKVN"  # G17 to A17
        im_res = self.im_scorer.evaluate_mutation_safety(wt_seq, mut_seq, 17, [], analysis_mode="GENERAL_PROTEIN")
        self.assertFalse(im_res["is_applicable"])
        self.assertEqual(im_res["medical_safety_index"], 50.0)

        af_res = self.af_scorer.evaluate_agronomic_fitness(wt_seq, mut_seq, 17, [], analysis_mode="GENERAL_PROTEIN")
        self.assertFalse(af_res["is_applicable"])
        self.assertEqual(af_res["agronomic_fitness_index"], 50.0)

    # -------------------------------------------------------------
    # 7. Pareto Optimization & Non-Dominated Sorting
    # -------------------------------------------------------------
    def test_10_pareto_dominance_and_knee_point(self):
        candidates = [
            {"medical_safety_index": 90.0, "food_functionality_index": 85.0, "agronomic_fitness_index": 80.0},
            {"medical_safety_index": 70.0, "food_functionality_index": 60.0, "agronomic_fitness_index": 50.0},  # Dominated
            {"medical_safety_index": 95.0, "food_functionality_index": 75.0, "agronomic_fitness_index": 85.0},  # Pareto Rank 1
        ]
        out = self.pareto_opt.compute_pareto_front(candidates, weight_safety=0.4, weight_food=0.35, weight_agronomic=0.25)
        self.assertEqual(out["pareto_count"], 2)
        self.assertFalse(out["all_candidates"][-1]["is_pareto"])
        self.assertIsNotNone(out["knee_point"])
        # Weights normalized sum to 1.0
        w = out["weights_used"]
        self.assertAlmostEqual(w["safety"] + w["food"] + w["agronomic"], 1.0, places=3)

    # -------------------------------------------------------------
    # 8. CRISPR sgRNA Engine & Authentic CDS
    # -------------------------------------------------------------
    def test_11_crispr_sgrna_authentic_cds_and_hexaploid_caveat(self):
        p18573 = self.uniprot_fetcher.fetch_protein("P18573")
        cds = p18573.get("cds_sequence")
        self.assertIsNotNone(cds, "P18573 must have canonical CDS from EMBL:X17361.1")
        self.assertEqual(len(cds), 921)

        crispr_res = self.crispr_engine.design_sgrnas_for_mutation(
            full_protein_sequence=p18573["sequence"],
            mutation_pos_1indexed=77,
            wildtype_aa="Q",
            mutant_aa="S",
            cds_sequence=cds
        )
        self.assertTrue(crispr_res["has_verified_cds"])
        self.assertIn("EMBL:X17361.1", crispr_res["dna_provenance"])
        self.assertIn("Hexaploid wheat", crispr_res["hexaploid_wheat_caveat"])

        # Check Doench Rule Set 2 score labeling
        guides = crispr_res["guides"]
        if guides:
            for g in guides:
                self.assertGreaterEqual(g["on_target_score"], 0.0)
                self.assertLessEqual(g["on_target_score"], 100.0)

    # -------------------------------------------------------------
    # 9. Bilingual Localization & RTL
    # -------------------------------------------------------------
    def test_12_bilingual_translations(self):
        ar_title = get_text("app_title", "ar")
        en_title = get_text("app_title", "en")
        self.assertIn("HypoGen-AI", ar_title)
        self.assertIn("HypoGen-AI", en_title)

        # RTL CSS
        rtl_css = get_rtl_css(is_ar=True)
        self.assertIn("direction: rtl;", rtl_css)
        self.assertIn(".seq-box", rtl_css)
        self.assertIn("direction: ltr !important;", rtl_css)


if __name__ == "__main__":
    unittest.main()
