"""
core/epitope_scanner.py: Dynamic & Universal Epitope Scanner with Mode Awareness.
Distinguishes Celiac Disease (HLA-DQ2.5/DQ8 T-cell enteropathy) from IgE-mediated food allergies.
Provides evidence level labeling:
- [RETRIEVED_REFERENCE] for curated databases & published literature.
- [COMPUTATIONAL_PREDICTION] for sliding-window physicochemical predictors.
Mode-aware:
- FOOD_ALLERGEN: calculates food allergen epitope density.
- GENERAL_PROTEIN: evaluates physicochemical surface antigenicity (antibody binding propensity)
  without assigning a false food allergen pathogenicity score.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np

# Calis et al. (IEDB) Amino Acid Immunogenicity Propensity Matrix
# Citation: Calis JJA, et al. (2013) Properties of MHC Class I Presented Peptides That Enhance Immunogenicity. PLoS Comput Biol 9(10): e1003266.
IEDB_AMINO_ACID_WEIGHTS: Dict[str, float] = {
    "W": 0.493, "F": 0.380, "I": 0.291, "L": 0.262, "Y": 0.258,
    "M": 0.157, "H": 0.142, "K": 0.115, "R": 0.108, "V": 0.084,
    "P": 0.045, "E": 0.012, "D": -0.003, "Q": -0.052, "C": -0.084,
    "T": -0.108, "N": -0.134, "A": -0.150, "S": -0.180, "G": -0.252
}

# Kolaskar & Tongaonkar Antigenicity Scale (1990)
# Citation: Kolaskar AS, Tongaonkar PC (1990) A semi-empirical method for prediction of antigenic determinants on protein antigens. FEBS Lett 276(1-2):172-174.
KOLASKAR_TONGAONKAR_SCALE: Dict[str, float] = {
    'A': 1.064, 'C': 1.412, 'D': 0.866, 'E': 0.851, 'F': 1.091,
    'G': 0.874, 'H': 1.105, 'I': 1.152, 'K': 0.930, 'L': 1.250,
    'M': 0.826, 'N': 0.776, 'P': 1.064, 'Q': 1.015, 'R': 0.873,
    'S': 1.012, 'T': 0.909, 'V': 1.383, 'W': 0.893, 'Y': 1.161
}

# Parker et al. (1986) Hydrophilicity Scale (Surface Accessibility)
# Citation: Parker JM, Guo D, Hodges RS (1986) New hydrophilicity scale derived from high-performance liquid chromatography. Biochemistry 25(19):5425-5432.
PARKER_HYDROPHILICITY_SCALE: Dict[str, float] = {
    'D': 3.0, 'E': 3.0, 'K': 3.0, 'R': 3.0, 'S': 1.2,
    'N': 0.2, 'Q': 0.2, 'P': 0.0, 'T': -0.4, 'G': -1.0,
    'H': -0.5, 'A': -0.5, 'M': -1.3, 'C': -1.0, 'V': -1.5,
    'Y': -1.9, 'I': -1.8, 'L': -1.8, 'F': -2.5, 'W': -3.4
}

# Curated Allergenic Epitopes with Verified Citations & Immunological Classification
CURATED_EPITOPES: List[Dict[str, Any]] = [
    # Celiac Disease (HLA-DQ2.5 / DQ8 T-Cell Mediated Enteropathy)
    {
        "id": "CELIAC-33MER",
        "name": "Celiac Alpha-Gliadin 33-mer locus",
        "motif": "LQLQPFPQPQLPYPQPQLPYPQPQLPYPQPQPF",
        "disease": "Celiac Disease (HLA-DQ2.5 T-cell enteropathy)",
        "immune_mechanism": "HLA-DQ2.5 Presentation (tTG2 Deamidation Dependent)",
        "allergen_type": "Gluten Prolamin",
        "risk_level": "Severe",
        "severity_weight": 5.0,
        "citation": "Shan L, et al. (2002) Science 297:2275-2279",
        "description": "Protease-resistant 33-mer peptide containing 6 overlapping HLA-DQ2.5 epitopes. Non-IgE T-cell driven enteropathy."
    },
    {
        "id": "DQ2.5-GLIA-A1A",
        "name": "DQ2.5-glia-alpha1a",
        "motif": "PFPQPQLPY",
        "disease": "Celiac Disease (HLA-DQ2.5 T-cell enteropathy)",
        "immune_mechanism": "HLA-DQ2.5 Presentation",
        "allergen_type": "Gluten Prolamin",
        "risk_level": "High",
        "severity_weight": 4.2,
        "citation": "Arentz-Hansen H, et al. (2000) J Exp Med 191:603-612",
        "description": "Canonical DQ2.5-restricted immunodominant alpha-gliadin epitope (deamidated at Q)."
    },
    {
        "id": "DQ2.5-GLIA-A2",
        "name": "DQ2.5-glia-alpha2",
        "motif": "PYPQPQLPY",
        "disease": "Celiac Disease (HLA-DQ2.5 T-cell enteropathy)",
        "immune_mechanism": "HLA-DQ2.5 Presentation",
        "allergen_type": "Gluten Prolamin",
        "risk_level": "High",
        "severity_weight": 4.2,
        "citation": "Sollid LM, et al. (2012) Immunogenetics 64:455-460",
        "description": "Overlapping DQ2.5-glia-alpha2 epitope with high mucosal T-cell receptor binding affinity."
    },
    {
        "id": "DQ2.5-GLIA-A1B",
        "name": "DQ2.5-glia-alpha1b",
        "motif": "PQPQLPYPQ",
        "disease": "Celiac Disease (HLA-DQ2.5 T-cell enteropathy)",
        "immune_mechanism": "HLA-DQ2.5 Presentation",
        "allergen_type": "Gluten Prolamin",
        "risk_level": "High",
        "severity_weight": 3.8,
        "citation": "Arentz-Hansen H, et al. (2000) J Exp Med 191:603-612",
        "description": "Overlapping DQ2.5 alpha-gliadin motif active in gut mucosal T-cell stimulation."
    },
    {
        "id": "DQ8-GLIA-A1",
        "name": "DQ8-glia-alpha1",
        "motif": "QGSFQPSQQ",
        "disease": "Celiac Disease (HLA-DQ8 T-cell enteropathy)",
        "immune_mechanism": "HLA-DQ8 Presentation",
        "allergen_type": "Gluten Prolamin",
        "risk_level": "Moderate",
        "severity_weight": 3.2,
        "citation": "van de Wal Y, et al. (1998) J Exp Med 187:475-482",
        "description": "HLA-DQ8 restricted alpha-gliadin T-cell stimulatory core motif."
    },
    # Peanut Ara h 2 (True IgE-Mediated Type I Hypersensitivity / Anaphylaxis)
    {
        "id": "ARAH2-EP3",
        "name": "Ara h 2 Epitope 3 (IgE immunodominant)",
        "motif": "HASARQQWEL",
        "disease": "Peanut Anaphylaxis (IgE-mediated Type I)",
        "immune_mechanism": "IgE Paratope Cross-Linking (Mast Cell Degranulation)",
        "allergen_type": "2S Albumin / Conglutin",
        "risk_level": "Severe",
        "severity_weight": 4.8,
        "citation": "Stanley JS, et al. (1997) Arch Biochem Biophys 342:244-253",
        "description": "N-terminal dominant linear IgE epitope associated with severe clinical peanut anaphylaxis."
    },
    {
        "id": "ARAH2-EP6-7",
        "name": "Ara h 2 Epitope 6/7 (IgE core)",
        "motif": "KRELRNLPQQCGLRAPQRCD",
        "disease": "Peanut Anaphylaxis (IgE-mediated Type I)",
        "immune_mechanism": "IgE Paratope Cross-Linking",
        "allergen_type": "2S Albumin / Conglutin",
        "risk_level": "Severe",
        "severity_weight": 5.0,
        "citation": "Stanley JS, et al. (1997) Arch Biochem Biophys 342:244-253",
        "description": "C-terminal immunodominant linear IgE-binding region containing critical basic residues."
    },
    {
        "id": "ARAH2-EP4",
        "name": "Ara h 2 Epitope 4 (IgE binding)",
        "motif": "QWELQGDRRC",
        "disease": "Peanut Anaphylaxis (IgE-mediated Type I)",
        "immune_mechanism": "IgE Paratope Cross-Linking",
        "allergen_type": "2S Albumin / Conglutin",
        "risk_level": "High",
        "severity_weight": 3.9,
        "citation": "Stanley JS, et al. (1997) Arch Biochem Biophys 342:244-253",
        "description": "Core linear IgE-binding peptide immediately downstream of Epitope 3."
    },
    # Bovine Milk Beta-lactoglobulin (Bos d 5) Linear IgE Epitopes
    {
        "id": "BLG-EP1",
        "name": "Beta-Lactoglobulin IgE Core 1",
        "motif": "AIVTQTMKGLDIQK",
        "disease": "Cow's Milk Anaphylaxis (IgE-mediated)",
        "immune_mechanism": "IgE Paratope Cross-Linking",
        "allergen_type": "Lipocalin",
        "risk_level": "High",
        "severity_weight": 3.7,
        "citation": "Ball G, et al. (1994) Clin Exp Allergy 24:758-764",
        "description": "N-terminal alpha-helix linear IgE binding region in Bos d 5."
    },
    {
        "id": "BLG-EP2",
        "name": "Beta-Lactoglobulin IgE Core 2",
        "motif": "VYVEELKPTPEGDLEILLQKW",
        "disease": "Cow's Milk Anaphylaxis (IgE-mediated)",
        "immune_mechanism": "IgE Paratope Cross-Linking",
        "allergen_type": "Lipocalin",
        "risk_level": "Severe",
        "severity_weight": 4.5,
        "citation": "Ball G, et al. (1994) Clin Exp Allergy 24:758-764",
        "description": "Dominant IgE epitope spanning beta-strands C and D with high patient serum reactivity."
    },
    # Soybean Major Allergens
    {
        "id": "SOY-GLYM4-EP1",
        "name": "Gly m 4 Bet v 1-homologous loop",
        "motif": "GVFNYEAET",
        "disease": "Soy Anaphylaxis / Pollen-Food Syndrome (IgE)",
        "immune_mechanism": "IgE Paratope Cross-Linking",
        "allergen_type": "PR-10 Protein",
        "risk_level": "High",
        "severity_weight": 3.8,
        "citation": "Kleine-Tebbe J, et al. (2002) J Allergy Clin Immunol 110:797-804",
        "description": "Cross-reactive IgE binding loop in soybean pathogenesis-related PR-10 allergen."
    },
    # Egg White Ovomucoid (Gal d 1)
    {
        "id": "EGG-GALD1-EP1",
        "name": "Gal d 1 Ovomucoid Domain 3 IgE Epitope",
        "motif": "VTEESCPE",
        "disease": "Egg White Anaphylaxis (IgE-mediated)",
        "immune_mechanism": "IgE Paratope Cross-Linking",
        "allergen_type": "Kazal-type Serine Protease Inhibitor",
        "risk_level": "Severe",
        "severity_weight": 4.7,
        "citation": "Besler M, et al. (1999) Internet Symp Food Allergens 1:137-146",
        "description": "Thermal-resistant dominant IgE epitope in ovomucoid responsible for persistent allergy."
    }
]


class EpitopeScanner:
    """Dynamic epitope scanner with clear mode awareness and scientific evidence provenance."""

    def __init__(self):
        self.curated_epitopes = CURATED_EPITOPES
        self.amino_acid_weights = IEDB_AMINO_ACID_WEIGHTS
        self.kt_scale = KOLASKAR_TONGAONKAR_SCALE
        self.parker_scale = PARKER_HYDROPHILICITY_SCALE

    def scan_sequence(
        self,
        sequence: str,
        uniprot_features: Optional[List[Dict[str, Any]]] = None,
        analysis_mode: str = "FOOD_ALLERGEN"
    ) -> Dict[str, Any]:
        """
        Executes multi-tiered epitope identification:
        - In FOOD_ALLERGEN mode: matches curated allergenic motifs and predicts surface epitopes.
        - In GENERAL_PROTEIN mode: runs physicochemical prediction for general B-cell antigenicity,
          and flags that this is a non-food protein.
        """
        clean_seq = sequence.upper().strip()
        seq_len = len(clean_seq)
        if seq_len == 0:
            return {
                "epitopes": [],
                "epitope_density_score": 0.0,
                "residue_scores": [],
                "analysis_mode": analysis_mode
            }

        detected_epitopes: List[Dict[str, Any]] = []
        residue_hit_mask = np.zeros(seq_len, dtype=float)
        residue_severity = np.zeros(seq_len, dtype=float)

        # -------------------------------------------------------------
        # 1. CURATED ALLERGEN DATABASE (Only active in FOOD_ALLERGEN mode)
        # -------------------------------------------------------------
        if analysis_mode == "FOOD_ALLERGEN":
            for item in self.curated_epitopes:
                motif = item["motif"]
                start = 0
                while True:
                    idx = clean_seq.find(motif, start)
                    if idx == -1:
                        break
                    end_idx = idx + len(motif)
                    detected_epitopes.append({
                        "id": item["id"],
                        "name": item["name"],
                        "start": idx + 1,
                        "end": end_idx,
                        "matched_sequence": clean_seq[idx:end_idx],
                        "target_motif": motif,
                        "match_type": "Exact",
                        "disease": item["disease"],
                        "immune_mechanism": item.get("immune_mechanism", "Immune Recognition"),
                        "allergen_type": item["allergen_type"],
                        "risk_level": item["risk_level"],
                        "severity_weight": item["severity_weight"],
                        "citation": item.get("citation", "Curated Reference"),
                        "evidence_level": "[RETRIEVED_REFERENCE]",
                        "description": item["description"],
                        "source": "Curated Allergen Database"
                    })
                    residue_hit_mask[idx:end_idx] = 1.0
                    residue_severity[idx:end_idx] = np.maximum(residue_severity[idx:end_idx], item["severity_weight"])
                    start = idx + 1

        # -------------------------------------------------------------
        # 2. UNIPROT CURATED IMMUNOLOGICAL FEATURE PARSING
        # -------------------------------------------------------------
        if uniprot_features:
            for feat in uniprot_features:
                desc = feat.get("description", "")
                desc_lower = desc.lower()
                f_type = feat.get("type", "")

                is_immuno = any(k in desc_lower for k in [
                    "epitope", "ige", "allergen", "antigen", "celiac", "binding site"
                ]) or (f_type in ["Region", "Site"] and any(k in desc_lower for k in ["ige", "epitope", "allergen"]))

                if is_immuno:
                    f_start = feat.get("start", 0)
                    f_end = feat.get("end", 0)
                    if 1 <= f_start <= f_end <= seq_len:
                        dup = any(abs(e["start"] - f_start) <= 2 and abs(e["end"] - f_end) <= 2 for e in detected_epitopes)
                        if not dup:
                            sev = 4.0 if ("ige" in desc_lower or "celiac" in desc_lower) else 3.2
                            detected_epitopes.append({
                                "id": f"UNIPROT_{f_type.upper()}_{f_start}",
                                "name": desc[:35] if desc else "UniProt Immunological Site",
                                "start": f_start,
                                "end": f_end,
                                "matched_sequence": clean_seq[f_start - 1 : f_end],
                                "target_motif": clean_seq[f_start - 1 : f_end],
                                "match_type": "UniProtKB Curated Feature",
                                "disease": "Allergic Sensitization / IgE" if analysis_mode == "FOOD_ALLERGEN" else "Immune Epitope",
                                "immune_mechanism": "Curated Annotation",
                                "allergen_type": "Curated Feature",
                                "risk_level": "High" if sev >= 4.0 else "Moderate",
                                "severity_weight": sev,
                                "citation": "UniProtKB Annotation",
                                "evidence_level": "[RETRIEVED_REFERENCE]",
                                "description": desc or "Curated immunological feature in UniProtKB entry.",
                                "source": "UniProtKB Record"
                            })
                            residue_hit_mask[f_start - 1 : f_end] = 1.0
                            residue_severity[f_start - 1 : f_end] = np.maximum(residue_severity[f_start - 1 : f_end], sev)

        # -------------------------------------------------------------
        # 3. PHYSICOCHEMICAL SLIDING-WINDOW PREDICTOR (Kolaskar & Parker)
        # -------------------------------------------------------------
        window_size = 7
        predicted_windows = []
        for i in range(0, seq_len - window_size + 1):
            sub = clean_seq[i : i + window_size]
            kt_val = np.mean([self.kt_scale.get(aa, 1.0) for aa in sub])
            parker_val = np.mean([self.parker_scale.get(aa, 0.0) for aa in sub])

            # Antigenic criteria: KT >= 1.02 and surface accessible (Parker >= 0.5)
            if kt_val >= 1.02 and parker_val >= 0.5:
                predicted_windows.append((i, i + window_size, kt_val, parker_val))

        # Merge overlapping clusters
        if predicted_windows:
            merged_clusters = []
            curr_start, curr_end, max_kt, max_pk = predicted_windows[0]

            for w_start, w_end, kt_val, pk_val in predicted_windows[1:]:
                if w_start <= curr_end:
                    curr_end = max(curr_end, w_end)
                    max_kt = max(max_kt, kt_val)
                    max_pk = max(max_pk, pk_val)
                else:
                    merged_clusters.append((curr_start, curr_end, max_kt, max_pk))
                    curr_start, curr_end, max_kt, max_pk = w_start, w_end, kt_val, pk_val
            merged_clusters.append((curr_start, curr_end, max_kt, max_pk))

            for c_start, c_end, c_kt, c_pk in merged_clusters:
                already_hit = any(e["start"] <= c_end + 1 and e["end"] >= c_start + 1 for e in detected_epitopes)
                if not already_hit and (c_end - c_start) >= 6:
                    sev = min(3.5, 2.0 + (c_kt - 1.0) * 8.0 + (c_pk * 0.3))
                    pred_disease = "Predicted B-Cell / IgE Epitope" if analysis_mode == "FOOD_ALLERGEN" else "Predicted Surface Antigenic Loop"
                    detected_epitopes.append({
                        "id": f"PRED_AG_{c_start + 1}",
                        "name": f"Antigenic Surface Loop ({c_start + 1}-{c_end})",
                        "start": c_start + 1,
                        "end": c_end,
                        "matched_sequence": clean_seq[c_start:c_end],
                        "target_motif": clean_seq[c_start:c_end],
                        "match_type": "Kolaskar-Parker Windowed Prediction",
                        "disease": pred_disease,
                        "immune_mechanism": "Physicochemical Surface Accessibility",
                        "allergen_type": "Solvent-Exposed Loop",
                        "risk_level": "Moderate" if sev >= 3.0 else "Low",
                        "severity_weight": round(float(sev), 2),
                        "citation": "Kolaskar & Tongaonkar (1990); Parker et al. (1986)",
                        "evidence_level": "[COMPUTATIONAL_PREDICTION]",
                        "description": f"Predicted solvent-accessible antigenic site (KT: {c_kt:.2f}, Hydrophilicity: {c_pk:.2f}).",
                        "source": "Physicochemical Predictor"
                    })
                    residue_hit_mask[c_start:c_end] = 1.0
                    residue_severity[c_start:c_end] = np.maximum(residue_severity[c_start:c_end], sev)

        # -------------------------------------------------------------
        # 4. RESIDUE-LEVEL PROFILE & EPITOPE DENSITY SCORE
        # -------------------------------------------------------------
        residue_scores: List[float] = []
        for i, aa in enumerate(clean_seq):
            base_iedb = self.amino_acid_weights.get(aa, 0.0)
            norm_iedb = max(0.05, (base_iedb + 0.3) / 0.8)
            sev_multiplier = 1.0 + (residue_severity[i] * 0.45) if residue_hit_mask[i] > 0 else 0.45
            final_res_score = min(1.0, norm_iedb * sev_multiplier)
            residue_scores.append(round(float(final_res_score), 4))

        total_hit_residues = float(np.sum(residue_hit_mask))
        weighted_hit_sum = float(np.sum(residue_severity))

        if analysis_mode == "FOOD_ALLERGEN":
            density_raw = (weighted_hit_sum / (seq_len * 4.5)) * 100.0
            if total_hit_residues > 0:
                coverage_pct = (total_hit_residues / seq_len) * 100.0
                epitope_density_score = min(100.0, max(round(coverage_pct * 0.8, 2), round(density_raw, 2)))
            else:
                epitope_density_score = 0.0
        else:
            # GENERAL_PROTEIN mode: surface antigenicity percentage (0-100), clearly disclaimed as non-food
            epitope_density_score = round((total_hit_residues / seq_len) * 100.0, 1)

        detected_epitopes.sort(key=lambda x: x["start"])

        return {
            "epitopes": detected_epitopes,
            "epitope_count": len(detected_epitopes),
            "epitope_residues_count": int(total_hit_residues),
            "sequence_length": seq_len,
            "epitope_density_score": epitope_density_score,
            "residue_scores": residue_scores,
            "analysis_mode": analysis_mode,
            "has_celiac_33mer": any(e["id"] == "CELIAC-33MER" for e in detected_epitopes),
            "has_peanut_arah2": any("ARAH2" in e["id"] for e in detected_epitopes)
        }
