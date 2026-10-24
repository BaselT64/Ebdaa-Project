"""
core/uniprot_fetcher.py: REST client pulling UniProt metadata, sequences, and annotations.
Supports live REST API queries, exact benchmark references with verified checksums,
and clear separation between Food Allergen Mode and General Protein Mode.
"""

from typing import Dict, Any, Optional, List
import requests
import re
import datetime
import hashlib
import logging

logger = logging.getLogger(__name__)

# Valid canonical 20 IUPAC amino acids
CANONICAL_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

# Verified Benchmark Protein Registry with exact full-length sequences, coordinates, and checksums
BENCHMARK_PROTEINS: Dict[str, Dict[str, Any]] = {
    "P18573": {
        "accession": "P18573",
        "entry_name": "GDA9_WHEAT",
        "gene_name": "Gli-alpha",
        "organism": "Triticum aestivum (Wheat)",
        "organism_id": 4565,
        "protein_name": "Alpha/beta-gliadin MM1",
        "sequence": (
            "MKTFLILALLAIVATTARIAVRVPVPQLQPQNPSQQQPQEQVPLVQQQQFPGQQQPFPPQQPYPQPQPFPSQQPY"
            "LQLQPFPQPQLPYPQPQLPYPQPQLPYPQPQPFRPQQPYPQSQPQYSQPQQPISQQQQQQQQQQQQKQQQQQQQQ"
            "ILQQILQQQLIPCRDVVLQQHSIAYGSSQVLQQSTYQLVQQLCCQQLWQIPEQSRCQAIHNVVHAIILHQQQQQQ"
            "QQQQQQPLSQVSFQQPQQQYPSGQGSFQPSQQNPQAQGSVQPQQLPQFEEIRNLALETLPAMCNVYIPPYCTIAP"
            "VGIFGTN"
        ),
        "length": 307,
        "mass_da": 35147,
        "crc64": "06C1858BD96F1E08",
        "md5": "FDE61BB72122490DEABFA341173C88DD",
        "signal_peptide_end": 20,
        "mature_start": 21,
        "analysis_mode": "FOOD_ALLERGEN",
        "allergen_class": "Gluten Prolamin (Celiac Disease & Wheat Allergy Trigger)",
        "has_celiac_33mer": True,
        "celiac_33mer_coords": (76, 108),
        "nucleotide_cds_reference": {
            "database": "EMBL",
            "accession": "X17361.1",
            "protein_id": "CAA35238.1",
            "cds_length_bp": 921,
            "molecule": "mRNA/cDNA",
            "verified_translation_match": True
        },
        "cds_sequence": (
            "ATGAAGACCTTTCTCATCCTTGCCCTCCTTGCTATTGTAGCAACCACCGCCAGAATTGCAGTTAGAGTTCCAGTGCCACAATTGCAGCCACAAAATCCATC"
            "TCAGCAACAACCACAAGAGCAAGTTCCATTGGTACAACAACAACAATTTCCAGGGCAGCAACAACCATTTCCACCACAACAGCCATATCCGCAGCCGCAAC"
            "CATTTCCATCACAACAACCATATCTGCAGCTGCAACCATTTCCGCAGCCGCAACTACCATATCCGCAGCCGCAACTACCATATCCGCAGCCGCAACTACCA"
            "TATCCGCAGCCGCAACCATTTCGACCACAACAACCATATCCACAATCGCAACCACAGTATTCGCAACCACAACAACCAATTTCGCAGCAGCAGCAGCAGCA"
            "ACAACAACAACAACAACAAAAACAACAACAACAACAACAACAACAGATCCTTCAACAAATTTTGCAACAACAACTGATTCCATGCAGGGATGTTGTATTGC"
            "AACAACACAGCATAGCGTATGGAAGCTCACAAGTTTTGCAACAAAGTACTTACCAGCTGGTGCAACAATTGTGTTGTCAGCAGCTGTGGCAGATCCCCGAG"
            "CAGTCGCGGTGCCAAGCCATCCACAATGTTGTTCATGCTATTATTCTGCATCAACAGCAACAACAACAACAACAACAACAACAACAACCGTTGAGCCAGGT"
            "CTCCTTCCAACAGCCTCAACAACAATATCCATCAGGCCAGGGCTCCTTCCAGCCATCTCAGCAAAACCCACAGGCCCAGGGCTCTGTCCAGCCTCAACAAC"
            "TGCCCCAGTTTGAGGAAATAAGGAACCTAGCGCTAGAGACGCTACCTGCAATGTGCAATGTCTATATCCCTCCATATTGCACCATTGCTCCAGTTGGCATC"
            "TTCGGTACTAAC"
        ),
        "features": [
            {"type": "Signal", "description": "Signal peptide", "start": 1, "end": 20},
            {"type": "Chain", "description": "Alpha/beta-gliadin MM1 mature chain", "start": 21, "end": 307},
            {"type": "Region", "description": "Immunodominant Celiac 33-mer peptide locus (Shan et al. 2002)", "start": 76, "end": 108},
            {"type": "Disulfide bond", "description": "Inter/Intrachain disulfide bridge", "start": 174, "end": 265},
            {"type": "Disulfide bond", "description": "Intrachain disulfide bridge", "start": 175, "end": 206},
            {"type": "Disulfide bond", "description": "Intrachain disulfide bridge", "start": 214, "end": 297},
            {"type": "Compositional bias", "description": "Poly-glutamine repeat region", "start": 118, "end": 135},
        ],
        "category": "Celiac Disease Prolamin Target",
        "data_source": "Verified Benchmark Reference (Local Cache)"
    },
    "Q6PSU2": {
        "accession": "Q6PSU2",
        "entry_name": "CONG7_ARAHY",
        "gene_name": "Ara h 2",
        "organism": "Arachis hypogaea (Peanut)",
        "organism_id": 3818,
        "protein_name": "Conglutin-7 (Major Peanut Allergen Ara h 2.0201)",
        "sequence": (
            "MAKLTILVALALFLLAAHASARQQWELQGDRRCQSQLERANLRPCEQHLMQKIQRDEDSYGRDPYSPSQDPYSPS"
            "QDPDRRDPYSPSPYDRRGAGSSQHQERCCNELNEFENNQRCMCEALQQIMENQSDRLQGRQQEQQFKRELRNLPQ"
            "QCGLRAPQRCDLEVESGGRDRY"
        ),
        "length": 172,
        "mass_da": 19890,
        "crc64": "B8BB91C8D8C143AB",
        "md5": "03E2815DA526691B3104ABE4B2690F5C",
        "signal_peptide_end": 21,
        "mature_start": 22,
        "analysis_mode": "FOOD_ALLERGEN",
        "allergen_class": "2S Albumin / Conglutin (Peanut Anaphylaxis Allergen)",
        "has_celiac_33mer": False,
        "celiac_33mer_coords": None,
        "nucleotide_cds_reference": {
            "database": "GenBank",
            "accession": "AY007229.1",
            "cds_length_bp": 516,
            "molecule": "mRNA/cDNA",
            "verified_translation_match": True
        },
        "features": [
            {"type": "Signal", "description": "Signal peptide", "start": 1, "end": 21},
            {"type": "Chain", "description": "Conglutin-7 (Ara h 2)", "start": 22, "end": 172},
            {"type": "Region", "description": "Immunodominant linear IgE epitope 3", "start": 20, "end": 29},
            {"type": "Region", "description": "Immunodominant linear IgE epitope 6/7", "start": 133, "end": 152},
            {"type": "Disulfide bond", "description": "Disulfide bridge 1", "start": 33, "end": 116},
            {"type": "Disulfide bond", "description": "Disulfide bridge 2", "start": 45, "end": 103},
            {"type": "Disulfide bond", "description": "Disulfide bridge 3", "start": 104, "end": 152},
            {"type": "Disulfide bond", "description": "Disulfide bridge 4", "start": 118, "end": 160},
        ],
        "category": "Peanut Anaphylaxis Allergen",
        "data_source": "Verified Benchmark Reference (Local Cache)"
    },
    "P02754": {
        "accession": "P02754",
        "entry_name": "LACB_BOVIN",
        "gene_name": "LGB",
        "organism": "Bos taurus (Bovine)",
        "organism_id": 9913,
        "protein_name": "Beta-lactoglobulin (Major Milk Allergen Bos d 5)",
        "sequence": (
            "MKCLLLALALTCGAQALIVTQTMKGLDIQKVAGTWYSLAMAASDISLLDAQSAPLRVYVEELKPTPEGDLEILLQ"
            "KWENGECAQKKIIAEKTKIPAVFKIDALNENKVLVLDTDYKKYLLFCMENSAEPEQSLACQCLVRTPEVDDEALE"
            "KFDKALKALPMHIRLSFNPTQLEEQCHI"
        ),
        "length": 178,
        "mass_da": 19883,
        "crc64": "225F10A78C63A6B2",
        "md5": "53418D25489EEF6015DCD5078C66BE35",
        "signal_peptide_end": 16,
        "mature_start": 17,
        "analysis_mode": "FOOD_ALLERGEN",
        "allergen_class": "Lipocalin (Cow Milk Anaphylaxis Allergen)",
        "has_celiac_33mer": False,
        "celiac_33mer_coords": None,
        "nucleotide_cds_reference": {
            "database": "EMBL",
            "accession": "X14710.1",
            "cds_length_bp": 534,
            "molecule": "mRNA/cDNA",
            "verified_translation_match": True
        },
        "features": [
            {"type": "Signal", "description": "Signal peptide", "start": 1, "end": 16},
            {"type": "Chain", "description": "Beta-lactoglobulin mature chain", "start": 17, "end": 178},
            {"type": "Disulfide bond", "description": "Disulfide bridge 1", "start": 82, "end": 176},
            {"type": "Disulfide bond", "description": "Disulfide bridge 2", "start": 122, "end": 135},
            {"type": "Region", "description": "Major linear IgE epitope cluster", "start": 57, "end": 77},
        ],
        "category": "Dairy Anaphylaxis Allergen",
        "data_source": "Verified Benchmark Reference (Local Cache)"
    },
    "P68871": {
        "accession": "P68871",
        "entry_name": "HBB_HUMAN",
        "gene_name": "HBB",
        "organism": "Homo sapiens (Human)",
        "organism_id": 9606,
        "protein_name": "Hemoglobin subunit beta",
        "sequence": (
            "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGL"
            "AHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH"
        ),
        "length": 147,
        "mass_da": 15998,
        "crc64": "A31F6D621C6556A1",
        "md5": "209D686939D0B8D1EA089368150140E2",
        "signal_peptide_end": None,
        "mature_start": 2,
        "analysis_mode": "GENERAL_PROTEIN",
        "allergen_class": "Non-Allergenic (Human Globular Oxygen Transport Protein)",
        "has_celiac_33mer": False,
        "celiac_33mer_coords": None,
        "nucleotide_cds_reference": {
            "database": "RefSeq",
            "accession": "NM_000518.5",
            "cds_length_bp": 444,
            "molecule": "mRNA/cDNA",
            "verified_translation_match": True
        },
        "features": [
            {"type": "Chain", "description": "Hemoglobin subunit beta", "start": 2, "end": 147},
            {"type": "Domain", "description": "Globin domain", "start": 3, "end": 147},
            {"type": "Binding site", "description": "Heme (iron) binding", "start": 64, "end": 64},
        ],
        "category": "General Non-Allergenic Control Protein",
        "data_source": "Verified Benchmark Reference (Local Cache)"
    }
}


def compute_crc64(sequence: str) -> str:
    """Calculates CRC64 checksum matching UniProt specifications."""
    table = []
    poly = 0xD800000000000000
    for i in range(256):
        part = i
        for _ in range(8):
            if part & 1:
                part = (part >> 1) ^ poly
            else:
                part >>= 1
        table.append(part)

    crc = 0
    for byte in sequence.encode("ascii", "ignore"):
        crc = table[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return f"{crc:016X}"


def classify_protein_mode(
    accession_or_dict: Any,
    protein_name: str = "",
    organism: str = "",
    features: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Classifies a protein into either:
    - 'FOOD_ALLERGEN' (gluten, seed storage prolamins, 2S albumins, Ara h, Bos d, Gal d, Gly m, etc.)
    - 'GENERAL_PROTEIN' (human proteins, hemoglobins, enzymes, structural non-food proteins)
    Supports passing a protein record dictionary or individual string arguments.
    """
    if isinstance(accession_or_dict, dict):
        d = accession_or_dict
        accession = str(d.get("accession", ""))
        protein_name = str(d.get("protein_name", ""))
        organism = str(d.get("organism", ""))
        features = d.get("features", []) or []
    else:
        accession = str(accession_or_dict)
        features = features or []

    text_corpus = f"{accession} {protein_name} {organism}".lower()
    
    # Explicit non-allergen exclusion
    if "homo sapiens" in text_corpus or "hemoglobin" in text_corpus or "myoglobin" in text_corpus:
        return "GENERAL_PROTEIN"

    # Known food allergen indicators
    allergen_keywords = [
        "gliadin", "glutenin", "gluten", "conglutin", "vicilin", "glycinin",
        "ara h", "bos d", "gal d", "gly m", "ana o", "cor a", "mal d",
        "lactoglobulin", "casein", "ovalbumin", "ovomucoid", "tropomyosin",
        "2s albumin", "7s globulin", "11s globulin", "allergen"
    ]
    if any(k in text_corpus for k in allergen_keywords):
        return "FOOD_ALLERGEN"

    # Check curated features
    for f in features:
        desc = f.get("description", "").lower()
        if any(k in desc for k in ["allergen", "celiac", "ige", "anaphylaxis"]):
            return "FOOD_ALLERGEN"

    return "GENERAL_PROTEIN"


class UniProtFetcher:
    """Production client for UniProtKB REST API with rigorous state handling and data classification."""

    BASE_URL = "https://rest.uniprot.org/uniprotkb"

    def __init__(self, timeout_seconds: int = 8):
        self.timeout = timeout_seconds

    def fetch_protein(self, identifier: str) -> Dict[str, Any]:
        """
        Fetch protein data by UniProt Accession.
        Attempts live REST call to rest.uniprot.org.
        If live query fails:
          - If identifier is a documented benchmark, falls back to the exact verified local reference.
          - Otherwise, raises an informative ValueError without fabricating results.
        """
        clean_id = identifier.strip().upper()
        if not clean_id:
            raise ValueError("UniProt accession cannot be empty.")

        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Try live REST API query first
        live_error = None
        try:
            record = self._query_rest_api(clean_id)
            record["retrieval_timestamp"] = now_iso
            record["data_source"] = "UniProtKB REST API (Live)"
            return record
        except Exception as e:
            live_error = e
            logger.warning(f"Live UniProt query failed for {clean_id}: {e}")

        # Benchmark fallback ONLY if identifier is specifically in verified benchmark registry
        if clean_id in BENCHMARK_PROTEINS:
            cached = BENCHMARK_PROTEINS[clean_id].copy()
            cached["retrieval_timestamp"] = now_iso
            cached["data_source"] = "Verified Benchmark Reference (Local Cache - Offline Mode)"
            cached["offline_fallback_reason"] = str(live_error)
            return cached

        # For arbitrary accessions, do NOT fabricate data
        raise ValueError(
            f"Could not retrieve UniProt entry '{clean_id}'. "
            f"Reason: {live_error}. Please verify the accession format (e.g. P18573, P68871, Q6PSU2)."
        )

    def _query_rest_api(self, accession: str) -> Dict[str, Any]:
        """Queries https://rest.uniprot.org/uniprotkb/{ID}.json and extracts structured fields."""
        url = f"{self.BASE_URL}/{accession}.json"
        headers = {"Accept": "application/json"}
        
        response = requests.get(url, headers=headers, timeout=self.timeout)
        if response.status_code == 404:
            raise ValueError(f"UniProt entry '{accession}' not found in UniProtKB (HTTP 404).")
        response.raise_for_status()

        data = response.json()
        return self._parse_uniprot_json(data, accession)

    def _parse_uniprot_json(self, data: Dict[str, Any], requested_id: str) -> Dict[str, Any]:
        """Extracts accession, gene name, organism, protein name, sequence, checksums, and curated features."""
        accession = data.get("primaryAccession", requested_id)
        entry_name = data.get("uniProtkbId", accession)
        
        # Protein Name
        protein_name = "Unknown Protein"
        desc = data.get("proteinDescription", {})
        if "recommendedName" in desc and "fullName" in desc["recommendedName"]:
            protein_name = desc["recommendedName"]["fullName"].get("value", protein_name)
        elif "submissionNames" in desc and len(desc["submissionNames"]) > 0:
            protein_name = desc["submissionNames"][0].get("fullName", {}).get("value", protein_name)

        # Gene Name
        gene_name = "N/A"
        genes = data.get("genes", [])
        if genes:
            gene_name = genes[0].get("geneName", {}).get("value") or \
                        genes[0].get("orderedLocusNames", [{}])[0].get("value") or "N/A"

        # Organism
        org_data = data.get("organism", {})
        sci_name = org_data.get("scientificName", "Unknown Organism")
        com_name = org_data.get("commonName")
        organism = f"{sci_name} ({com_name})" if com_name else sci_name
        organism_id = org_data.get("taxonId", 0)

        # Sequence & Checksums
        seq_data = data.get("sequence", {})
        raw_sequence = seq_data.get("value", "")
        clean_sequence = re.sub(r"\s+", "", raw_sequence).upper()
        mass_da = seq_data.get("molWeight", 0)
        crc64 = seq_data.get("crc64", compute_crc64(clean_sequence))
        md5 = seq_data.get("md5", hashlib.md5(clean_sequence.encode("ascii")).hexdigest().upper())

        # Features & Coordinate Markers
        features: List[Dict[str, Any]] = []
        signal_end = None
        for feat in data.get("features", []):
            feat_type = feat.get("type", "Feature")
            feat_desc = feat.get("description", "")
            loc = feat.get("location", {})
            start = loc.get("start", {}).get("value", 0)
            end = loc.get("end", {}).get("value", 0)
            if start and end:
                features.append({
                    "type": feat_type,
                    "description": feat_desc or feat_type,
                    "start": start,
                    "end": end
                })
                if feat_type == "Signal":
                    signal_end = end

        # Classification
        mode = classify_protein_mode(accession, protein_name, organism, features)

        # Celiac 33-mer Check
        from core.coordinate_mapper import CoordinateMapper
        mapping_33mer = CoordinateMapper.map_33mer_to_protein(clean_sequence)
        has_33mer = mapping_33mer.get("found", False)
        coords_33mer = (mapping_33mer.get("precursor_start"), mapping_33mer.get("precursor_end")) if has_33mer else None

        # Cross-reference CDS if available in UniProt
        cds_ref = None
        for xref in data.get("uniProtKBCrossReferences", []):
            if xref.get("database") in ["EMBL", "GenBank", "RefSeq"]:
                cds_ref = {
                    "database": xref.get("database"),
                    "accession": xref.get("id"),
                    "molecule": "mRNA/cDNA"
                }
                break
        if not cds_ref and accession in BENCHMARK_PROTEINS:
            cds_ref = BENCHMARK_PROTEINS[accession].get("nucleotide_cds_reference")

        cds_sequence = None
        if accession in BENCHMARK_PROTEINS and "cds_sequence" in BENCHMARK_PROTEINS[accession]:
            cds_sequence = BENCHMARK_PROTEINS[accession]["cds_sequence"]

        return {
            "accession": accession,
            "entry_name": entry_name,
            "gene_name": gene_name,
            "organism": organism,
            "organism_id": organism_id,
            "protein_name": protein_name,
            "sequence": clean_sequence,
            "length": len(clean_sequence),
            "mass_da": mass_da,
            "crc64": crc64,
            "md5": md5,
            "signal_peptide_end": signal_end,
            "mature_start": (signal_end + 1) if signal_end else 1,
            "features": features,
            "analysis_mode": mode,
            "is_food_allergen": (mode == "FOOD_ALLERGEN"),
            "has_celiac_33mer": has_33mer,
            "celiac_33mer_coords": coords_33mer,
            "nucleotide_cds_reference": cds_ref,
            "cds_sequence": cds_sequence,
            "category": "Food Allergen" if mode == "FOOD_ALLERGEN" else "General Protein"
        }

    @staticmethod
    def sanitize_fasta(raw_input: str) -> Dict[str, Any]:
        """
        Validates and sanitizes arbitrary raw FASTA or sequence strings according to standard IUPAC conventions.
        Returns header, sanitized sequence, and checksums.
        """
        lines = [line.strip() for line in raw_input.strip().splitlines() if line.strip()]
        if not lines:
            raise ValueError("Input sequence is empty.")

        header = "Custom_User_Sequence"
        seq_parts = []

        for line in lines:
            if line.startswith(">"):
                header = line[1:].strip()
            else:
                seq_parts.append(line)

        raw_seq = "".join(seq_parts).upper()
        raw_seq = re.sub(r"[\s0-9\-\*]", "", raw_seq)

        sanitized_chars = []
        invalid_chars = set()

        for char in raw_seq:
            if char in CANONICAL_AMINO_ACIDS:
                sanitized_chars.append(char)
            else:
                invalid_chars.add(char)

        sanitized_sequence = "".join(sanitized_chars)
        if len(sanitized_sequence) < 10:
            raise ValueError(
                f"Valid sequence too short ({len(sanitized_sequence)} residues). "
                f"A minimum of 10 canonical amino acids is required for biological scanning."
            )

        est_mass = len(sanitized_sequence) * 110
        crc64 = compute_crc64(sanitized_sequence)
        md5 = hashlib.md5(sanitized_sequence.encode("ascii")).hexdigest().upper()
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        mode = classify_protein_mode("CUSTOM_FASTA", header, "User Defined", [])

        from core.coordinate_mapper import CoordinateMapper
        mapping_33mer = CoordinateMapper.map_33mer_to_protein(sanitized_sequence)

        return {
            "accession": "CUSTOM_FASTA",
            "entry_name": header[:25],
            "gene_name": "Synthetic / De Novo",
            "organism": "User Defined",
            "organism_id": 0,
            "protein_name": header,
            "sequence": sanitized_sequence,
            "length": len(sanitized_sequence),
            "mass_da": est_mass,
            "crc64": crc64,
            "md5": md5,
            "signal_peptide_end": None,
            "mature_start": 1,
            "features": [],
            "analysis_mode": mode,
            "is_food_allergen": (mode == "FOOD_ALLERGEN"),
            "has_celiac_33mer": mapping_33mer.get("found", False),
            "celiac_33mer_coords": (mapping_33mer.get("precursor_start"), mapping_33mer.get("precursor_end")) if mapping_33mer.get("found") else None,
            "category": "User Supplied FASTA",
            "data_source": "User FASTA Input",
            "retrieval_timestamp": now_iso,
            "rejected_residues": list(invalid_chars)
        }
