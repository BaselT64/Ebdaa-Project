"""
core/translations.py: Comprehensive Bilingual Localization (Arabic RTL / English LTR).
Maintains scientific rigor, clear evidence labeling, and BiDi-safe text formatting.
"""

from typing import Dict, Any

# Evidence Level Labels (Mandatory Scientific Categorization)
EVIDENCE_LABELS = {
    "RETRIEVED_REFERENCE": {
        "en": "[RETRIEVED_REFERENCE] Curated Database / Published Literature",
        "ar": "[بيانات مرجعية مسترجعة] قواعد بيانات موثقة / منشورات علمية محكمة"
    },
    "COMPUTATIONAL_PREDICTION": {
        "en": "[COMPUTATIONAL_PREDICTION] Algorithmic / Deep Learning Model Output",
        "ar": "[تنبؤ حوسبي] مخرجات نماذج الذكاء الاصطناعي وخوارزميات التعلم العميق"
    },
    "ILLUSTRATIVE_HEURISTIC": {
        "en": "[ILLUSTRATIVE_HEURISTIC] Simplified Sequence-Based Biophysical Proxy",
        "ar": "[نموذج استدلالي تقريبي] مؤشر فيزيوكيميائي تقريبي مبني على التسلسل"
    },
    "EXPERIMENTAL_MEASUREMENT": {
        "en": "[EXPERIMENTAL_MEASUREMENT] Empirically Validated In-Vitro / In-Vivo Assay",
        "ar": "[قياس تجريبي مخبري] قياسات معملية موثقة في المختبر أو في الكائن الحي"
    }
}

TRANSLATIONS: Dict[str, Dict[str, str]] = {
    # App Header & Title
    "app_title": {
        "en": "HypoGen-AI: De-Novo In-Silico Allergen Neutralization Engine",
        "ar": "HypoGen-AI: محرك تحييد مسببات الحساسية والموازنة الوظيفية الحاسوبية"
    },
    "app_subtitle": {
        "en": "Multi-Objective In-Silico Allergen Epitope Ablation & Functional Balancing for Crop Proteins",
        "ar": "استئصال الحبيبات المستضدية عبر الذكاء الاصطناعي والموازنة متعددة الأهداف لجودة الغذاء وحيوية المحصول"
    },
    "app_caption": {
        "en": "HypoGen-AI v2.6 | ISEF & Ibdaa Computational Biology Track. Powered by PyTorch, ESM-2, UniProtKB, AlphaFold DB & 3Dmol.js.",
        "ar": "HypoGen-AI الإصدار 2.6 | مسار الأحياء الحاسوبية والمعلوماتية الحيوية (إبداع / ISEF). مدعوم بنماذج ESM-2 وAlphaFold DB وUniProtKB."
    },

    # Sidebar
    "language_label": {
        "en": "Interface Language / لغة الواجهة",
        "ar": "لغة الواجهة / Interface Language"
    },
    "experiment_setup": {
        "en": "Experiment & Target Setup",
        "ar": "إعداد التجربة والبروتين المستهدف"
    },
    "target_source_label": {
        "en": "Select Target Sequence Source",
        "ar": "حدد مصدر التسلسل المستهدف"
    },
    "custom_accession_label": {
        "en": "Enter UniProtKB Accession",
        "ar": "أدخل رمز دخول البروتين (UniProt Accession)"
    },
    "fetch_rest_btn": {
        "en": "Fetch from UniProtKB REST",
        "ar": "جلب البيانات المباشرة من UniProt REST"
    },
    "custom_fasta_label": {
        "en": "Paste IUPAC Amino Acid FASTA Sequence",
        "ar": "الصق تسلسل الأحماض الأمينية القياسي (FASTA)"
    },
    "validate_fasta_btn": {
        "en": "Validate & Sanitize FASTA",
        "ar": "التحقق والتعقيم الأحيائي للتسلسل"
    },
    "weights_header": {
        "en": "Multi-Objective Optimization Weights",
        "ar": "أوزان الاستمثال متعدد الأهداف (Pareto Weights)"
    },
    "w_safety_label": {
        "en": "Medical Safety Weight (w₁)",
        "ar": "وزن الأمان الطبي وتقليل التحسس (w₁)"
    },
    "w_food_label": {
        "en": "Food Functionality Weight (w₂)",
        "ar": "وزن جودة الغذاء وخواص العجين (w₂)"
    },
    "w_agro_label": {
        "en": "Agronomic Fitness Weight (w₃)",
        "ar": "وزن حيوية الإنبات والنمو الزراعي (w₃)"
    },

    # Ribbon Cards
    "metric_ai_model": {
        "en": "Active AI Foundation Model",
        "ar": "نموذج الذكاء الاصطناعي النشط"
    },
    "metric_ai_sub": {
        "en": "Transformer Masked Marginal Likelihood Ratio (ΔLLR)",
        "ar": "نسبة احتمالية النموذج اللغوي المقنع (ΔLLR)"
    },
    "metric_protein_source": {
        "en": "Protein Sequence & Coordinate Space",
        "ar": "تسلسل البروتين ونظام الإحداثيات"
    },
    "metric_target_host": {
        "en": "Target Organism & Protein Family",
        "ar": "الكائن العائل وفصيلة البروتين"
    },
    "metric_pathogenicity": {
        "en": "Epitope Density / Antigenicity Index",
        "ar": "مؤشر كثافة الحبيبات المستضدية"
    },

    # Tabs
    "tab1_title": {
        "en": "🧬 Tab 1: Sequence Alignment, Coordinates & 3D Structure",
        "ar": "🧬 1. محاذاة التسلسل ونظام الإحداثيات والتركيب 3D"
    },
    "tab2_title": {
        "en": "⚡ Tab 2: AI Mutation Engine & Pareto Optimization",
        "ar": "⚡ 2. محرك طفرات ESM-2 واستمثال باريتو متعدد الأهداف"
    },
    "tab3_title": {
        "en": "✂️ Tab 3: CRISPR sgRNA Precision Delivery & Provenance Export",
        "ar": "✂️ 3. تصميم كريسبر الدقيق ومركز تصدير البيانات المخبرية"
    },

    # Tab 1 Specifics
    "tab1_header": {
        "en": "1. Sequence Verification, Coordinate Mapping & 3D Structure",
        "ar": "1. التحقق من التسلسل، محاذاة الإحداثيات، والتركيب الفراغي ثلاثي الأبعاد"
    },
    "coord_mapping_title": {
        "en": "Biological Coordinate Systems Alignment (Precursor vs Mature vs Peptide)",
        "ar": "محاذاة أنظمة الإحداثيات الحيوية (البروتين الكامل مقابل الناضج مقابل الببتيد)"
    },
    "wheat_celiac_provenance": {
        "en": "Wheat Alpha-Gliadin (P18573) Celiac 33-mer Provenance: Shan et al. (2002) Science 297:2275-2279.",
        "ar": "توثيق ببتيد 33-mer للسيلياك في ألفا غليادين القمح (P18573): Shan et al. (2002) Science 297:2275-2279."
    },
    "structure_3d_title": {
        "en": "Interactive 3D Protein Structure (AlphaFold DB)",
        "ar": "التركيب الفراغي ثلاثي الأبعاد (AlphaFold DB)"
    },
    "structure_3d_disclaimer": {
        "en": "Predicted 3D model retrieved from AlphaFold DB (DeepMind/EMBL-EBI). Red: Curated Epitopes | Green: Active Mutation Locus | Yellow: Conserved Disulfide Bridges.",
        "ar": "النموذج التنبؤي ثلاثي الأبعاد مسترجع من AlphaFold DB. الأحمر: حبيبات مستضدية موثقة | الأخضر: موقع الطفرة المستهدف | الأصفر: جسور كبريتيد محفوظة."
    },
    "linear_seq_title": {
        "en": "Full-Length Sequence & Locus Annotations",
        "ar": "التسلسل الكامل والتوصيف الموقعي للأحماض الأمينية"
    },
    "iedb_curve_title": {
        "en": "Residue-Level Immunogenicity Profile (Calis et al. IEDB Model)",
        "ar": "منحنى إمراضية الأحماض الأمينية (نموذج IEDB - Calis et al.)"
    },
    "epitope_registry_title": {
        "en": "Curated & Predicted Epitope Registry",
        "ar": "سجل الحبيبات المستضدية الموثقة والتنبؤية"
    },

    # Tab 2 Specifics
    "tab2_header": {
        "en": "2. AI Masked Marginal Mutagenesis & Multi-Objective Pareto Optimization",
        "ar": "2. طفرات الذكاء الاصطناعي المقنعة واستمثال باريتو متعدد الأهداف"
    },
    "tab2_intro": {
        "en": "Hugging Face ESM-2 evaluates masked sequence log-likelihood ratios (ΔLLR). Candidates are scored across Medical Safety, Food Functionality, and Agronomic Fitness, then ranked by non-dominated sorting.",
        "ar": "يقوم نموذج ESM-2 بحساب نسب لوغاريتم الاحتمالية المقنعة (ΔLLR). تُقيّم الطفرات عبر الأمان الطبي، جودة الغذاء، وحيوية المحصول، ثم تُفرز عبر جبهة باريتو غير المهيمنة."
    },
    "esm_explanation": {
        "en": "ESM-2 computes ΔLLR = log P(mutant|context) - log P(wildtype|context). Note: ΔLLR reflects evolutionary sequence fitness in homologous protein families, not direct experimental ΔΔG thermodynamic stability.",
        "ar": "يحسب ESM-2 الفرق اللوغاريتمي ΔLLR = log P(mutant) - log P(wt). تنبيه: تعكس هذه القيمة الملاءمة التطورية للتسلسل ولا تعني قياساً تجريبياً مباشراً للطاقة الحرة الديناميكية الحرارية ΔΔG."
    },
    "select_mutation_loci": {
        "en": "Select Target Residue Positions for ESM-2 Directed Evolution",
        "ar": "حدد مواقع الأحماض الأمينية المستهدفة لتطور الطفرات (ESM-2)"
    },
    "run_optimization_btn": {
        "en": "🚀 Run ESM-2 Inference & Pareto Optimization",
        "ar": "🚀 تشغيل استدلال ESM-2 واستمثال باريتو متعدد الأهداف"
    },
    "pareto_landscape_title": {
        "en": "Interactive Multi-Objective Pareto Frontier Landscape",
        "ar": "فضاء جبهة باريتو التفاعلي متعدد الأبعاد"
    },
    "knee_point_title": {
        "en": "Optimal Knee Point Candidate (Minimum Distance to Ideal Utopia Point)",
        "ar": "المرشح الأمثل عند نقطة الركبة (أقصر مسافة إقليدية للنقطة المثالية)"
    },
    "structure_retention_title": {
        "en": "3D Molecular Structure Comparison (Wild-Type vs. Engineered Variant)",
        "ar": "المقارنة الفراغية ثلاثية الأبعاد (البروتين البري مقابل المتغير المهندس)"
    },
    "structure_retention_disclaimer": {
        "en": "Structural Model Notice: Display shows the AlphaFold template backbone with in-silico point substitution. Detailed all-atom molecular dynamics (MD) relaxation or crystallography is required for experimental RMSD verification.",
        "ar": "تنبيه النمذجة الهيكلية: يُظهر العرض الهيكل الكربوني الأساسي لـ AlphaFold مع استبدال موضعي حاسوبي. يلزم إجراء محاكاة ديناميكا جزيئية شاملة (MD) أو حيود أشعة سينية للتأكيد المعملي لقيمة RMSD."
    },

    # Tab 3 Specifics
    "tab3_header": {
        "en": "3. Precision CRISPR-Cas9 sgRNA Design & Data Provenance Center",
        "ar": "3. تصميم مرشدات كريسبر SpCas9 ومركز تصدير البيانات المخبرية الموثقة"
    },
    "wheat_hexaploid_warning": {
        "en": "CRITICAL WHEAT GENOMICS NOTICE: Bread wheat (Triticum aestivum, AABBDD, 2n=6x=42) is allohexaploid and contains 30 to 50+ alpha-gliadin paralogs across chromosome group 6. Editing a single locus in-silico does not eliminate gluten toxicity without targeting conserved protospacers across all homoeologous loci (e.g. Sanchez-Leon et al. 2018 Plant Biotechnol J).",
        "ar": "تنبيه وراثي حرج لمحصول القمح: قمح الخبز (Triticum aestivum سداسي الصيغة الصبغية AABBDD) يحتوي على 30 إلى أكثر من 50 جيناً مناظراً للألفا غليادين موزعة على المجموعة الصبغية 6. تعديل موضع واحد حاسوبياً لا يضمن استئصال سمية الغلوتين بالكامل ما لم يتم استهداف مواقع بروتوسبيسر المحفوظة عبر كافة النسخ الصبغية (مثل دراسة Sanchez-Leon et al. 2018)."
    },
    "cds_source_notice": {
        "en": "Nucleotide Context: Targeting verified biological coding sequence (CDS reference: EMBL:X17361.1). Reverse-translation with plant codon usage is utilized only when verified genomic/cDNA is absent.",
        "ar": "السياق النيوكليوتيدي: يتم الاستهداف على التسلسل المشفر الحيوي الموثق (المرجع: EMBL:X17361.1). يُستخدم الترجمة العكسية بجدول كودونات المحاصيل فقط عند غياب التسلسل الوراثي الموثق."
    },
    "doench_score_label": {
        "en": "Predicted On-Target Cleavage Activity (Doench Rule Set 2 Heuristic Score, 0-100)",
        "ar": "نشاط القطع المتوقع على الهدف (درجة نموذج Doench Rule Set 2 الاستدلالية، 0-100)"
    },
    "doench_disclaimer": {
        "en": "Notice: Score reflects computational cleavage propensity, not final plant transformation or editing frequency.",
        "ar": "تنبيه: تعكس الدرجة كفاءة الارتباط والقطع الحوسبي، ولا تمثل النسبة المئوية النهائية للتحول الوراثي في النبات."
    },
    "export_center_title": {
        "en": "Data Export Center & Experimental Provenance CSV",
        "ar": "مركز تصدير البيانات وسجلات التوثيق المخبري (CSV)"
    },

    # General Protein Mode Messages
    "general_protein_mode_banner": {
        "en": "General Protein Mode Active: Non-food/general protein detected. Wheat gluten rheology and crop agronomic fitness are not applicable (N/A). Multi-objective Pareto focuses on Evolutionary Likelihood (ESM-2) vs. Physicochemical Conservation.",
        "ar": "نمط البروتين العام نشط: تم التعرف على بروتين غير غذائي. مؤشرات عجين الغلوتين وحيوية إنبات المحصول غير قابلة للتطبيق (N/A). يركز استمثال باريتو على الملاءمة التطورية (ESM-2) مقابل الحفظ الفيزيوكيميائي."
    }
}


def get_text(key: str, lang: str = "en") -> str:
    """Retrieves localized text with graceful fallback to English."""
    entry = TRANSLATIONS.get(key)
    if not entry:
        return key
    return entry.get(lang, entry.get("en", key))


def get_rtl_css(is_ar: bool) -> str:
    """Returns CSS styles configuring layout direction and BiDi hygiene."""
    if is_ar:
        return """
        <style>
            /* Arabic RTL Global Adjustments */
            .main .block-container {
                direction: rtl;
                text-align: right;
            }
            /* Preserve LTR for biological sequences, symbols, formulas, and accessions */
            .seq-box, .ltr-span, code, pre, .stPlotlyChart {
                direction: ltr !important;
                text-align: left !important;
                unicode-bidi: embed;
            }
            div[data-testid="stSidebar"] {
                direction: rtl;
                text-align: right;
            }
            .metric-card {
                text-align: right;
            }
        </style>
        """
    else:
        return """
        <style>
            .main .block-container {
                direction: ltr;
                text-align: left;
            }
        </style>
        """
