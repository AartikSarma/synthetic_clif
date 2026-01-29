"""mCIDE (minimum Common ICU Data Elements) loader and manager."""

import os
from functools import cached_property
from pathlib import Path
from typing import Optional
import urllib.request
import zipfile
import tempfile
import shutil

import pandas as pd


class MCIDELoader:
    """Load and manage mCIDE permissible values from CSV files.

    mCIDE defines the standardized category values used in CLIF tables.
    This class loads these values from local CSV files or can fetch them
    from the CLIF GitHub repository.

    Example:
        >>> mcide = MCIDELoader()
        >>> mcide.vital_categories
        ['temp_c', 'heart_rate', 'sbp', 'dbp', 'spo2', 'respiratory_rate', ...]
        >>> mcide.get_category('discharge')
        ['Home', 'SNF', 'Expired', 'Hospice', ...]
    """

    GITHUB_RAW_BASE = "https://raw.githubusercontent.com/clif-consortium/mCIDE/main"

    def __init__(self, mcide_dir: Optional[Path] = None):
        """Initialize the mCIDE loader.

        Args:
            mcide_dir: Path to directory containing mCIDE CSV files.
                      If None, uses bundled default or embedded values.
        """
        self.mcide_dir = mcide_dir
        self._cache: dict[str, pd.DataFrame] = {}
        self._embedded_values = self._get_embedded_values()

    def _default_mcide_dir(self) -> Path:
        """Return path to bundled mCIDE directory."""
        return Path(__file__).parent / "mcide"

    def _get_embedded_values(self) -> dict[str, list[str]]:
        """Return embedded mCIDE values for common categories.

        These values are embedded in the code for reliability when
        external CSV files are not available. Based on CLIF 2.1.0.
        """
        return {
            # Patient demographics
            "sex": ["Female", "Male", "Other", "Unknown"],
            "race": [
                "American Indian or Alaska Native",
                "Asian",
                "Black or African American",
                "Native Hawaiian or Other Pacific Islander",
                "White",
                "Other",
                "Unknown",
            ],
            "ethnicity": [
                "Hispanic or Latino",
                "Not Hispanic or Latino",
                "Unknown",
            ],
            # Hospitalization
            "admission_type": [
                "Emergency",
                "Urgent",
                "Elective",
                "Newborn",
                "Trauma",
                "Other",
                "Unknown",
            ],
            "discharge": [
                "Home",
                "SNF",
                "Expired",
                "Hospice",
                "LTACH",
                "Acute Care Hospital",
                "AMA",
                "Rehab",
                "Other",
                "Unknown",
            ],
            # ADT
            "location": [
                "ed",
                "ward",
                "stepdown",
                "icu",
                "procedural",
                "l&d",
                "hospice",
                "psych",
                "rehab",
                "radiology",
                "dialysis",
                "other",
            ],
            # Vitals
            "vital": [
                "temp_c",
                "heart_rate",
                "sbp",
                "dbp",
                "spo2",
                "respiratory_rate",
                "map",
                "height_cm",
                "weight_kg",
            ],
            "vital_meas_site": [
                "Arterial",
                "Venous",
                "Oral",
                "Rectal",
                "Axillary",
                "Tympanic",
                "Temporal",
                "Bladder",
                "Skin",
                "Other",
            ],
            # Respiratory support
            "respiratory_device": [
                "IMV",
                "NIPPV",
                "CPAP",
                "High Flow NC",
                "Face Mask",
                "Trach Collar",
                "Nasal Cannula",
                "Room Air",
                "Other",
            ],
            "respiratory_mode": [
                "Volume Control",
                "Pressure Control",
                "Pressure Support",
                "SIMV",
                "PRVC",
                "APRV",
                "CPAP",
                "BiPAP",
                "Other",
            ],
            # Labs (abbreviated - common categories)
            "lab": [
                "sodium",
                "potassium",
                "chloride",
                "bicarbonate",
                "bun",
                "creatinine",
                "glucose",
                "calcium",
                "magnesium",
                "phosphate",
                "albumin",
                "total_protein",
                "ast",
                "alt",
                "alkaline_phosphatase",
                "bilirubin_total",
                "bilirubin_direct",
                "lactate",
                "hemoglobin",
                "hematocrit",
                "wbc",
                "platelets",
                "inr",
                "pt",
                "ptt",
                "fibrinogen",
                "d_dimer",
                "troponin",
                "bnp",
                "crp",
                "procalcitonin",
                "ph",
                "pco2",
                "po2",
                "base_excess",
                "anion_gap",
            ],
            "lab_type": ["Routine", "STAT", "Timed", "Point of Care"],
            # Medications
            "medication": [
                "norepinephrine",
                "epinephrine",
                "vasopressin",
                "dopamine",
                "dobutamine",
                "phenylephrine",
                "milrinone",
                "propofol",
                "dexmedetomidine",
                "midazolam",
                "fentanyl",
                "morphine",
                "hydromorphone",
                "ketamine",
                "rocuronium",
                "cisatracurium",
                "vecuronium",
                "heparin",
                "insulin",
                "vancomycin",
                "piperacillin_tazobactam",
                "cefepime",
                "meropenem",
                "other",
            ],
            "med_route": [
                "IV",
                "PO",
                "IM",
                "SC",
                "Inhaled",
                "Topical",
                "PR",
                "SL",
                "Enteral",
                "Other",
            ],
            "mar_action": [
                "Given",
                "Held",
                "Refused",
                "Not Given",
                "Pending",
            ],
            # Microbiology
            "culture_fluid": [
                "Blood",
                "Urine",
                "Respiratory",
                "Wound",
                "CSF",
                "Peritoneal",
                "Pleural",
                "Other",
            ],
            "organism": [
                "Staphylococcus aureus",
                "MRSA",
                "Escherichia coli",
                "Klebsiella pneumoniae",
                "Pseudomonas aeruginosa",
                "Enterococcus faecalis",
                "Enterococcus faecium",
                "Candida albicans",
                "Candida glabrata",
                "Acinetobacter baumannii",
                "Streptococcus pneumoniae",
                "Enterobacter cloacae",
                "Proteus mirabilis",
                "Other",
                "No Growth",
            ],
            "organism_group": [
                "Gram Positive",
                "Gram Negative",
                "Fungal",
                "Viral",
                "Other",
            ],
            "antibiotic": [
                "Vancomycin",
                "Ceftriaxone",
                "Cefepime",
                "Meropenem",
                "Piperacillin-Tazobactam",
                "Ciprofloxacin",
                "Levofloxacin",
                "Gentamicin",
                "Ampicillin",
                "Oxacillin",
                "Daptomycin",
                "Linezolid",
                "Other",
            ],
            "susceptibility": [
                "Susceptible",
                "Intermediate",
                "Resistant",
            ],
            # Assessments
            "assessment": [
                "gcs_total",
                "gcs_eye",
                "gcs_verbal",
                "gcs_motor",
                "rass",
                "cam_icu",
                "pain_score",
                "braden_score",
                "morse_fall_risk",
                "apache_ii",
                "sofa",
                "other",
            ],
            # Procedures
            "procedure": [
                "Central Line Insertion",
                "Arterial Line Insertion",
                "Intubation",
                "Extubation",
                "Bronchoscopy",
                "Chest Tube Insertion",
                "Paracentesis",
                "Thoracentesis",
                "Lumbar Puncture",
                "Dialysis Catheter Insertion",
                "PICC Line Insertion",
                "Tracheostomy",
                "Other",
            ],
            # Diagnosis
            "poa": ["Yes", "No", "Unknown", "Exempt"],
            # Code status
            "code_status": [
                "Full Code",
                "DNR",
                "DNI",
                "DNR/DNI",
                "Comfort Care",
                "Unknown",
            ],
            # Position
            "position": [
                "Supine",
                "Prone",
                "Left Lateral",
                "Right Lateral",
                "Semi-Fowler",
                "High Fowler",
                "Trendelenburg",
                "Reverse Trendelenburg",
                "Other",
            ],
            # CRRT
            "crrt_mode": ["CVVH", "CVVHD", "CVVHDF", "SCUF", "Other"],
            # Concept tables
            "ecmo_device": ["ECMO", "VAD", "IABP", "Impella", "Other"],
            "ecmo_config": ["VV", "VA", "VAV", "Other"],
            "io": [
                "IV Fluids",
                "Blood Products",
                "Enteral Nutrition",
                "PO Intake",
                "Urine Output",
                "Drain Output",
                "Stool",
                "NG Output",
                "Other",
            ],
            "hemodynamic": [
                "CVP",
                "PA Systolic",
                "PA Diastolic",
                "PA Mean",
                "PCWP",
                "Cardiac Output",
                "Cardiac Index",
                "SVR",
                "PVR",
                "SvO2",
            ],
            "icu_order": [
                "PT Evaluation",
                "OT Evaluation",
                "Speech Evaluation",
                "Nutrition Consult",
                "Palliative Care Consult",
                "Social Work Consult",
                "Other",
            ],
            "micro_nonculture_test": [
                "COVID-19 PCR",
                "Influenza PCR",
                "RSV PCR",
                "Respiratory Viral Panel",
                "C. diff Toxin",
                "Procalcitonin",
                "Other",
            ],
            "micro_result": ["Positive", "Negative", "Indeterminate"],
            "provider_role": [
                "Attending",
                "Resident",
                "Fellow",
                "NP",
                "PA",
                "RN",
                "Pharmacist",
                "Respiratory Therapist",
                "Physical Therapist",
                "Other",
            ],
            "therapy": [
                "Physical Therapy",
                "Occupational Therapy",
                "Speech Therapy",
                "Respiratory Therapy",
                "Other",
            ],
            "blood_product": [
                "Packed RBCs",
                "Fresh Frozen Plasma",
                "Platelets",
                "Cryoprecipitate",
                "Whole Blood",
                "Other",
            ],
        }

    def get_category(self, category: str) -> list[str]:
        """Get permissible values for a category.

        Args:
            category: The mCIDE category name

        Returns:
            List of valid values for this category
        """
        # First try embedded values
        if category in self._embedded_values:
            return self._embedded_values[category].copy()

        # Try to load from CSV if directory specified
        if self.mcide_dir and self.mcide_dir.exists():
            df = self._load_category_csv(category)
            if df is not None and "category" in df.columns:
                return df["category"].dropna().tolist()

        # Return empty list if not found
        return []

    def _load_category_csv(self, category: str) -> Optional[pd.DataFrame]:
        """Load a category CSV file."""
        if category in self._cache:
            return self._cache[category]

        # Search for matching CSV file
        search_patterns = [
            f"**/clif_{category}_categories.csv",
            f"**/{category}_categories.csv",
            f"**/clif_{category}.csv",
            f"**/{category}.csv",
        ]

        for pattern in search_patterns:
            files = list(self.mcide_dir.glob(pattern))
            if files:
                try:
                    df = pd.read_csv(files[0])
                    self._cache[category] = df
                    return df
                except Exception:
                    pass

        return None

    def fetch_from_github(self, version: str = "main") -> Path:
        """Download mCIDE CSVs from the CLIF GitHub repository.

        Args:
            version: Git branch/tag to download from

        Returns:
            Path to downloaded mCIDE directory
        """
        url = f"https://github.com/clif-consortium/mCIDE/archive/refs/heads/{version}.zip"

        # Create destination directory
        dest_dir = self._default_mcide_dir()
        dest_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "mcide.zip"

            # Download zip file
            urllib.request.urlretrieve(url, zip_path)

            # Extract
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tmpdir)

            # Find extracted directory and copy contents
            extracted = list(Path(tmpdir).glob("mCIDE-*"))
            if extracted:
                for item in extracted[0].iterdir():
                    dest = dest_dir / item.name
                    if item.is_dir():
                        if dest.exists():
                            shutil.rmtree(dest)
                        shutil.copytree(item, dest)
                    else:
                        shutil.copy2(item, dest)

        # Update mcide_dir to use downloaded files
        self.mcide_dir = dest_dir
        self._cache.clear()

        return dest_dir

    # Convenience properties for common categories

    @cached_property
    def vital_categories(self) -> list[str]:
        """Vital sign categories."""
        return self.get_category("vital")

    @cached_property
    def lab_categories(self) -> list[str]:
        """Lab test categories."""
        return self.get_category("lab")

    @cached_property
    def device_categories(self) -> list[str]:
        """Respiratory device categories."""
        return self.get_category("respiratory_device")

    @cached_property
    def medication_categories(self) -> list[str]:
        """Medication categories."""
        return self.get_category("medication")

    @cached_property
    def location_categories(self) -> list[str]:
        """ADT location categories."""
        return self.get_category("location")

    @cached_property
    def discharge_categories(self) -> list[str]:
        """Discharge disposition categories."""
        return self.get_category("discharge")

    @cached_property
    def sex_categories(self) -> list[str]:
        """Sex categories."""
        return self.get_category("sex")

    @cached_property
    def race_categories(self) -> list[str]:
        """Race categories."""
        return self.get_category("race")

    @cached_property
    def ethnicity_categories(self) -> list[str]:
        """Ethnicity categories."""
        return self.get_category("ethnicity")

    def get_lab_reference_units(self) -> dict[str, str]:
        """Get reference units for lab categories."""
        return {
            "sodium": "mEq/L",
            "potassium": "mEq/L",
            "chloride": "mEq/L",
            "bicarbonate": "mEq/L",
            "bun": "mg/dL",
            "creatinine": "mg/dL",
            "glucose": "mg/dL",
            "calcium": "mg/dL",
            "magnesium": "mg/dL",
            "phosphate": "mg/dL",
            "albumin": "g/dL",
            "total_protein": "g/dL",
            "ast": "U/L",
            "alt": "U/L",
            "alkaline_phosphatase": "U/L",
            "bilirubin_total": "mg/dL",
            "bilirubin_direct": "mg/dL",
            "lactate": "mmol/L",
            "hemoglobin": "g/dL",
            "hematocrit": "%",
            "wbc": "K/uL",
            "platelets": "K/uL",
            "inr": "",
            "pt": "seconds",
            "ptt": "seconds",
            "fibrinogen": "mg/dL",
            "d_dimer": "ng/mL",
            "troponin": "ng/mL",
            "bnp": "pg/mL",
            "crp": "mg/L",
            "procalcitonin": "ng/mL",
            "ph": "",
            "pco2": "mmHg",
            "po2": "mmHg",
            "base_excess": "mEq/L",
            "anion_gap": "mEq/L",
        }

    def get_lab_reference_ranges(self) -> dict[str, tuple[float, float]]:
        """Get normal reference ranges for lab categories."""
        return {
            "sodium": (136, 145),
            "potassium": (3.5, 5.0),
            "chloride": (98, 106),
            "bicarbonate": (22, 28),
            "bun": (7, 20),
            "creatinine": (0.7, 1.3),
            "glucose": (70, 100),
            "calcium": (8.5, 10.5),
            "magnesium": (1.7, 2.3),
            "phosphate": (2.5, 4.5),
            "albumin": (3.5, 5.0),
            "total_protein": (6.0, 8.0),
            "ast": (10, 40),
            "alt": (10, 40),
            "alkaline_phosphatase": (40, 130),
            "bilirubin_total": (0.1, 1.2),
            "bilirubin_direct": (0, 0.3),
            "lactate": (0.5, 2.0),
            "hemoglobin": (12, 17),
            "hematocrit": (36, 50),
            "wbc": (4.5, 11.0),
            "platelets": (150, 400),
            "inr": (0.9, 1.1),
            "pt": (11, 14),
            "ptt": (25, 35),
            "fibrinogen": (200, 400),
            "d_dimer": (0, 500),
            "troponin": (0, 0.04),
            "bnp": (0, 100),
            "crp": (0, 10),
            "procalcitonin": (0, 0.5),
            "ph": (7.35, 7.45),
            "pco2": (35, 45),
            "po2": (80, 100),
            "base_excess": (-2, 2),
            "anion_gap": (8, 12),
        }
