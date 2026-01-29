# synthetic_clif

Generate synthetic [CLIF (Common Longitudinal ICU Format)](https://clif-consortium.github.io/website/) 2.1.0 datasets for testing and development.

## Purpose

This package creates realistic synthetic ICU data that follows the CLIF 2.1.0 specification. Because the data is entirely synthetic, it contains **no Protected Health Information (PHI)**, enabling:

- **Development on non-HIPAA compliant systems** - Use cloud IDEs, AI coding assistants (like Claude), and other tools that cannot access real patient data
- **Testing analysis pipelines** - Validate your CLIF analysis code before running on real data
- **Sharing reproducible examples** - Create datasets that can be freely shared for debugging and collaboration
- **CI/CD integration** - Run automated tests against synthetic data in any environment

## Pre-made Dataset

A pre-generated dataset with **10,000 hospitalizations** (~16 million rows across 28 tables) is available for download:

**[Download from GitHub Releases](https://github.com/AartikSarma/synthetic_clif/releases)** *(coming soon)*

Or generate your own using the instructions below.

## Installation

```bash
# Clone the repository
git clone https://github.com/AartikSarma/synthetic_clif.git
cd synthetic_clif

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
pip install -e .
```

## Generating Synthetic Data

### Quick Start

```bash
# Generate a small test dataset (10 patients, 12 hospitalizations)
python -m synthetic_clif --patients 10 --hospitalizations 12 --output data/test/

# Generate a larger dataset (8,000 patients, 10,000 hospitalizations)
python -m synthetic_clif --patients 8000 --hospitalizations 10000 --output data/full/
```

### Command Line Options

```
Usage: python -m synthetic_clif [OPTIONS]

Options:
  --patients INT          Number of patients to generate (default: 10)
  --hospitalizations INT  Number of hospitalizations to generate (default: 12)
  --output PATH           Output directory for parquet files (default: data/)
  --seed INT              Random seed for reproducibility (default: 42)
  --format [parquet|csv]  Output format (default: parquet)
  --no-concept-tables     Skip generating concept tables (draft status)
```

### Python API

```python
from synthetic_clif import SyntheticCLIFDataset
from pathlib import Path

# Create dataset generator
dataset = SyntheticCLIFDataset(
    n_patients=100,
    n_hospitalizations=120,
    seed=42
)

# Generate all tables
tables = dataset.generate()

# Access individual tables
vitals_df = tables["vitals"]
labs_df = tables["labs"]

# Write to parquet files
dataset.to_parquet(Path("output/"))
```

## Generated Tables

The package generates all 28 CLIF 2.1.0 tables:

### Beta Tables (16)
| Table | Description |
|-------|-------------|
| patient | Patient demographics |
| hospitalization | Hospital encounters |
| adt | Admission/discharge/transfer events |
| vitals | Vital signs (HR, BP, SpO2, temp, RR) |
| labs | Laboratory results (52 categories) |
| respiratory_support | Ventilator settings and oxygen delivery |
| medication_admin_continuous | IV infusions (vasopressors, sedatives) |
| medication_admin_intermittent | Scheduled and PRN medications |
| microbiology_culture | Culture results and organisms |
| microbiology_susceptibility | Antibiotic susceptibilities |
| patient_assessments | GCS, pain scores, sedation scales |
| patient_procedures | ICD-10-PCS and CPT procedures |
| hospital_diagnosis | ICD-10-CM diagnoses |
| code_status | DNR/DNI and comfort care orders |
| position | Patient positioning (prone/supine) |
| crrt_therapy | Continuous renal replacement therapy |

### Concept Tables (12)
| Table | Description |
|-------|-------------|
| clinical_trial | Trial enrollment |
| ecmo_mcs | ECMO and mechanical circulatory support |
| intake_output | Fluid balance |
| invasive_hemodynamics | PA catheter and arterial line data |
| key_icu_orders | PT/OT/Speech evaluations |
| medication_orders | Prescription orders |
| microbiology_nonculture | PCR and rapid diagnostics |
| patient_diagnosis | Problem list diagnoses |
| place_based_index | ADI and SVI geographic indices |
| provider | Care team assignments |
| therapy_details | PT/OT/Speech session details |
| transfusion | Blood product administration |

## Data Characteristics

The synthetic data includes realistic artifacts found in real EHR data:

- **Temporal autocorrelation** - Sequential vital signs follow physiologically plausible patterns
- **Inter-variable correlation** - Low SpO2 triggers respiratory support escalation; low MAP triggers vasopressors
- **Irregular measurement frequency** - ICU vitals ~hourly, ward vitals ~q4h
- **Missingness patterns** - MCAR, MAR, and MNAR based on clinical documentation practices
- **Outliers** - Physiologically plausible extremes (fever spikes, hypotensive episodes)
- **Variable length of stay** - Log-normal distribution (median ~5 days, range 1-60+)

## Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_vitals.py
```

## License

MIT

## Related Projects

- [CLIF Consortium](https://clif-consortium.github.io/website/) - The Common Longitudinal ICU Format specification
- [CLIF GitHub](https://github.com/clif-consortium) - Official CLIF tools and documentation
