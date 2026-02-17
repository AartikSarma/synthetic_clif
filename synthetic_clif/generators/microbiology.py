"""Microbiology culture and susceptibility generators."""

from datetime import datetime, timedelta, timezone
from typing import Optional
import uuid

import numpy as np
import pandas as pd

from synthetic_clif.generators.base import BaseGenerator
from synthetic_clif.config.mcide import MCIDELoader


class MicrobiologyCultureGenerator(BaseGenerator):
    """Generate synthetic microbiology culture data.

    Creates microbiology_culture table with:
    - ~30% of patients have positive cultures
    - Realistic organism distribution
    - Proper timestamp ordering: order < collect < result
    """

    # Organism probabilities by fluid type
    ORGANISM_BY_FLUID = {
        "Blood": {
            "organisms": [
                "Staphylococcus aureus",
                "MRSA",
                "Escherichia coli",
                "Klebsiella pneumoniae",
                "Enterococcus faecalis",
                "Candida albicans",
                "Pseudomonas aeruginosa",
                "No Growth",
            ],
            "weights": [0.12, 0.08, 0.15, 0.10, 0.08, 0.05, 0.07, 0.35],
        },
        "Urine": {
            "organisms": [
                "Escherichia coli",
                "Klebsiella pneumoniae",
                "Enterococcus faecalis",
                "Pseudomonas aeruginosa",
                "Proteus mirabilis",
                "Candida albicans",
                "No Growth",
            ],
            "weights": [0.30, 0.12, 0.10, 0.08, 0.08, 0.07, 0.25],
        },
        "Respiratory": {
            "organisms": [
                "Staphylococcus aureus",
                "MRSA",
                "Pseudomonas aeruginosa",
                "Klebsiella pneumoniae",
                "Acinetobacter baumannii",
                "Streptococcus pneumoniae",
                "No Growth",
            ],
            "weights": [0.15, 0.10, 0.18, 0.12, 0.08, 0.10, 0.27],
        },
        "Wound": {
            "organisms": [
                "Staphylococcus aureus",
                "MRSA",
                "Pseudomonas aeruginosa",
                "Escherichia coli",
                "Enterobacter cloacae",
                "No Growth",
            ],
            "weights": [0.20, 0.15, 0.15, 0.12, 0.08, 0.30],
        },
    }

    # Organism to group mapping
    ORGANISM_GROUPS = {
        "Staphylococcus aureus": "Gram Positive",
        "MRSA": "Gram Positive",
        "Escherichia coli": "Gram Negative",
        "Klebsiella pneumoniae": "Gram Negative",
        "Pseudomonas aeruginosa": "Gram Negative",
        "Enterococcus faecalis": "Gram Positive",
        "Enterococcus faecium": "Gram Positive",
        "Candida albicans": "Fungal",
        "Candida glabrata": "Fungal",
        "Acinetobacter baumannii": "Gram Negative",
        "Streptococcus pneumoniae": "Gram Positive",
        "Enterobacter cloacae": "Gram Negative",
        "Proteus mirabilis": "Gram Negative",
    }

    def generate(
        self,
        hospitalizations_df: pd.DataFrame,
        culture_rate: float = 0.5,
        positive_rate: float = 0.3,
    ) -> pd.DataFrame:
        """Generate microbiology culture data.

        Args:
            hospitalizations_df: Hospitalization table DataFrame
            culture_rate: Proportion of hospitalizations with cultures
            positive_rate: Proportion of cultures that are positive

        Returns:
            DataFrame with microbiology_culture columns
        """
        records = []

        for _, hosp in hospitalizations_df.iterrows():
            hosp_id = hosp["hospitalization_id"]
            admit_time = hosp["admission_dttm"]
            discharge_time = hosp["discharge_dttm"]

            if pd.isna(admit_time):
                continue

            if pd.isna(discharge_time):
                discharge_time = admit_time + timedelta(days=5)

            # Determine if this hospitalization has cultures
            if self.rng.random() > culture_rate:
                continue

            hosp_cultures = self._generate_hospitalization_cultures(
                hosp_id, admit_time, discharge_time, positive_rate
            )
            records.extend(hosp_cultures)

        df = pd.DataFrame(records)

        if len(df) > 0:
            df["order_dttm"] = pd.to_datetime(df["order_dttm"], utc=True)
            df["collect_dttm"] = pd.to_datetime(df["collect_dttm"], utc=True)
            df["result_dttm"] = pd.to_datetime(df["result_dttm"], utc=True)

        return df

    def _generate_hospitalization_cultures(
        self,
        hospitalization_id: str,
        admit_time: datetime,
        discharge_time: datetime,
        positive_rate: float,
    ) -> list[dict]:
        """Generate cultures for one hospitalization."""
        records = []
        los_hours = (discharge_time - admit_time).total_seconds() / 3600

        # Number of culture sets (1-4 depending on LOS)
        n_sets = min(4, max(1, int(los_hours / 48) + 1))

        for _ in range(n_sets):
            # Culture timing (usually early in admission or with fever)
            hours_from_admit = self.rng.uniform(0, min(72, los_hours))
            order_time = admit_time + timedelta(hours=hours_from_admit)

            if order_time >= discharge_time:
                continue

            # Sample fluid type
            fluid = self.rng.choice(
                ["Blood", "Urine", "Respiratory", "Wound"],
                p=[0.4, 0.3, 0.2, 0.1],
            )

            # Generate culture ID
            culture_id = str(uuid.uuid4())[:8]

            # Determine organism
            is_positive = self.rng.random() < positive_rate
            if is_positive:
                organism_data = self.ORGANISM_BY_FLUID.get(
                    fluid, self.ORGANISM_BY_FLUID["Blood"]
                )
                # Exclude "No Growth" for positive cultures
                organisms = [
                    o for o in organism_data["organisms"] if o != "No Growth"
                ]
                weights = organism_data["weights"][:-1]
                weights = np.array(weights) / sum(weights)
                organism = self.rng.choice(organisms, p=weights)
                organism_id = str(uuid.uuid4())[:8]
                organism_group = self.ORGANISM_GROUPS.get(organism, "Other")
            else:
                organism = "No Growth"
                organism_id = None
                organism_group = None

            # Generate timestamps
            collect_delay = int(self.rng.integers(15, 60))  # minutes
            result_delay = int(self.rng.integers(24, 72))  # hours for cultures

            collect_time = order_time + timedelta(minutes=collect_delay)
            result_time = collect_time + timedelta(hours=result_delay)

            # Map fluid to CLIF 2.1.0 schema values
            fluid_category_map = {
                "Blood": "blood_buffy",
                "Urine": "genito_urinary_tract",
                "Respiratory": "respiratory_tract",
                "Wound": "woundsite",
            }
            fluid_category = fluid_category_map.get(fluid, "other_unspecified")

            # Map organism to CLIF 2.1.0 schema values (lowercase with underscores)
            organism_category_map = {
                "Staphylococcus aureus": "staphylococcus_aureus",
                "MRSA": "staphylococcus_aureus",  # MRSA is S. aureus
                "Escherichia coli": "escherichia_coli",
                "Klebsiella pneumoniae": "klebsiella_pneumoniae",
                "Pseudomonas aeruginosa": "pseudomonas_aeruginosa",
                "Enterococcus faecalis": "enterococcus_faecalis",
                "Enterococcus faecium": "enterococcus_faecium",
                "Candida albicans": "candida_albicans",
                "Candida glabrata": "candida_glabrata",
                "Acinetobacter baumannii": "acinetobacter_baumannii",
                "Streptococcus pneumoniae": "streptococcus_pneumoniae",
                "Enterobacter cloacae": "enterobacter_cloacae",
                "Proteus mirabilis": "proteus_mirabilis",
                "No Growth": "no_growth",
            }
            organism_cat = organism_category_map.get(organism, "bacteria_other") if organism != "No Growth" else "no_growth"

            # Map organism_group to CLIF 2.1.0 schema values
            organism_group_map = {
                "Gram Positive": "staphylococcus_coag_pos",
                "Gram Negative": "escherichia",
                "Fungal": "candida_nos",
                "Other": "other_organism",
            }
            org_group = organism_group_map.get(organism_group, "other_organism") if organism_group else "no_growth"

            records.append(
                {
                    "patient_id": hospitalization_id[:8],  # Extract patient prefix from hosp_id
                    "hospitalization_id": hospitalization_id,
                    "organism_id": organism_id if organism_id else str(uuid.uuid4())[:8],
                    "order_dttm": order_time,
                    "collect_dttm": collect_time,
                    "result_dttm": result_time,
                    "fluid_category": fluid_category,
                    "method_category": "culture",  # Required per CLIF 2.1.0
                    "organism_category": organism_cat,
                    "organism_group": org_group,
                }
            )

        return records


class MicrobiologySusceptibilityGenerator(BaseGenerator):
    """Generate synthetic susceptibility data for positive cultures.

    Creates microbiology_susceptibility table with:
    - Linked by organism_id to cultures
    - Realistic resistance patterns
    """

    # Susceptibility patterns by organism
    SUSCEPTIBILITY_PATTERNS = {
        "Staphylococcus aureus": {
            "antibiotics": ["Oxacillin", "Vancomycin", "Daptomycin", "Linezolid"],
            "susceptible_rates": [0.7, 1.0, 0.98, 0.99],
        },
        "MRSA": {
            "antibiotics": ["Oxacillin", "Vancomycin", "Daptomycin", "Linezolid"],
            "susceptible_rates": [0.0, 0.99, 0.95, 0.98],
        },
        "Escherichia coli": {
            "antibiotics": [
                "Ampicillin",
                "Ceftriaxone",
                "Ciprofloxacin",
                "Meropenem",
                "Piperacillin-Tazobactam",
            ],
            "susceptible_rates": [0.5, 0.85, 0.75, 0.98, 0.90],
        },
        "Klebsiella pneumoniae": {
            "antibiotics": [
                "Ampicillin",
                "Ceftriaxone",
                "Ciprofloxacin",
                "Meropenem",
                "Piperacillin-Tazobactam",
            ],
            "susceptible_rates": [0.0, 0.80, 0.85, 0.95, 0.85],
        },
        "Pseudomonas aeruginosa": {
            "antibiotics": [
                "Cefepime",
                "Ciprofloxacin",
                "Meropenem",
                "Piperacillin-Tazobactam",
                "Gentamicin",
            ],
            "susceptible_rates": [0.85, 0.80, 0.85, 0.88, 0.90],
        },
        "Enterococcus faecalis": {
            "antibiotics": ["Ampicillin", "Vancomycin", "Daptomycin", "Linezolid"],
            "susceptible_rates": [0.95, 0.95, 0.98, 0.99],
        },
        "Enterococcus faecium": {
            "antibiotics": ["Ampicillin", "Vancomycin", "Daptomycin", "Linezolid"],
            "susceptible_rates": [0.1, 0.70, 0.95, 0.98],
        },
    }

    def generate(
        self,
        cultures_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate susceptibility data for positive cultures.

        Args:
            cultures_df: Microbiology culture table DataFrame

        Returns:
            DataFrame with microbiology_susceptibility columns
        """
        records = []

        # Filter to positive cultures with organism_id
        positive_cultures = cultures_df[cultures_df["organism_id"].notna()]

        for _, culture in positive_cultures.iterrows():
            organism_id = culture["organism_id"]
            organism = culture["organism_category"]

            if organism is None:
                continue

            pattern = self.SUSCEPTIBILITY_PATTERNS.get(organism)
            if pattern is None:
                continue

            # Generate susceptibility for each antibiotic
            for abx, sus_rate in zip(
                pattern["antibiotics"], pattern["susceptible_rates"]
            ):
                # Determine susceptibility
                if self.rng.random() < sus_rate:
                    susceptibility = "Susceptible"
                elif self.rng.random() < 0.3:
                    susceptibility = "Intermediate"
                else:
                    susceptibility = "Resistant"

                # Generate MIC value (optional)
                mic_value = None
                if self.rng.random() < 0.7:
                    if susceptibility == "Susceptible":
                        mic_value = f"<={self.rng.choice([0.5, 1, 2, 4])}"
                    elif susceptibility == "Intermediate":
                        mic_value = f"{self.rng.choice([4, 8, 16])}"
                    else:
                        mic_value = f">={self.rng.choice([16, 32, 64])}"

                # Map antibiotic to CLIF 2.1.0 schema antimicrobial_category values
                antimicrobial_map = {
                    "Oxacillin": "oxacillin",
                    "Vancomycin": "vancomycin",
                    "Daptomycin": "daptomycin",
                    "Linezolid": "linezolid",
                    "Ampicillin": "ampicillin",
                    "Ceftriaxone": "ceftriaxone",
                    "Ciprofloxacin": "ciprofloxacin",
                    "Meropenem": "meropenem",
                    "Piperacillin-Tazobactam": "piperacillin_tazobactam",
                    "Cefepime": "cefepime",
                    "Gentamicin": "gentamicin",
                    "Levofloxacin": "levofloxacin",
                }
                antimicrobial_category = antimicrobial_map.get(abx, abx.lower().replace("-", "_"))

                # Map susceptibility to CLIF 2.1.0 schema values
                # susceptibility_category: susceptible, non_susceptible, intermediate, NA
                susceptibility_map = {
                    "Susceptible": "susceptible",
                    "Intermediate": "intermediate",
                    "Resistant": "non_susceptible",
                }
                susceptibility_cat = susceptibility_map.get(susceptibility, "NA")

                records.append(
                    {
                        "organism_id": organism_id,
                        "antimicrobial_category": antimicrobial_category,
                        "susceptibility_category": susceptibility_cat,
                    }
                )

        return pd.DataFrame(records)
