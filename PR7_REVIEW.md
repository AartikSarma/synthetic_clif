# PR #7 Review: Add missing CLIF 2.1.0 columns for clifpy compatibility

## Summary

This PR adds new columns required by the CLIF 2.1.0 schema to three generators (ADT, hospitalization, patient) and includes a pre-generated 10k hospitalization dataset. The intent is sound, but the PR has a **critical bug** that breaks the test suite and several design issues that should be addressed before merging.

## Critical: Test-Breaking Bug

### `_sample_discharge_category` — array size mismatch (`hospitalization.py:228`)

`DISCHARGE_CATEGORIES` has 17 entries. Removing "Expired" leaves 16 categories in `non_expired_cats`. However, the `weights` list only has **15 elements**. The slicing `weights[:len(non_expired_cats)]` evaluates to `weights[:16]`, which returns all 15 elements (since that's all there are). This results in 16 categories but only 15 probability weights, causing:

```
ValueError: a and p must have same size
```

This crashes **every** code path that generates a non-expired discharge, making the hospitalization generator completely non-functional. On master, all 18 patient/hospitalization tests pass; on this PR branch, 9 fail.

**Fix:** Add a 16th weight value for "Jail", or adjust the list to match.

## Major Issues

### 1. Admission type values diverge from mCIDE permissible values (`hospitalization.py:218-222`)

The old code sampled from mCIDE-defined admission types (`["Emergency", "Urgent", "Elective", "Newborn", "Trauma", "Other", "Unknown"]`) via `self.sample_category()`. The new code replaces these with a hardcoded list `["ed", "facility", "osh", "direct", "elective", "other"]`.

While the new values may match the CLIF 2.1.0 spec more closely, this bypasses the `MCIDELoader` entirely and breaks the contract between generators and the permissible value system. The mCIDE embedded values should be updated to match, or the test `test_admission_type_valid` (which validates against mCIDE) will fail once the discharge bug is fixed and tests can proceed further.

### 2. Discharge category values diverge from mCIDE permissible values (`hospitalization.py:192-212`)

Similarly, the new discharge categories (`"Skilled Nursing Facility (SNF)"`, `"Acute Inpatient Rehab Facility"`, etc.) don't match the existing mCIDE values (`"SNF"`, `"Rehab"`, etc.). The `test_discharge_category_valid` test will fail because it validates against mCIDE permissible values.

**Recommendation:** Update the mCIDE loader's embedded values to match the CLIF 2.1.0 spec, rather than bypassing it with hardcoded lists. This keeps the single-source-of-truth design.

### 3. `location_type = "general_icu"` used as placeholder for non-ICU locations (`adt.py:178-189`)

For ED, stepdown, and ward locations, `location_type` is set to `"general_icu"` with a comment saying "placeholder, schema requires a value." This is semantically incorrect — a ward location labeled as "general_icu" will produce misleading synthetic data. If the schema requires an ICU-type value even for non-ICU locations, this should be documented. Otherwise, use `None`/null or a more appropriate value.

### 4. ~240 MB of binary parquet files committed directly to git (`synth_clif_10k/`)

The PR adds 28 parquet files (plus duplicates without `clif_` prefix, totaling ~56 files) directly to the repository. The largest files:
- `clif_vitals.parquet`: 37 MB
- `clif_intake_output.parquet`: 27 MB
- `clif_patient_assessments.parquet`: 15 MB
- `clif_labs.parquet`: 13 MB

This will permanently bloat the git history. Large binary files should be:
- Managed via Git LFS, or
- Generated on demand via the CLI (`python -m synthetic_clif.generate`), or
- Hosted as a GitHub release artifact

Additionally, there appear to be **duplicate files** — every table exists both as `<table>.parquet` and `clif_<table>.parquet` with identical sizes (e.g., `adt.parquet` and `clif_adt.parquet` are both 690,221 bytes).

## Minor Issues

### 5. Patient DataFrame column reorder (`patient.py:111-118`)

The column order was changed (e.g., `birth_date` and `death_dttm` moved before demographic categories, `sex_category` moved after `ethnicity_category`). While this doesn't break functionality, it's an unnecessary diff that could confuse downstream consumers expecting a specific column order. If intentional (to match schema order), document why.

### 6. Language weights precision (`patient.py:46`)

The language weights sum to exactly 1.005, not 1.0. The normalization `weights /= weights.sum()` handles this, but it would be cleaner to provide weights that sum to 1.0 to begin with, for readability.

## What Works Well

- Adding `hospital_id`, `hospital_type`, `location_name`, `location_type`, and `language_category` is the right direction for CLIF 2.1.0 compliance
- The multi-hospital system simulation in the ADT generator is a reasonable approach
- The ICU subtype distribution (`LOCATION_TYPES` and weights) provides good variety
- Language category distribution roughly reflects US demographics
- Adding 5% missingness to `language_category` is realistic

## Recommendation

**Request changes.** The discharge weight bug makes the hospitalization generator non-functional. The mCIDE divergence needs a deliberate decision (update mCIDE or keep hardcoded), and the parquet files should not be committed directly to the repo.
