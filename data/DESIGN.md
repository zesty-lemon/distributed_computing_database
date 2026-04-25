# Design Doc: `generate_dataset.py`

**Project:** CS3120 — Secure Computation
**Authors:** Giles Lemmon, Nina Braddock
**Last updated:** 2026-04-25

---

## 1. Purpose

`generate_dataset.py` produces a synthetic patient cohort that drives our
ZKP performance comparison. The downstream system is a clinical-trial
eligibility check for early Alzheimer's disease: a patient (prover) wants
to convince a verifier they qualify for the trial **without revealing
their underlying medical record**.

The generator is the source of ground truth for that pipeline. It must:

1. Emit fields that map cleanly to ZKP-friendly primitives (numeric,
   boolean, bounded) — no free text, no clinician judgment.
2. Embed the trial's inclusion / exclusion rules in plain Python so we
   have a reference oracle to validate every ZKP backend against.
3. Produce a mix of *included*, *excluded*, and *borderline* patients so
   benchmarks measure real circuit behavior, not a single happy path.
4. Be deterministic given a seed so results are reproducible across the
   three (or more) ZKP libraries we benchmark.

Non-goals: clinical realism beyond what's needed for ZKP demonstration,
real PHI handling, statistically calibrated population frequencies.

---

## 2. Trial criteria (the spec being encoded)

Implemented from the project proposal:

**Inclusion (all required):**
- `age ∈ [55, 80]`
- `mmse ∈ [20, 26]`
- `moca ∈ [16, 25]`
- `cdr_global ∈ [0.5, 1.0]`
- `amyloid_status == 1`
- `adl_score ≥ 60`
- `decline_6mo_flag == 1`
- `caregiver_available == 1`

**Exclusion (any one disqualifies):**
- `stroke_history == 1`
- `major_depression_flag == 1` (derived from `depression_score ≥ 20`, PHQ-9 severe band)
- `microhemorrhage_count > 4`
- `cardio_risk_flag == 1`
- `egfr < 45`
- `alt > 168` (3× upper limit of normal, ULN = 56 U/L)
- `ast > 120` (3× ULN, ULN = 40 U/L)
- `other_neuro_disease == 1`
- `medication_exclusion_flag == 1`

A patient is `trial_inclusion == 1` iff every inclusion check passes
**and** no exclusion check fires. `trial_exclusion == 1` iff any
exclusion check fires (independent of inclusion).

These two outputs are what the ZKP ultimately proves.

---

## 3. Schema

One row per patient. All 30 input fields plus 2 target flags.

| Group | Field | Type | Range / domain | ZKP shape |
|---|---|---|---|---|
| ID | `patient_id` | int | ≥ 1 | n/a |
| Demographics | `age` | int | 40–95 | range proof |
| Cognitive | `mmse` | int | 0–30 | range proof |
| | `moca` | int | 0–30 | range proof |
| | `cdr_global` | float | {0.0, 0.5, 1.0, 2.0, 3.0} | equality / range |
| Functional | `adl_score` | int | 0–100 | threshold |
| | `iadl_score` | int | 0–100 | reference only |
| Biomarkers | `amyloid_status` | bool 0/1 | — | equality |
| | `tau_status` | bool 0/1 | — | reference |
| | `nfl_level` | float | ~8–35 | reference |
| | `apoe4_carrier` | int | 0/1/2 | reference |
| Imaging | `hippocampal_volume_pct` | float | 30–100 | reference |
| | `amyloid_pet_positive` | bool 0/1 | — | reference |
| | `fdg_pet_pattern` | int | 0/1/2 | reference |
| Safety / labs | `microhemorrhage_count` | int | 0–12 | threshold |
| | `egfr` | float | 20–120 | threshold |
| | `alt` | float | 7–250 | threshold |
| | `ast` | float | 10–250 | threshold |
| | `cardio_risk_flag` | bool 0/1 | — | equality |
| Comorbidities | `stroke_history` | bool 0/1 | — | equality |
| | `depression_score` | int | 0–27 (PHQ-9) | threshold (derived) |
| | `major_depression_flag` | bool 0/1 | derived: score ≥ 20 | equality |
| | `other_neuro_disease` | bool 0/1 | — | equality |
| Longitudinal | `decline_6mo_flag` | bool 0/1 | — | equality |
| | `cognitive_decline_rate` | float | 0–4 | reference |
| Behavioral | `sleep_disruption_score` | int | 0–10 | reference |
| | `gait_speed` | float | 0.4–1.4 m/s | reference |
| Logistics | `caregiver_available` | bool 0/1 | — | equality |
| | `medication_exclusion_flag` | bool 0/1 | — | equality |
| **Targets** | `trial_inclusion` | bool 0/1 | — | output |
| | `trial_exclusion` | bool 0/1 | — | output |

"Reference" fields don't gate the trial today but are kept so we can swap
in alternative criteria (e.g. require `apoe4_carrier > 0`) without
regenerating the schema.

---

## 4. Architecture

```
                ┌─────────────────────┐
                │ CLI args (argparse) │
                └─────────┬───────────┘
                          │  n, eligible_fraction, seed, out_dir
                          ▼
                  ┌────────────────┐
                  │   generate()   │  loops n times, picks regime per patient
                  └───────┬────────┘
                          ▼
              ┌──────────────────────────┐
              │  sample_patient(rng,…)   │  draws fields from one of two regimes
              └───────────┬──────────────┘
                          ▼
              ┌──────────────────────────┐
              │ evaluate_eligibility(p)  │  oracle: applies trial rules
              └───────────┬──────────────┘
                          ▼
              ┌──────────────────────────┐
              │ write_csv / write_jsonl  │  + summary JSON
              └──────────────────────────┘
```

Single file, no external dependencies — only Python stdlib (`random`,
`csv`, `json`, `argparse`, `dataclasses`, `pathlib`). Keeps the project
trivially portable to whatever ZKP toolchain we benchmark on.

### Key components

- **`Patient` dataclass** — frozen schema. Adding a field forces a
  compile-time-ish change in one place, and `asdict(p)` drives both CSV
  headers and JSONL keys, so the writers stay in sync automatically.
- **`sample_patient(rng, pid, target_eligible)`** — two-regime sampler.
  - `target_eligible=True`: every field is drawn inside its inclusion
    band and outside every exclusion band, so most of these patients
    will end up `trial_inclusion=1`. They are NOT all guaranteed
    eligible because correlated fields (e.g. `major_depression_flag`
    derived from `depression_score`) can still flip.
  - `target_eligible=False`: each field is drawn from a wide
    distribution that often violates at least one rule.
- **`evaluate_eligibility(p)`** — pure function that encodes the trial
  spec. This is the **reference oracle** every ZKP backend is tested
  against. If a ZK circuit ever disagrees with this function, the
  circuit is wrong.
- **Writers** — emit CSV (for spreadsheets / pandas), JSONL (one record
  per line, easy to stream into a circuit), and a summary JSON
  (cohort-level inclusion/exclusion counts).

---

## 5. Design decisions and tradeoffs

### Two-regime sampling instead of post-hoc rejection
We could have drawn every field from a single broad distribution and
accepted whatever inclusion rate fell out. Rejected because the
inclusion rate would be vanishingly small (8 conjoint constraints), so
benchmarks of "prove inclusion" would have almost no positive cases.
The two-regime approach lets us tune `--eligible-fraction` and still get
realistic noise (correlated derived fields cause some "likely eligible"
patients to fail).

### Inclusion rules are duplicated between sampler and oracle
The sampler knows the inclusion band for each field; the oracle
re-checks them. This duplication is intentional — the sampler is a
*hint*, the oracle is *truth*. If they diverged we want the dataset to
reflect the oracle, not the sampler.

### Derived fields stored as columns
`major_depression_flag` is computed from `depression_score`, but we
store both. ZKP circuits typically prefer to take the flag as a
witness rather than re-derive it inside the circuit, but having the
underlying score lets us also benchmark "prove threshold" circuits.

### Stdlib `random` rather than NumPy
NumPy would let us vectorize generation, but at 10⁴–10⁵ patients the
loop is well under a second and the no-dependency property is more
valuable for grading / reproducibility.

### Floats are rounded at write-time
`egfr`, `alt`, `ast`, `nfl_level`, `gait_speed`, etc. are rounded to 1–2
decimals before storage. Most ZKP libraries operate over integers /
field elements; downstream code can multiply by a fixed scale (e.g.
×10 for eGFR, ×100 for nfl) to get integers. Storing already-rounded
values prevents float-precision mismatches between regenerations.

---

## 6. CLI

```
python3 generate_dataset.py \
    -n 1000 \
    --eligible-fraction 0.4 \
    --seed 42 \
    --out-dir ./data \
    --basename trial_patients
```

| Flag | Default | Notes |
|---|---|---|
| `-n / --num-patients` | 1000 | Cohort size. |
| `--eligible-fraction` | 0.4 | P(patient drawn from likely-eligible regime). Final inclusion rate is lower because of correlated-field noise. |
| `--seed` | 42 | Determinism for benchmark reproducibility. |
| `--out-dir` | `./data` | Created if missing. |
| `--basename` | `trial_patients` | Used for `<basename>.csv`, `<basename>.jsonl`, `<basename>_summary.json`. |

---

## 7. Outputs

- `data/<basename>.csv` — header row + one row per patient.
- `data/<basename>.jsonl` — one JSON object per line, same fields as CSV.
- `data/<basename>_summary.json` — cohort counts:
  ```json
  {
    "n_patients": 1000,
    "n_trial_inclusion": 381,
    "n_trial_exclusion": 602,
    "n_neither": 17,
    "inclusion_rate": 0.381,
    "exclusion_rate": 0.602
  }
  ```

`n_neither` = patients who failed inclusion but didn't trip any
exclusion (e.g. amyloid-negative but otherwise healthy). Useful as
"hard negatives" for ZKP correctness tests.

---

## 8. How this connects to the ZKP benchmark

For each patient row, the experiment generates a proof of
`trial_inclusion == 1` (or `trial_exclusion == 1`) under several
backends — candidates: Groth16 / SNARK on a circom-style circuit,
Bulletproofs for the range checks, and an MPC-based comparator for the
threshold checks. Each backend reads the same JSONL file and is
required to agree with `evaluate_eligibility` on every row.

We measure, per backend:
- proving time
- verification time
- proof size
- circuit/setup size

The dataset is the *fixed input*; the proof system is the *variable*.
That's why the generator's determinism and oracle correctness are the
two properties we care about most.

---

## 9. Extension points

Likely future changes, listed so the file structure can absorb them
without a rewrite:

- **Adversarial test vectors.** Add a `--stress` mode that emits one
  patient per individual constraint failure (one patient who only
  fails `egfr`, one who only fails `mmse`, etc.). These are the unit
  tests for the ZKP circuits.
- **Time-series fields.** The proposal calls out longitudinal MMSE.
  Replace `mmse: int` with `mmse_history: list[int]` and have the
  oracle check the latest value; circuits would prove "exists an
  index i such that mmse_history[i] ∈ [20,26]".
- **Measurement provenance.** Add `device_id`, `measurement_timestamp`
  for data-attestation experiments (signed witnesses).
- **Alternate trial profiles.** Factor the criteria out of
  `evaluate_eligibility` into a config dict so we can run the same
  cohort against multiple trials.

---

## 10. Open questions

- Should `cdr_global` be stored as `cdr_global_x10` (int) up front to
  avoid the float-scaling step in every circuit? Leaning yes once we
  pick the ZKP backend.
- What inclusion rate is "right" for the benchmark? Currently ~38%;
  if proving time dominates we may want higher to amortize setup, if
  verification dominates we want a more even split.
