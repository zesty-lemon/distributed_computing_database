"""
Synthetic clinical-trial eligibility dataset generator.

Generates patient records for an early-Alzheimer's-disease trial
eligibility check that will be turned into ZKP circuits. Every field
is numeric, boolean, or bounded so that constraints can be expressed
as range / equality / threshold proofs.

Trial criteria implemented (CS3120 project proposal):

  Inclusion (all required):
    age in [55, 80]
    mmse in [20, 26]
    moca in [16, 25]
    cdr_global in [0.5, 1.0]
    amyloid_status == 1
    adl_score >= 60
    decline_6mo_flag == 1
    caregiver_available == 1

  Exclusion (any one disqualifies):
    stroke_history == 1
    major_depression_flag == 1
    microhemorrhage_count > 4
    cardio_risk_flag == 1                 (uncontrolled CV disease)
    egfr < 45
    alt > 3 * 56  (= 168 U/L)             (3x upper limit of normal)
    ast > 3 * 40  (= 120 U/L)
    other_neuro_disease == 1
    medication_exclusion_flag == 1

A patient is trial_inclusion == 1 iff all inclusion constraints hold
AND no exclusion constraint fires.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

ALT_ULN = 56  # U/L, upper limit of normal
AST_ULN = 40
ALT_EXCLUSION = 3 * ALT_ULN
AST_EXCLUSION = 3 * AST_ULN

# Vertical split (HW4-style cross-silo). patient_id is the join key and
# appears in both parties' files. trial_inclusion / trial_exclusion are
# oracle ground truth and stay out of both party files.
CLINIC_FIELDS = [
    "patient_id",
    "age",
    "mmse",
    "moca",
    "cdr_global",
    "adl_score",
    "iadl_score",
    "depression_score",
    "major_depression_flag",
    "decline_6mo_flag",
    "cognitive_decline_rate",
    "sleep_disruption_score",
    "gait_speed",
    "caregiver_available",
]

HOSPITAL_FIELDS = [
    "patient_id",
    "amyloid_status",
    "tau_status",
    "nfl_level",
    "apoe4_carrier",
    "hippocampal_volume_pct",
    "amyloid_pet_positive",
    "fdg_pet_pattern",
    "microhemorrhage_count",
    "egfr",
    "alt",
    "ast",
    "cardio_risk_flag",
    "stroke_history",
    "other_neuro_disease",
    "medication_exclusion_flag",
]


@dataclass
class Patient:
    patient_id: int

    # demographics
    age: int

    # cognitive
    mmse: int
    moca: int
    cdr_global: float

    # functional
    adl_score: int
    iadl_score: int

    # biomarkers
    amyloid_status: int
    tau_status: int
    nfl_level: float
    apoe4_carrier: int

    # imaging
    hippocampal_volume_pct: float
    amyloid_pet_positive: int
    fdg_pet_pattern: int  # 0 = normal, 1 = AD-typical, 2 = atypical

    # safety / labs
    microhemorrhage_count: int
    egfr: float
    alt: float
    ast: float
    cardio_risk_flag: int

    # comorbidities
    stroke_history: int
    depression_score: int  # PHQ-9 style 0..27
    major_depression_flag: int
    other_neuro_disease: int

    # longitudinal
    decline_6mo_flag: int
    cognitive_decline_rate: float  # MMSE points lost per year

    # behavioral / digital
    sleep_disruption_score: int  # 0..10
    gait_speed: float            # m/s

    # logistics
    caregiver_available: int
    medication_exclusion_flag: int

    # ZKP target outputs
    trial_inclusion: int
    trial_exclusion: int


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def sample_patient(rng: random.Random, pid: int, target_eligible: bool) -> Patient:
    """
    Sample a patient. If target_eligible is True, draw from distributions
    that mostly satisfy inclusion and avoid exclusion; otherwise draw
    broadly so many constraints will fail.
    """
    if target_eligible:
        age = rng.randint(55, 80)
        mmse = rng.randint(20, 26)
        moca = rng.randint(16, 25)
        cdr_global = rng.choice([0.5, 0.5, 1.0])
        amyloid_status = 1
        adl_score = rng.randint(60, 100)
        decline_6mo_flag = 1
        caregiver_available = 1
        # safety: keep within allowed bounds
        microhemorrhage_count = rng.randint(0, 4)
        egfr = round(rng.uniform(45, 110), 1)
        alt = round(rng.uniform(7, ALT_EXCLUSION - 5), 1)
        ast = round(rng.uniform(10, AST_EXCLUSION - 5), 1)
        cardio_risk_flag = 0
        stroke_history = 0
        depression_score = rng.randint(0, 14)  # below severe
        other_neuro_disease = 0
        medication_exclusion_flag = 0
    else:
        age = rng.randint(40, 95)
        mmse = rng.randint(0, 30)
        moca = rng.randint(0, 30)
        cdr_global = rng.choice([0.0, 0.5, 1.0, 2.0, 3.0])
        amyloid_status = rng.choices([0, 1], weights=[0.55, 0.45])[0]
        adl_score = rng.randint(20, 100)
        decline_6mo_flag = rng.choices([0, 1], weights=[0.5, 0.5])[0]
        caregiver_available = rng.choices([0, 1], weights=[0.3, 0.7])[0]
        microhemorrhage_count = rng.randint(0, 12)
        egfr = round(rng.uniform(20, 120), 1)
        alt = round(rng.uniform(7, 250), 1)
        ast = round(rng.uniform(10, 250), 1)
        cardio_risk_flag = rng.choices([0, 1], weights=[0.7, 0.3])[0]
        stroke_history = rng.choices([0, 1], weights=[0.85, 0.15])[0]
        depression_score = rng.randint(0, 27)
        other_neuro_disease = rng.choices([0, 1], weights=[0.9, 0.1])[0]
        medication_exclusion_flag = rng.choices([0, 1], weights=[0.85, 0.15])[0]

    # Fields not directly constrained by the trial but kept clinically plausible.
    iadl_score = clamp(adl_score - rng.randint(0, 20), 0, 100)

    # Tau and NfL correlate with amyloid + disease activity.
    tau_status = 1 if (amyloid_status == 1 and rng.random() < 0.75) else \
                 (1 if rng.random() < 0.15 else 0)
    nfl_level = round(rng.uniform(8, 25) + (10 if amyloid_status else 0), 2)
    apoe4_carrier = rng.choices([0, 1, 2], weights=[0.6, 0.32, 0.08])[0]

    # Imaging: more atrophy / AD-typical FDG pattern when amyloid+.
    base_vol = rng.uniform(60, 95)
    if amyloid_status:
        base_vol -= rng.uniform(5, 20)
    hippocampal_volume_pct = round(clamp(base_vol, 30, 100), 1)
    amyloid_pet_positive = amyloid_status if rng.random() < 0.95 else 1 - amyloid_status
    fdg_pet_pattern = rng.choices(
        [0, 1, 2],
        weights=[0.5, 0.4, 0.1] if amyloid_status else [0.8, 0.1, 0.1],
    )[0]

    # PHQ-9 >= 20 is severe depression -> exclusion.
    major_depression_flag = 1 if depression_score >= 20 else 0

    # Cognitive decline rate: faster if disease is active.
    cognitive_decline_rate = round(
        rng.uniform(0.5, 4.0) if decline_6mo_flag else rng.uniform(0.0, 1.0), 2
    )

    sleep_disruption_score = rng.randint(0, 10)
    gait_speed = round(rng.uniform(0.4, 1.4), 2)

    p = Patient(
        patient_id=pid,
        age=age,
        mmse=mmse,
        moca=moca,
        cdr_global=cdr_global,
        adl_score=adl_score,
        iadl_score=iadl_score,
        amyloid_status=amyloid_status,
        tau_status=tau_status,
        nfl_level=nfl_level,
        apoe4_carrier=apoe4_carrier,
        hippocampal_volume_pct=hippocampal_volume_pct,
        amyloid_pet_positive=amyloid_pet_positive,
        fdg_pet_pattern=fdg_pet_pattern,
        microhemorrhage_count=microhemorrhage_count,
        egfr=egfr,
        alt=alt,
        ast=ast,
        cardio_risk_flag=cardio_risk_flag,
        stroke_history=stroke_history,
        depression_score=depression_score,
        major_depression_flag=major_depression_flag,
        other_neuro_disease=other_neuro_disease,
        decline_6mo_flag=decline_6mo_flag,
        cognitive_decline_rate=cognitive_decline_rate,
        sleep_disruption_score=sleep_disruption_score,
        gait_speed=gait_speed,
        caregiver_available=caregiver_available,
        medication_exclusion_flag=medication_exclusion_flag,
        trial_inclusion=0,
        trial_exclusion=0,
    )
    incl, excl = evaluate_eligibility(p)
    p.trial_inclusion = incl
    p.trial_exclusion = excl
    return p


def evaluate_eligibility(p: Patient) -> tuple[int, int]:
    """Apply the trial's inclusion + exclusion rules."""
    inclusion_ok = (
        55 <= p.age <= 80
        and 20 <= p.mmse <= 26
        and 16 <= p.moca <= 25
        and 0.5 <= p.cdr_global <= 1.0
        and p.amyloid_status == 1
        and p.adl_score >= 60
        and p.decline_6mo_flag == 1
        and p.caregiver_available == 1
    )
    exclusion_hit = (
        p.stroke_history == 1
        or p.major_depression_flag == 1
        or p.microhemorrhage_count > 4
        or p.cardio_risk_flag == 1
        or p.egfr < 45
        or p.alt > ALT_EXCLUSION
        or p.ast > AST_EXCLUSION
        or p.other_neuro_disease == 1
        or p.medication_exclusion_flag == 1
    )
    trial_inclusion = 1 if (inclusion_ok and not exclusion_hit) else 0
    trial_exclusion = 1 if exclusion_hit else 0
    return trial_inclusion, trial_exclusion


def generate(n: int, eligible_fraction: float, seed: int) -> list[Patient]:
    rng = random.Random(seed)
    patients: list[Patient] = []
    for i in range(n):
        target = rng.random() < eligible_fraction
        patients.append(sample_patient(rng, pid=i + 1, target_eligible=target))
    return patients


def _project(patients: list[Patient], fields: list[str] | None) -> list[dict]:
    rows = [asdict(p) for p in patients]
    if fields is None:
        return rows
    return [{k: r[k] for k in fields} for r in rows]


def write_csv(
    patients: list[Patient], path: Path, fields: list[str] | None = None
) -> None:
    rows = _project(patients, fields)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(
    patients: list[Patient], path: Path, fields: list[str] | None = None
) -> None:
    rows = _project(patients, fields)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def summarize(patients: list[Patient]) -> dict:
    n = len(patients)
    incl = sum(p.trial_inclusion for p in patients)
    excl = sum(p.trial_exclusion for p in patients)
    return {
        "n_patients": n,
        "n_trial_inclusion": incl,
        "n_trial_exclusion": excl,
        "n_neither": n - incl - excl,
        "inclusion_rate": round(incl / n, 4) if n else 0.0,
        "exclusion_rate": round(excl / n, 4) if n else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--num-patients", type=int, default=1000)
    parser.add_argument(
        "--eligible-fraction",
        type=float,
        default=0.4,
        help="Approx fraction of patients drawn from the 'likely-eligible' "
             "distribution. Final inclusion rate will be lower due to noise.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent / "data",
    )
    parser.add_argument(
        "--basename",
        default="trial_patients",
        help="Base filename (without extension) for the output files.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    patients = generate(args.num_patients, args.eligible_fraction, args.seed)

    # Combined reference file — full record incl. oracle ground truth.
    combined_csv = args.out_dir / f"{args.basename}.csv"
    combined_jsonl = args.out_dir / f"{args.basename}.jsonl"
    write_csv(patients, combined_csv)
    write_jsonl(patients, combined_jsonl)

    # Vertical split — one party per file, joined by patient_id.
    clinic_csv = args.out_dir / "clinic_patients.csv"
    clinic_jsonl = args.out_dir / "clinic_patients.jsonl"
    hospital_csv = args.out_dir / "hospital_patients.csv"
    hospital_jsonl = args.out_dir / "hospital_patients.jsonl"
    write_csv(patients, clinic_csv, fields=CLINIC_FIELDS)
    write_jsonl(patients, clinic_jsonl, fields=CLINIC_FIELDS)
    write_csv(patients, hospital_csv, fields=HOSPITAL_FIELDS)
    write_jsonl(patients, hospital_jsonl, fields=HOSPITAL_FIELDS)

    stats = summarize(patients)
    stats_path = args.out_dir / f"{args.basename}_summary.json"
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)

    print(f"Wrote {len(patients)} patients to:")
    print(f"  {combined_csv}")
    print(f"  {combined_jsonl}")
    print(f"  {clinic_csv}  ({len(CLINIC_FIELDS)} cols)")
    print(f"  {clinic_jsonl}")
    print(f"  {hospital_csv}  ({len(HOSPITAL_FIELDS)} cols)")
    print(f"  {hospital_jsonl}")
    print(f"  {stats_path}")
    print("Summary:", json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
