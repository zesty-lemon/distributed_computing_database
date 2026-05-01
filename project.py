# Useful imports and utility functions
from pathlib import Path

import galois
import pandas as pd
import pychor

GF = galois.GF(2**31 - 1)

p1 = pychor.Party('clinic')
p2 = pychor.Party('hospital')


@pychor.local_function
def share(x):
    s1 = GF.Random()
    s2 = GF(x) - s1
    return s1, s2

# ------------- Data Load -------------
# Vertical federation: same patients in both files, joined by patient_id.
# Clinic (p1) holds cognitive / functional / behavioral / logistics columns.
# Hospital (p2) holds biomarker / imaging / lab / comorbidity columns.
DATA_DIR = Path(__file__).parent / "data"

_clinic_df = pd.read_csv(DATA_DIR / "clinic_patients.csv")
_hospital_df = pd.read_csv(DATA_DIR / "hospital_patients.csv")

assert len(_clinic_df) == len(_hospital_df), \
    "ERROR: Clinic and hospital files must have the same number of rows"
assert (_clinic_df['patient_id'].values == _hospital_df['patient_id'].values).all(), \
    "ERROR: Rows must be aligned by patient_id between the two files"

NUM_ROWS = len(_clinic_df)


def get_clinic_data():
    """Located clinic-side dataframe (party p1)."""
    return p1.constant(_clinic_df)


def get_hospital_data():
    """Located hospital-side dataframe (party p2)."""
    return p2.constant(_hospital_df)

# ------------- Useful Definitions -------------
from dataclasses import dataclass

multiplication_triples = []

@dataclass
class SecInt:
    # s1 is p1's share of the value, and s2 is p2's share
    s1: galois.GF
    s2: galois.GF

    @classmethod
    def input(cls, val):
        """Secret share an input: p1 holds s1, and p2 holds s2"""
        s1, s2 = share(val).untup(2)
        if p1 in val.parties:
            s2.send(p1, p2)
            return SecInt(s1, s2)
        else:
            s1.send(p2, p1)
            return SecInt(s1, s2)

    def __add__(x, y):
        """Add two SecInt objects using local addition of shares"""
        return SecInt(x.s1 + y.s1, x.s2 + y.s2)

    def __mul__(x, y):
        """Multiply two SecInt objects using a triple"""
        triple = multiplication_triples.pop()
        r1, r2 = protocol_mult((x.s1, x.s2), (y.s1, y.s2), triple)
        return SecInt(r1, r2)

    def reveal(self):
        """Reveal the secret value by broadcast and reconstruction"""
        self.s1.send(p1, p2)
        self.s2.send(p2, p1)
        return self.s1 + self.s2

def functionality_gen_triple():
    Fgen = pychor.Party('Fgen')

    def deal_shares(x):
        s1, s2 = share(x).untup(2)
        s1.send(Fgen, p1)
        s2.send(Fgen, p2)
        return s1, s2

    # Step 1: generate a, b, c
    a = Fgen.constant(GF.Random())
    b = Fgen.constant(GF.Random())
    c = a * b

    # Step 2: secret share a, b, c
    a1, a2 = deal_shares(a)
    b1, b2 = deal_shares(b)
    c1, c2 = deal_shares(c)
    return (a1, a2), (b1, b2), (c1, c2)

def protocol_mult(x, y, triple):
    x1, x2 = x
    y1, y2 = y
    (a1, a2), (b1, b2), (c1, c2) = triple

    # Step 1. P1 computes d_1 = x_1 - a_1 and sends the result to P2
    d1 = x1 - a1
    d1.send(p1, p2)

    # Step 2. P2 computes d_2 = x_2 - a_2 and sends the result to P1
    d2 = x2 - a2
    d2.send(p2, p1)

    # Step 3: P1 and P2 both compute $d = d_1 + d_2 = x - a$
    d = d1 + d2

    # Step 4. P1 computes e_1 = y_1 - b_1 and sends the result to P2
    e1 = y1 - b1
    e1.send(p1, p2)

    # Step 5. P2 computes e_2 = y_2 - b_2 and sends the result to P1
    e2 = y2 - b2
    e2.send(p2, p1)

    # Step 6. P1 and P2 both compute $e = e_1 + e_2 = y - b$
    e = e1 + e2

    # Step 7. P1 computes r_1 = d*e + d*b_1 + e*a_1 + c_1
    r_1 = d * e + d * b1 + e * a1 + c1

    # Step 8. P2 computes r_2 = 0 + d*b_2 + e*a_2 + c_2
    r_2 = d * b2 + e * a2 + c2

    return r_1, r_2


def gen_triples(n):
    for _ in range(n):
        multiplication_triples.append(functionality_gen_triple())


# ------------- MPC Protocol (stubs) -------------
# Per-party local reduction of all clinic / hospital trial constraints
# down to a single 0/1. The MPC then ANDs the two bits with one Beaver
# triple per patient.

@pychor.local_function
def clinic_local_ok_at(df, i):
    """
    The Clinic Decides if df@i (the current patient) is a fit or not
    :param df: Clinics Data
    :param i: Patient Index
    :return:
    """
    # Get Clinic's Info
    age = int(df.at[i, 'age'])
    mmse = int(df.at[i, 'mmse'])
    moca = int(df.at[i, 'moca'])
    cdr_global = float(df.at[i, 'cdr_global'])
    adl_score = int(df.at[i, 'adl_score'])
    decline_6mo_flag = int(df.at[i, 'decline_6mo_flag'])
    caregiver_available = int(df.at[i, 'caregiver_available'])
    major_depression_flag = int(df.at[i, 'major_depression_flag'])

    # Decide if info is within allowable parameters
    age_allowed = 55 <= age <= 80
    mmse_allowed = 20 <= mmse <= 26
    moca_allowed = 16 <= moca <= 25
    cdr_global_allowed = 0.5 <= cdr_global <= 1.0
    adl_score_allowed = 60 <= adl_score
    decline_6mo_allowed = decline_6mo_flag == 1
    caregiver_available_allowed = caregiver_available == 1
    major_depression_allowed = major_depression_flag != 1 # DO NOT allow patients with depression

    if age_allowed and mmse_allowed and moca_allowed and cdr_global_allowed and adl_score_allowed and decline_6mo_allowed and caregiver_available_allowed and major_depression_allowed:
        return 1
    else:
        return 0


@pychor.local_function
def hospital_local_ok_at(df, i):
    """Reduce all hospital-side trial constraints for row i into a single 0/1.

    Returns 1 iff the patient passes the hospital-local inclusion check
    AND fails every hospital-local exclusion check. Located @{hospital}.

    Inclusion (hospital-local):
        amyloid_status == 1
    Exclusion (hospital-local):
        microhemorrhage_count > 4
        egfr < 45
        alt > 168                    # 3 * ULN
        ast > 120                    # 3 * ULN
        cardio_risk_flag == 1
        stroke_history == 1
        other_neuro_disease == 1
        medication_exclusion_flag == 1
    """
    raise NotImplementedError("TODO: implement hospital-side local boolean reduction")


def run_mpc_eligibility():
    """Run the MPC eligibility protocol over the full cohort.

    For each row: each party reduces its local constraints in plaintext,
    secret-shares the resulting bit, and the parties multiply the two
    shares (1 Beaver triple per row). Returns a list of revealed bits.
    """
    gen_triples(NUM_ROWS)

    clinic_data = get_clinic_data()
    hospital_data = get_hospital_data()

    revealed = []
    for i in range(NUM_ROWS):
        c_bit = clinic_local_ok_at(clinic_data, i)
        # h_bit = hospital_local_ok_at(hospital_data, i)
        c_sec = SecInt.input(c_bit)
        h_sec = SecInt.input(h_bit)
        out = c_sec * h_sec
        revealed.append(out.reveal())
    return revealed


# ------------- Test Runners -------------

def _load_oracle():
    """Ground-truth eligibility bits from the dataset generator."""
    combined = pd.read_csv(DATA_DIR / "trial_patients.csv")
    return combined['trial_inclusion'].values


def test_mpc_protocol():
    """Run the MPC protocol end-to-end and validate against the oracle."""
    print("\n=== MPC eligibility test ===")
    oracle = _load_oracle()
    try:
        with pychor.LocalBackend():
            revealed = run_mpc_eligibility()
            bits = [int(r.val) for r in revealed]
    except NotImplementedError as e:
        print(f"  [skip] {e}")
        return

    matches = sum(1 for r, t in zip(bits, oracle) if r == t)
    eligible = sum(bits)
    print(f"  matched oracle: {matches}/{NUM_ROWS}")
    print(f"  eligible (revealed): {eligible}/{NUM_ROWS}")
    assert matches == NUM_ROWS, "MPC output disagrees with the plaintext oracle"
    print("  OK")


if __name__ == "__main__":
    print(f"Loaded {NUM_ROWS} patients")
    print(f"  clinic columns:   {list(_clinic_df.columns)}")
    print(f"  hospital columns: {list(_hospital_df.columns)}")

    test_mpc_protocol()
