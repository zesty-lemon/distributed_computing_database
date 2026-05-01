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

# ---------- Data Load ----------
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


if __name__ == "__main__":
    print(f"Loaded {NUM_ROWS} patients")
    print(get_clinic_data())
    print(get_hospital_data())
