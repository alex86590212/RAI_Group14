from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"

UCI_ID = 296
POSITIVE_LABEL = "<30"
GROUP_COL = "patient_nbr"
TARGET_COL = "is_under_30"
PRIMARY_ATTR = "race"
SECONDARY_ATTR = "gender"
NON_FEATURE_COLS = [TARGET_COL, "readmitted", GROUP_COL, "encounter_id"]

SEEDS = list(range(20))
TEST_SIZE = 0.2
VAL_SIZE = 0.2
N_BOOTSTRAP = 1000
