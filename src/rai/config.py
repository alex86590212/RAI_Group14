from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"

UCI_ID = 296
POSITIVE_LABEL = "<30"
GROUP_COL = "patient_nbr"

SEEDS = list(range(20))
TEST_SIZE = 0.2
VAL_SIZE = 0.2
N_BOOTSTRAP = 1000
