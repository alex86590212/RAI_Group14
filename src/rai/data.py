import pandas as pd
from ucimlrepo import fetch_ucirepo

from rai.config import RAW_DIR, UCI_ID

CACHE_PATH = RAW_DIR / "diabetes_130.csv"


def load_raw(refresh: bool = False) -> pd.DataFrame:
    if CACHE_PATH.exists() and not refresh:
        return pd.read_csv(CACHE_PATH, low_memory=False)

    ds = fetch_ucirepo(id=UCI_ID)
    parts = [ds.data.ids, ds.data.features, ds.data.targets]
    df = pd.concat([p for p in parts if p is not None], axis=1)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_PATH, index=False)
    return df
