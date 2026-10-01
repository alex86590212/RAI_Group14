import pandas as pd
from ucimlrepo import fetch_ucirepo

from rai.config import ROOT

CACHE_NAME = "diabetes_130.csv"


def load_raw(cfg: dict, refresh: bool = False) -> pd.DataFrame:
    cache = ROOT / cfg["data"]["raw_dir"] / CACHE_NAME
    if cache.exists() and not refresh:
        return pd.read_csv(cache, low_memory=False)

    ds = fetch_ucirepo(id=cfg["data"]["uci_id"])
    parts = [ds.data.ids, ds.data.features, ds.data.targets]
    df = pd.concat([p for p in parts if p is not None], axis=1)

    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df
