import pickle

def save_preprocessed_cache(cache_path: str, payload: dict):
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_preprocessed_cache(cache_path: str) -> dict:
    with open(cache_path, "rb") as f:
        return pickle.load(f)


