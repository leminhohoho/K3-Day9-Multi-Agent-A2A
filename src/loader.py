import json
import pandas as pd
from pathlib import Path
from src.config import DATA_DIR, INPUT_DIR


def load_all_data() -> dict[str, pd.DataFrame]:
    """Load all 9 CSV datasets into a dict of DataFrames."""
    data_path = Path(DATA_DIR)
    datasets = {
        "orders": pd.read_csv(data_path / "olist_orders_dataset.csv"),
        "items": pd.read_csv(data_path / "olist_order_items_dataset.csv"),
        "payments": pd.read_csv(data_path / "olist_order_payments_dataset.csv"),
        "customers": pd.read_csv(data_path / "olist_customers_dataset.csv"),
        "sellers": pd.read_csv(data_path / "olist_sellers_dataset.csv"),
        "products": pd.read_csv(data_path / "olist_products_dataset.csv"),
        "reviews": pd.read_csv(data_path / "olist_order_reviews_dataset.csv"),
        "geolocation": pd.read_csv(data_path / "olist_geolocation_dataset.csv"),
        "category_translation": pd.read_csv(data_path / "product_category_name_translation.csv"),
    }
    return datasets


def load_input_case(case_id: str) -> dict:
    """Load a single input case JSON."""
    path = Path(INPUT_DIR) / f"{case_id}.json"
    with open(path) as f:
        return json.load(f)


def list_case_ids() -> list[str]:
    """Return sorted list of case IDs (EC_001, EC_002, ...)."""
    paths = sorted(Path(INPUT_DIR).glob("EC_*.json"))
    return [p.stem for p in paths]
