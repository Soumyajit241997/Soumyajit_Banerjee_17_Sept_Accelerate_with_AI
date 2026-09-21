"""Generates a synthetic 6-month e-commerce dataset across THREE file formats:
orders.csv, returns.json, products.xlsx — deliberately split this way so the
demo exercises RADAR's multi-format landing zone.

Apparel and Electronics are seeded with a rising monthly return rate (sizing
and DOA-defect problems respectively) so the Reporter agent's forecast has a
real trend to project, while the other categories stay low and roughly flat.

Run with: python scripts/generate_sample_data.py
"""
from __future__ import annotations

import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import SAMPLE_DATA_DIR  # noqa: E402

random.seed(42)

MONTHS = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06"]

CATEGORY_CONFIG = {
    "Apparel": {
        "sub_categories": ["Men's Wear", "Women's Wear", "Kids Wear"],
        "price_range": (15, 90), "base_return_rate": 0.10, "monthly_increment": 0.035,
        "reasons": {"wrong_size": 0.55, "not_as_described": 0.20, "changed_mind": 0.20, "damaged_in_transit": 0.05},
    },
    "Electronics": {
        "sub_categories": ["Audio", "Mobile Accessories", "Wearables"],
        "price_range": (20, 250), "base_return_rate": 0.16, "monthly_increment": 0.020,
        "reasons": {"defective": 0.60, "not_as_described": 0.15, "changed_mind": 0.15, "damaged_in_transit": 0.10},
    },
    "Home & Kitchen": {
        "sub_categories": ["Cookware", "Storage", "Decor"],
        "price_range": (10, 120), "base_return_rate": 0.05, "monthly_increment": 0.002,
        "reasons": {"damaged_in_transit": 0.4, "not_as_described": 0.3, "changed_mind": 0.3},
    },
    "Beauty": {
        "sub_categories": ["Skincare", "Haircare", "Makeup"],
        "price_range": (8, 60), "base_return_rate": 0.04, "monthly_increment": 0.001,
        "reasons": {"not_as_described": 0.5, "changed_mind": 0.5},
    },
    "Sporting Goods": {
        "sub_categories": ["Fitness", "Outdoor", "Team Sports"],
        "price_range": (15, 150), "base_return_rate": 0.06, "monthly_increment": 0.003,
        "reasons": {"defective": 0.3, "changed_mind": 0.4, "not_as_described": 0.3},
    },
    "Books": {
        "sub_categories": ["Fiction", "Non-fiction", "Children"],
        "price_range": (6, 35), "base_return_rate": 0.02, "monthly_increment": 0.0005,
        "reasons": {"damaged_in_transit": 0.5, "not_as_described": 0.5},
    },
}

SUPPLIERS = ["Northwind Traders", "Globex Supply Co", "Acme Wholesale", "Meridian Goods", "Union Sourcing"]
CHANNELS = ["web", "mobile_app", "marketplace"]


def _weighted_choice(weights: dict[str, float]) -> str:
    names, probs = zip(*weights.items())
    return random.choices(names, weights=probs, k=1)[0]


def _random_day(month: str) -> date:
    year, mon = map(int, month.split("-"))
    day = random.randint(1, 28)
    return date(year, mon, day)


def generate_products() -> pd.DataFrame:
    rows = []
    product_seq = 1
    for category, cfg in CATEGORY_CONFIG.items():
        for sub_category in cfg["sub_categories"]:
            for _ in range(6):  # 6 SKUs per sub-category
                low, high = cfg["price_range"]
                price = round(random.uniform(low, high), 2)
                rows.append({
                    "product_id": f"P{product_seq:04d}",
                    "category": category,
                    "sub_category": sub_category,
                    "product_name": f"{sub_category} Item {product_seq}",
                    "cost_price": round(price * random.uniform(0.4, 0.6), 2),
                    "supplier": random.choice(SUPPLIERS),
                    "launch_date": (date(2023, 1, 1) + timedelta(days=random.randint(0, 700))).isoformat(),
                    "_unit_price": price,  # internal only, dropped before saving orders' price source
                })
                product_seq += 1
    return pd.DataFrame(rows)


def generate_orders_and_returns(products: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    order_rows = []
    return_rows = []
    order_seq = 1
    return_seq = 1

    for month_idx, month in enumerate(MONTHS):
        for category, cfg in CATEGORY_CONFIG.items():
            cat_products = products[products["category"] == category]
            n_orders = random.randint(70, 110)
            return_rate = min(cfg["base_return_rate"] + cfg["monthly_increment"] * month_idx, 0.85)

            for _ in range(n_orders):
                product = cat_products.sample(1).iloc[0]
                quantity = random.randint(1, 3)
                unit_price = product["_unit_price"]
                order_amount = round(unit_price * quantity, 2)
                order_date = _random_day(month)
                order_id = f"O{order_seq:06d}"

                order_rows.append({
                    "order_id": order_id,
                    "order_date": order_date.isoformat(),
                    "customer_id": f"C{random.randint(1, 500):04d}",
                    "product_id": product["product_id"],
                    "quantity": quantity,
                    "unit_price": unit_price,
                    "order_amount": order_amount,
                    "channel": random.choice(CHANNELS),
                })
                order_seq += 1

                if random.random() < return_rate:
                    return_date = order_date + timedelta(days=random.randint(3, 21))
                    refund_amount = round(order_amount * random.uniform(0.85, 1.0), 2)
                    return_rows.append({
                        "return_id": f"R{return_seq:06d}",
                        "order_id": order_id,
                        "return_date": return_date.isoformat(),
                        "return_reason": _weighted_choice(cfg["reasons"]),
                        "refund_amount": refund_amount,
                        "condition": random.choice(["unopened", "opened", "damaged"]),
                    })
                    return_seq += 1

    return pd.DataFrame(order_rows), return_rows


def main():
    SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    products = generate_products()
    orders, returns = generate_orders_and_returns(products)

    orders.to_csv(SAMPLE_DATA_DIR / "orders.csv", index=False)

    with open(SAMPLE_DATA_DIR / "returns.json", "w", encoding="utf-8") as f:
        json.dump(returns, f, indent=2)

    products.drop(columns=["_unit_price"]).to_excel(SAMPLE_DATA_DIR / "products.xlsx", index=False)

    print(f"orders.csv:    {len(orders)} rows")
    print(f"returns.json:  {len(returns)} rows")
    print(f"products.xlsx: {len(products)} rows")
    print(f"Overall return rate: {len(returns) / len(orders):.1%}")
    print(f"Written to {SAMPLE_DATA_DIR}")


if __name__ == "__main__":
    main()
