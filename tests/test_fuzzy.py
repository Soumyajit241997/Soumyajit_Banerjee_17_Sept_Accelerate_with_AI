from core.fuzzy import best_match


def test_exact_match_case_insensitive():
    assert best_match("Orders", ["orders", "returns", "products"]) == "orders"


def test_substring_match():
    assert best_match("order_transactions", ["orders", "returns", "products"]) == "orders"
    assert best_match("sales_orders", ["orders", "returns", "products"]) == "orders"


def test_close_typo_match():
    assert best_match("orderz", ["orders", "returns", "products"]) == "orders"


def test_no_match_returns_none():
    assert best_match("inventory", ["orders", "returns", "products"], cutoff=0.6) is None


def test_empty_inputs():
    assert best_match("", ["orders"]) is None
    assert best_match("orders", []) is None
