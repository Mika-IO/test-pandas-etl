import pytest
import pandas as pd
from unittest.mock import patch

from main import (
    parse_date,
    transform_customers,
    transform_orders,
    transform_payments,
    join_and_enrich,
    aggregate_daily_revenue_by_country,
    aggregate_customer_metrics,
    aggregate_country_metrics,
    main,
    EXCHANGE_RATES,
)


# ---------------------------------------------------------------------------
# Fixtures — reusable mock DataFrames
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_customers():
    """Raw customers with whitespace, duplicates, missing country, mixed casing."""
    return pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 1, 5],
        "name": [" Alice ", "Bob", " Charlie ", "Diana", "Alice Updated", "Eve"],
        "country": [" us ", "GB", "de", None, "US", "  "],
    })


@pytest.fixture
def clean_customers():
    """Already-transformed customers for use as a dependency in other tests."""
    return pd.DataFrame({
        "customer_id": ["1", "2", "3"],
        "name": ["Alice", "Bob", "Charlie"],
        "country": ["US", "GB", "DE"],
    })


@pytest.fixture
def raw_orders():
    """Raw orders with various date formats, invalid amounts, bad currencies."""
    return pd.DataFrame({
        "order_id": ["O1", "O2", "O3", "O4", "O5", "O6", "O7"],
        "customer_id": [1, 2, 3, 1, 99, 2, 3],
        "order_date": [
            "15/01/2023",       # dd/mm/yyyy
            "2023-01-16",       # yyyy-mm-dd
            "17-01-2023",       # dd-mm-yyyy
            "18/01/2023",       # dd/mm/yyyy
            "2023-01-19",       # valid date but invalid customer
            "invalid_date",     # bad date
            "20/01/2023",       # valid
        ],
        "amount": ["100.50", "200", "50", "abc", "300", "100", "75.25"],
        "currency": ["USD", "GBP", "EUR", "USD", "USD", "USD", " eur "],
    })


@pytest.fixture
def raw_payments():
    """Raw payments with mixed status casing, invalid amounts, bad currencies."""
    return pd.DataFrame({
        "payment_id": ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"],
        "order_id": ["O1", "O1", "O2", "O3", "O3", "O1", "O4", "O5"],
        "paid_amount": ["50.00", "50.50", "200", "25", "25", "bad", "75", "100"],
        "currency": ["USD", "USD", "GBP", "EUR", "EUR", "USD", "USD", "JPY"],
        "status": ["paid", "PAID", " Paid ", "PAID", "failed", "PAID", "PAID", "PAID"],
    })


@pytest.fixture
def enriched_df():
    """Pre-built enriched DataFrame for aggregation tests."""
    return pd.DataFrame({
        "order_id": ["O1", "O2", "O3", "O4", "O5", "O6"],
        "customer_id": ["1", "1", "2", "2", "3", "3"],
        "order_date": [
            "2023-01-01", "2023-01-01",
            "2023-01-01", "2023-01-02",
            "2023-01-02", "2023-01-02",
        ],
        "amount": [100.0, 200.0, 150.0, 50.0, 300.0, 75.0],
        "currency": ["USD", "USD", "GBP", "EUR", "USD", "EUR"],
        "country": ["US", "US", "GB", "GB", "DE", "DE"],
        "total_paid_usd": [100.0, 200.0, 195.0, 0.0, 300.0, 82.5],
    })


# ===========================================================================
# Tests — parse_date
# ===========================================================================

class TestParseDate:

    def test_dd_mm_yyyy_slash(self):
        result = parse_date("15/01/2023")
        assert result == pd.Timestamp("2023-01-15")

    def test_dd_mm_yyyy_dash(self):
        result = parse_date("15-01-2023")
        assert result == pd.Timestamp("2023-01-15")

    def test_yyyy_mm_dd_dash(self):
        result = parse_date("2023-01-15")
        assert result == pd.Timestamp("2023-01-15")

    def test_yyyy_mm_dd_slash(self):
        result = parse_date("2023/01/15")
        assert result == pd.Timestamp("2023-01-15")

    def test_invalid_date_returns_nat(self):
        result = parse_date("not_a_date")
        assert pd.isna(result)

    def test_none_returns_nat(self):
        result = parse_date(None)
        assert pd.isna(result)

    def test_empty_string_returns_nat(self):
        result = parse_date("")
        assert pd.isna(result)


# ===========================================================================
# Tests — transform_customers
# ===========================================================================

class TestTransformCustomers:

    def test_strips_whitespace(self, raw_customers):
        result = transform_customers(raw_customers)
        for col in result.select_dtypes(include=["object"]).columns:
            for val in result[col]:
                assert val == val.strip()

    def test_normalizes_country_to_uppercase(self, raw_customers):
        result = transform_customers(raw_customers)
        for country in result["country"]:
            assert country == country.upper()

    def test_drops_missing_country(self, raw_customers):
        result = transform_customers(raw_customers)
        assert result["country"].isna().sum() == 0

    def test_drops_empty_country(self, raw_customers):
        result = transform_customers(raw_customers)
        assert "" not in result["country"].values

    def test_deduplicates_customer_id_keeps_last(self, raw_customers):
        result = transform_customers(raw_customers)
        assert result["customer_id"].is_unique
        # customer_id=1 appears twice; the last one has name "Alice Updated"
        row = result[result["customer_id"] == "1"]
        assert row.iloc[0]["name"] == "Alice Updated"

    def test_customer_id_is_string(self, raw_customers):
        result = transform_customers(raw_customers)
        assert result["customer_id"].dtype == object or pd.api.types.is_string_dtype(
            result["customer_id"]
        )

    def test_does_not_mutate_input(self, raw_customers):
        original = raw_customers.copy()
        transform_customers(raw_customers)
        pd.testing.assert_frame_equal(raw_customers, original)


# ===========================================================================
# Tests — transform_orders
# ===========================================================================

class TestTransformOrders:

    def test_normalizes_date_format(self, raw_orders, clean_customers):
        result = transform_orders(raw_orders, clean_customers)
        for date_str in result["order_date"]:
            # Should match YYYY-MM-DD
            pd.to_datetime(date_str, format="%Y-%m-%d")

    def test_drops_invalid_dates(self, raw_orders, clean_customers):
        result = transform_orders(raw_orders, clean_customers)
        # "invalid_date" row should be gone
        assert len(result[result["order_id"] == "O6"]) == 0

    def test_drops_non_numeric_amount(self, raw_orders, clean_customers):
        result = transform_orders(raw_orders, clean_customers)
        # O4 has amount="abc"
        assert len(result[result["order_id"] == "O4"]) == 0

    def test_amount_is_numeric(self, raw_orders, clean_customers):
        result = transform_orders(raw_orders, clean_customers)
        assert pd.api.types.is_numeric_dtype(result["amount"])

    def test_drops_invalid_currency(self, raw_orders, clean_customers):
        # All remaining currencies should be in EXCHANGE_RATES
        result = transform_orders(raw_orders, clean_customers)
        assert result["currency"].isin(EXCHANGE_RATES.keys()).all()

    def test_drops_orders_with_unknown_customer(self, raw_orders, clean_customers):
        result = transform_orders(raw_orders, clean_customers)
        # customer_id=99 does not exist in clean_customers
        assert len(result[result["customer_id"] == "99"]) == 0

    def test_strips_whitespace_from_currency(self, raw_orders, clean_customers):
        result = transform_orders(raw_orders, clean_customers)
        # O7 has currency=" eur " which should become "EUR"
        o7 = result[result["order_id"] == "O7"]
        if len(o7) > 0:
            assert o7.iloc[0]["currency"] == "EUR"

    def test_does_not_mutate_input(self, raw_orders, clean_customers):
        original = raw_orders.copy()
        transform_orders(raw_orders, clean_customers)
        pd.testing.assert_frame_equal(raw_orders, original)


# ===========================================================================
# Tests — transform_payments
# ===========================================================================

class TestTransformPayments:

    def test_filters_only_paid_status(self, raw_payments):
        result = transform_payments(raw_payments)
        # Only PAID payments should contribute to aggregation
        # P5 is "failed", so O3 should have only one payment (P4)
        assert isinstance(result, pd.DataFrame)

    def test_drops_non_numeric_paid_amount(self, raw_payments):
        result = transform_payments(raw_payments)
        # P6 has paid_amount="bad" — should not appear
        assert pd.api.types.is_numeric_dtype(result["total_paid_usd"])

    def test_drops_invalid_currency(self, raw_payments):
        result = transform_payments(raw_payments)
        # P8 has currency="JPY" which is not in EXCHANGE_RATES
        # O5 should not appear in results
        assert "O5" not in result["order_id"].values

    def test_converts_to_usd_correctly(self, raw_payments):
        result = transform_payments(raw_payments)
        # O2 has one PAID payment: 200 GBP -> 200 * 1.3 = 260.0
        o2 = result[result["order_id"] == "O2"]
        assert len(o2) == 1
        assert o2.iloc[0]["total_paid_usd"] == pytest.approx(260.0)

    def test_aggregates_multiple_payments_per_order(self, raw_payments):
        result = transform_payments(raw_payments)
        # O1: P1 (50 USD, paid) + P2 (50.50 USD, PAID) = 100.50
        # P6 is "bad" amount so dropped
        o1 = result[result["order_id"] == "O1"]
        assert len(o1) == 1
        assert o1.iloc[0]["total_paid_usd"] == pytest.approx(100.50)

    def test_eur_conversion(self, raw_payments):
        result = transform_payments(raw_payments)
        # O3: P4 (25 EUR, PAID) -> 25 * 1.1 = 27.5 (P5 is failed, excluded)
        o3 = result[result["order_id"] == "O3"]
        assert len(o3) == 1
        assert o3.iloc[0]["total_paid_usd"] == pytest.approx(27.5)

    def test_returns_dataframe_with_expected_columns(self, raw_payments):
        result = transform_payments(raw_payments)
        assert "order_id" in result.columns
        assert "total_paid_usd" in result.columns

    def test_does_not_mutate_input(self, raw_payments):
        original = raw_payments.copy()
        transform_payments(raw_payments)
        pd.testing.assert_frame_equal(raw_payments, original)


# ===========================================================================
# Tests — join_and_enrich
# ===========================================================================

class TestJoinAndEnrich:

    def test_joins_orders_with_customers(self, clean_customers):
        orders = pd.DataFrame({
            "order_id": ["O1", "O2"],
            "customer_id": ["1", "2"],
            "order_date": ["2023-01-01", "2023-01-02"],
            "amount": [100.0, 200.0],
            "currency": ["USD", "GBP"],
        })
        payments_agg = pd.DataFrame({
            "order_id": ["O1"],
            "total_paid_usd": [100.0],
        })
        result = join_and_enrich(orders, clean_customers, payments_agg)
        assert "country" in result.columns
        assert len(result) == 2

    def test_fills_missing_payments_with_zero(self, clean_customers):
        orders = pd.DataFrame({
            "order_id": ["O1", "O2"],
            "customer_id": ["1", "2"],
            "order_date": ["2023-01-01", "2023-01-02"],
            "amount": [100.0, 200.0],
            "currency": ["USD", "GBP"],
        })
        payments_agg = pd.DataFrame({
            "order_id": ["O1"],
            "total_paid_usd": [100.0],
        })
        result = join_and_enrich(orders, clean_customers, payments_agg)
        o2 = result[result["order_id"] == "O2"]
        assert o2.iloc[0]["total_paid_usd"] == 0.0

    def test_excludes_orders_with_invalid_customer(self, clean_customers):
        orders = pd.DataFrame({
            "order_id": ["O1", "O2"],
            "customer_id": ["1", "999"],
            "order_date": ["2023-01-01", "2023-01-02"],
            "amount": [100.0, 200.0],
            "currency": ["USD", "GBP"],
        })
        payments_agg = pd.DataFrame({"order_id": [], "total_paid_usd": []})
        result = join_and_enrich(orders, clean_customers, payments_agg)
        # customer_id=999 not in clean_customers, inner join excludes it
        assert len(result) == 1

    def test_no_double_counting_with_single_payment(self, clean_customers):
        orders = pd.DataFrame({
            "order_id": ["O1"],
            "customer_id": ["1"],
            "order_date": ["2023-01-01"],
            "amount": [100.0],
            "currency": ["USD"],
        })
        payments_agg = pd.DataFrame({
            "order_id": ["O1"],
            "total_paid_usd": [100.0],
        })
        result = join_and_enrich(orders, clean_customers, payments_agg)
        assert len(result) == 1


# ===========================================================================
# Tests — aggregate_daily_revenue_by_country
# ===========================================================================

class TestAggregateDailyRevenue:

    def test_groups_by_date_and_country(self, enriched_df):
        result = aggregate_daily_revenue_by_country(enriched_df)
        assert "date" in result.columns
        assert "country" in result.columns

    def test_total_orders_count(self, enriched_df):
        result = aggregate_daily_revenue_by_country(enriched_df)
        # 2023-01-01, US: O1 and O2 -> 2 orders
        us_jan1 = result[(result["date"] == "2023-01-01") & (result["country"] == "US")]
        assert us_jan1.iloc[0]["total_orders"] == 2

    def test_total_revenue_is_rounded(self, enriched_df):
        result = aggregate_daily_revenue_by_country(enriched_df)
        for val in result["total_revenue_usd"]:
            assert val == round(val, 2)

    def test_revenue_sum_correct(self, enriched_df):
        result = aggregate_daily_revenue_by_country(enriched_df)
        # 2023-01-02, DE: O5 (300.0) + O6 (82.5) = 382.5
        de_jan2 = result[(result["date"] == "2023-01-02") & (result["country"] == "DE")]
        assert de_jan2.iloc[0]["total_revenue_usd"] == pytest.approx(382.5)

    def test_schema(self, enriched_df):
        result = aggregate_daily_revenue_by_country(enriched_df)
        expected_cols = {"date", "country", "total_orders", "total_revenue_usd"}
        assert set(result.columns) == expected_cols


# ===========================================================================
# Tests — aggregate_customer_metrics
# ===========================================================================

class TestAggregateCustomerMetrics:

    def test_groups_by_customer(self, enriched_df):
        result = aggregate_customer_metrics(enriched_df)
        assert result["customer_id"].is_unique

    def test_total_orders_per_customer(self, enriched_df):
        result = aggregate_customer_metrics(enriched_df)
        c1 = result[result["customer_id"] == "1"]
        assert c1.iloc[0]["total_orders"] == 2

    def test_avg_order_revenue(self, enriched_df):
        result = aggregate_customer_metrics(enriched_df)
        # customer_id=1: (100 + 200) / 2 = 150.0
        c1 = result[result["customer_id"] == "1"]
        assert c1.iloc[0]["avg_order_revenue_usd"] == pytest.approx(150.0)

    def test_max_order_revenue(self, enriched_df):
        result = aggregate_customer_metrics(enriched_df)
        # customer_id=1: max(100, 200) = 200.0
        c1 = result[result["customer_id"] == "1"]
        assert c1.iloc[0]["max_order_revenue_usd"] == pytest.approx(200.0)

    def test_values_are_rounded(self, enriched_df):
        result = aggregate_customer_metrics(enriched_df)
        for col in ["avg_order_revenue_usd", "max_order_revenue_usd"]:
            for val in result[col]:
                assert val == round(val, 2)


# ===========================================================================
# Tests — aggregate_country_metrics
# ===========================================================================

class TestAggregateCountryMetrics:

    def test_groups_by_country(self, enriched_df):
        result = aggregate_country_metrics(enriched_df)
        assert result["country"].is_unique

    def test_total_revenue(self, enriched_df):
        result = aggregate_country_metrics(enriched_df)
        # US: 100 + 200 = 300.0
        us = result[result["country"] == "US"]
        assert us.iloc[0]["total_revenue_usd"] == pytest.approx(300.0)

    def test_avg_order_revenue(self, enriched_df):
        result = aggregate_country_metrics(enriched_df)
        # GB: (195 + 0) / 2 = 97.5
        gb = result[result["country"] == "GB"]
        assert gb.iloc[0]["avg_order_revenue_usd"] == pytest.approx(97.5)

    def test_values_are_rounded(self, enriched_df):
        result = aggregate_country_metrics(enriched_df)
        for col in ["avg_order_revenue_usd", "total_revenue_usd"]:
            for val in result[col]:
                assert val == round(val, 2)

    def test_schema(self, enriched_df):
        result = aggregate_country_metrics(enriched_df)
        expected_cols = {"country", "avg_order_revenue_usd", "total_revenue_usd"}
        assert set(result.columns) == expected_cols


# ===========================================================================
# Tests — main (mocked I/O)
# ===========================================================================

class TestMain:

    @patch("main.os.makedirs")
    @patch("main.pd.read_csv")
    def test_main_reads_all_csv_files(self, mock_read_csv, mock_makedirs):
        """Verify main() calls pd.read_csv for all 3 input files."""
        mock_read_csv.side_effect = [
            # customers
            pd.DataFrame({
                "customer_id": [1],
                "name": ["Alice"],
                "country": ["US"],
            }),
            # orders
            pd.DataFrame({
                "order_id": ["O1"],
                "customer_id": [1],
                "order_date": ["2023-01-01"],
                "amount": [100.0],
                "currency": ["USD"],
            }),
            # payments
            pd.DataFrame({
                "payment_id": ["P1"],
                "order_id": ["O1"],
                "paid_amount": ["100"],
                "currency": ["USD"],
                "status": ["PAID"],
            }),
        ]

        with patch("builtins.print"):
            with patch("main.pd.DataFrame.to_csv"):
                main()

        assert mock_read_csv.call_count == 3
        mock_read_csv.assert_any_call("data/customers.csv")
        mock_read_csv.assert_any_call("data/orders.csv")
        mock_read_csv.assert_any_call("data/payments.csv")

    @patch("main.os.makedirs")
    @patch("main.pd.read_csv")
    def test_main_creates_output_directory(self, mock_read_csv, mock_makedirs):
        """Verify main() creates the output directory."""
        mock_read_csv.side_effect = [
            pd.DataFrame({"customer_id": [1], "name": ["A"], "country": ["US"]}),
            pd.DataFrame({
                "order_id": ["O1"], "customer_id": [1],
                "order_date": ["2023-01-01"], "amount": [100.0], "currency": ["USD"],
            }),
            pd.DataFrame({
                "payment_id": ["P1"], "order_id": ["O1"],
                "paid_amount": ["100"], "currency": ["USD"], "status": ["PAID"],
            }),
        ]

        with patch("builtins.print"):
            with patch("main.pd.DataFrame.to_csv"):
                main()

        mock_makedirs.assert_called_once_with("output", exist_ok=True)


# ===========================================================================
# Edge case tests
# ===========================================================================

class TestEdgeCases:

    def test_empty_customers(self):
        df = pd.DataFrame(columns=["customer_id", "name", "country"])
        result = transform_customers(df)
        assert len(result) == 0

    def test_empty_orders(self):
        customers = pd.DataFrame({
            "customer_id": ["1"], "name": ["A"], "country": ["US"],
        })
        orders = pd.DataFrame(columns=["order_id", "customer_id", "order_date", "amount", "currency"])
        result = transform_orders(orders, customers)
        assert len(result) == 0

    def test_empty_payments(self):
        df = pd.DataFrame(columns=["payment_id", "order_id", "paid_amount", "currency", "status"])
        result = transform_payments(df)
        assert len(result) == 0

    def test_all_payments_failed(self):
        df = pd.DataFrame({
            "payment_id": ["P1", "P2"],
            "order_id": ["O1", "O2"],
            "paid_amount": ["100", "200"],
            "currency": ["USD", "EUR"],
            "status": ["failed", "refunded"],
        })
        result = transform_payments(df)
        assert len(result) == 0

    def test_customer_id_with_alphanumeric(self):
        """Ensure customer_ids like '1054w' are preserved as strings."""
        df = pd.DataFrame({
            "customer_id": ["1054w", "1065t", "100"],
            "name": ["A", "B", "C"],
            "country": ["US", "GB", "DE"],
        })
        result = transform_customers(df)
        assert "1054w" in result["customer_id"].values
        assert "1065t" in result["customer_id"].values

    def test_orders_all_invalid_dates(self):
        customers = pd.DataFrame({
            "customer_id": ["1"], "name": ["A"], "country": ["US"],
        })
        orders = pd.DataFrame({
            "order_id": ["O1", "O2"],
            "customer_id": [1, 1],
            "order_date": ["bad", "also_bad"],
            "amount": [100.0, 200.0],
            "currency": ["USD", "EUR"],
        })
        result = transform_orders(orders, customers)
        assert len(result) == 0

    def test_payments_all_invalid_currency(self):
        df = pd.DataFrame({
            "payment_id": ["P1"],
            "order_id": ["O1"],
            "paid_amount": ["100"],
            "currency": ["JPY"],
            "status": ["PAID"],
        })
        result = transform_payments(df)
        assert len(result) == 0

    def test_join_with_empty_payments(self):
        customers = pd.DataFrame({
            "customer_id": ["1"], "name": ["A"], "country": ["US"],
        })
        orders = pd.DataFrame({
            "order_id": ["O1"],
            "customer_id": ["1"],
            "order_date": ["2023-01-01"],
            "amount": [100.0],
            "currency": ["USD"],
        })
        payments_agg = pd.DataFrame({"order_id": pd.Series(dtype="str"), "total_paid_usd": pd.Series(dtype="float")})
        result = join_and_enrich(orders, customers, payments_agg)
        assert len(result) == 1
        assert result.iloc[0]["total_paid_usd"] == 0.0