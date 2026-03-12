import pytest
import pandas as pd
import numpy as np

from extra_metrics import (
    payment_rate,
    order_payment_gap,
    payment_status_breakdown,
    avg_ticket_over_time,
    customer_spend_segments,
    repurchase_rate,
    revenue_by_weekday,
    daily_revenue_dod_change,
    revenue_moving_avg_7d,
    revenue_by_original_currency,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def orders_df():
    return pd.DataFrame({
        "order_id": ["O1", "O2", "O3", "O4"],
        "customer_id": ["1", "1", "2", "3"],
        "order_date": ["2023-01-02", "2023-01-03", "2023-01-03", "2023-01-04"],
        "amount": [100.0, 200.0, 150.0, 50.0],
        "currency": ["USD", "EUR", "GBP", "USD"],
    })


@pytest.fixture
def payments_agg():
    return pd.DataFrame({
        "order_id": ["O1", "O2"],
        "total_paid_usd": [100.0, 220.0],
    })


@pytest.fixture
def enriched_df():
    return pd.DataFrame({
        "order_id": ["O1", "O2", "O3", "O4"],
        "customer_id": ["1", "1", "2", "3"],
        "order_date": ["2023-01-02", "2023-01-03", "2023-01-03", "2023-01-04"],
        "amount": [100.0, 200.0, 150.0, 50.0],
        "currency": ["USD", "EUR", "GBP", "USD"],
        "country": ["US", "DE", "GB", "US"],
        "total_paid_usd": [100.0, 220.0, 0.0, 0.0],
    })


@pytest.fixture
def raw_payments():
    return pd.DataFrame({
        "payment_id": ["P1", "P2", "P3", "P4", "P5"],
        "order_id": ["O1", "O1", "O2", "O3", "O4"],
        "paid_amount": [80.0, 20.0, 220.0, 150.0, 50.0],
        "currency": ["USD", "USD", "EUR", "GBP", "USD"],
        "status": ["PAID", "PAID", " paid ", "FAILED", "refunded"],
    })


@pytest.fixture
def multi_day_enriched():
    """7+ days of data for moving average and DoD tests."""
    dates = [f"2023-01-{d:02d}" for d in range(1, 11)]
    return pd.DataFrame({
        "order_id": [f"O{i}" for i in range(10)],
        "customer_id": ["1"] * 5 + ["2"] * 5,
        "order_date": dates,
        "amount": [100.0] * 10,
        "currency": ["USD"] * 10,
        "country": ["US"] * 10,
        "total_paid_usd": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
    })


# ===========================================================================
# Tests — payment_rate
# ===========================================================================

class TestPaymentRate:

    def test_calculates_rate_correctly(self, orders_df, payments_agg):
        result = payment_rate(orders_df, payments_agg)
        assert result["total_orders"] == 4
        assert result["paid_orders"] == 2
        assert result["payment_rate_pct"] == 50.0

    def test_all_paid(self, orders_df):
        agg = pd.DataFrame({"order_id": ["O1", "O2", "O3", "O4"], "total_paid_usd": [1, 1, 1, 1]})
        result = payment_rate(orders_df, agg)
        assert result["payment_rate_pct"] == 100.0

    def test_none_paid(self, orders_df):
        agg = pd.DataFrame({"order_id": pd.Series(dtype="str"), "total_paid_usd": pd.Series(dtype="float")})
        result = payment_rate(orders_df, agg)
        assert result["payment_rate_pct"] == 0.0

    def test_empty_orders(self):
        orders = pd.DataFrame(columns=["order_id", "customer_id", "order_date", "amount", "currency"])
        agg = pd.DataFrame(columns=["order_id", "total_paid_usd"])
        result = payment_rate(orders, agg)
        assert result["total_orders"] == 0
        assert result["payment_rate_pct"] == 0.0


# ===========================================================================
# Tests — order_payment_gap
# ===========================================================================

class TestOrderPaymentGap:

    def test_fully_paid_gap_is_zero(self, enriched_df):
        result = order_payment_gap(enriched_df)
        o1 = result[result["order_id"] == "O1"]
        # 100 USD * 1.0 - 100.0 = 0
        assert o1.iloc[0]["gap_usd"] == 0.0

    def test_unpaid_order_shows_full_gap(self, enriched_df):
        result = order_payment_gap(enriched_df)
        o4 = result[result["order_id"] == "O4"]
        # 50 USD * 1.0 - 0.0 = 50.0
        assert o4.iloc[0]["gap_usd"] == 50.0

    def test_overpaid_shows_negative_gap(self, enriched_df):
        result = order_payment_gap(enriched_df)
        o2 = result[result["order_id"] == "O2"]
        # 200 EUR * 1.1 = 220.0 amount_usd, paid 220.0 -> gap = 0
        assert o2.iloc[0]["gap_usd"] == 0.0

    def test_output_columns(self, enriched_df):
        result = order_payment_gap(enriched_df)
        expected = {"order_id", "customer_id", "amount_usd", "total_paid_usd", "gap_usd"}
        assert set(result.columns) == expected

    def test_does_not_mutate_input(self, enriched_df):
        original = enriched_df.copy()
        order_payment_gap(enriched_df)
        pd.testing.assert_frame_equal(enriched_df, original)


# ===========================================================================
# Tests — payment_status_breakdown
# ===========================================================================

class TestPaymentStatusBreakdown:

    def test_counts_statuses(self, raw_payments, orders_df):
        result = payment_status_breakdown(raw_payments, orders_df)
        assert "status" in result.columns
        assert "count" in result.columns

    def test_normalizes_status_casing(self, raw_payments, orders_df):
        result = payment_status_breakdown(raw_payments, orders_df)
        for status in result["status"]:
            assert status == status.upper()
            assert status == status.strip()

    def test_groups_by_date(self, raw_payments, orders_df):
        result = payment_status_breakdown(raw_payments, orders_df)
        assert "order_date" in result.columns

    def test_empty_payments(self, orders_df):
        empty = pd.DataFrame(columns=["payment_id", "order_id", "paid_amount", "currency", "status"])
        result = payment_status_breakdown(empty, orders_df)
        assert len(result) == 0


# ===========================================================================
# Tests — avg_ticket_over_time
# ===========================================================================

class TestAvgTicketOverTime:

    def test_returns_per_customer_per_date(self, enriched_df):
        result = avg_ticket_over_time(enriched_df)
        assert "customer_id" in result.columns
        assert "order_date" in result.columns
        assert "avg_ticket_usd" in result.columns

    def test_values_are_rounded(self, enriched_df):
        result = avg_ticket_over_time(enriched_df)
        for val in result["avg_ticket_usd"]:
            assert val == round(val, 2)

    def test_single_order_per_day_equals_paid(self, enriched_df):
        result = avg_ticket_over_time(enriched_df)
        # customer 1, 2023-01-02 has one order with total_paid_usd=100
        row = result[(result["customer_id"] == "1") & (result["order_date"] == "2023-01-02")]
        assert row.iloc[0]["avg_ticket_usd"] == 100.0


# ===========================================================================
# Tests — customer_spend_segments
# ===========================================================================

class TestCustomerSpendSegments:

    def test_all_segments_present(self, enriched_df):
        result = customer_spend_segments(enriched_df)
        assert len(result) == 4  # 4 buckets

    def test_zero_revenue_bucket(self, enriched_df):
        result = customer_spend_segments(enriched_df)
        # customer 2 and 3 have total_paid_usd = 0
        zero_row = result[result["segment"] == "$0 (no revenue)"]
        assert zero_row.iloc[0]["customer_count"] == 2

    def test_high_spender_bucket(self, enriched_df):
        result = customer_spend_segments(enriched_df)
        # customer 1: 100 + 220 = 320 -> $200+ bucket
        high = result[result["segment"] == "$200+"]
        assert high.iloc[0]["customer_count"] == 1

    def test_empty_data(self):
        df = pd.DataFrame(columns=["order_id", "customer_id", "order_date", "amount", "currency", "country", "total_paid_usd"])
        result = customer_spend_segments(df)
        assert result["customer_count"].sum() == 0


# ===========================================================================
# Tests — repurchase_rate
# ===========================================================================

class TestRepurchaseRate:

    def test_calculates_correctly(self, enriched_df):
        result = repurchase_rate(enriched_df)
        # customer 1 has 2 orders (recurring), customer 2 and 3 have 1 each
        assert result["total_customers"] == 3
        assert result["recurring"] == 1
        assert result["one_time"] == 2
        assert result["repurchase_rate_pct"] == pytest.approx(33.33)

    def test_all_one_time(self):
        df = pd.DataFrame({
            "order_id": ["O1", "O2"],
            "customer_id": ["1", "2"],
            "total_paid_usd": [10, 20],
        })
        result = repurchase_rate(df)
        assert result["recurring"] == 0
        assert result["repurchase_rate_pct"] == 0.0

    def test_all_recurring(self):
        df = pd.DataFrame({
            "order_id": ["O1", "O2", "O3", "O4"],
            "customer_id": ["1", "1", "2", "2"],
            "total_paid_usd": [10, 20, 30, 40],
        })
        result = repurchase_rate(df)
        assert result["repurchase_rate_pct"] == 100.0

    def test_empty_data(self):
        df = pd.DataFrame(columns=["order_id", "customer_id", "total_paid_usd"])
        result = repurchase_rate(df)
        assert result["total_customers"] == 0


# ===========================================================================
# Tests — revenue_by_weekday
# ===========================================================================

class TestRevenueByWeekday:

    def test_returns_weekday_names(self, enriched_df):
        result = revenue_by_weekday(enriched_df)
        assert "weekday" in result.columns
        valid_days = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
        assert set(result["weekday"]).issubset(valid_days)

    def test_revenue_is_rounded(self, enriched_df):
        result = revenue_by_weekday(enriched_df)
        for val in result["total_revenue_usd"]:
            assert val == round(val, 2)

    def test_order_counts(self, enriched_df):
        result = revenue_by_weekday(enriched_df)
        assert result["total_orders"].sum() == len(enriched_df)


# ===========================================================================
# Tests — daily_revenue_dod_change
# ===========================================================================

class TestDailyRevenueDodChange:

    def test_first_day_has_nan_change(self, multi_day_enriched):
        result = daily_revenue_dod_change(multi_day_enriched)
        assert pd.isna(result.iloc[0]["dod_change_usd"])
        assert pd.isna(result.iloc[0]["dod_change_pct"])

    def test_change_values_are_numeric(self, multi_day_enriched):
        result = daily_revenue_dod_change(multi_day_enriched)
        # skip first row (NaN)
        assert pd.api.types.is_numeric_dtype(result["dod_change_usd"])
        assert pd.api.types.is_numeric_dtype(result["dod_change_pct"])

    def test_positive_trend(self, multi_day_enriched):
        result = daily_revenue_dod_change(multi_day_enriched)
        # each day has 10 more than previous, so changes should be positive
        changes = result["dod_change_usd"].dropna()
        assert (changes > 0).all()

    def test_output_columns(self, multi_day_enriched):
        result = daily_revenue_dod_change(multi_day_enriched)
        expected = {"order_date", "total_revenue_usd", "dod_change_usd", "dod_change_pct"}
        assert set(result.columns) == expected


# ===========================================================================
# Tests — revenue_moving_avg_7d
# ===========================================================================

class TestRevenueMovingAvg7d:

    def test_first_day_equals_itself(self, multi_day_enriched):
        result = revenue_moving_avg_7d(multi_day_enriched)
        # with min_periods=1, first row MA = first row value
        assert result.iloc[0]["ma_7d_usd"] == result.iloc[0]["total_revenue_usd"]

    def test_window_smooths_values(self, multi_day_enriched):
        result = revenue_moving_avg_7d(multi_day_enriched)
        # MA should be less volatile than raw values for increasing series
        raw_std = result["total_revenue_usd"].std()
        ma_std = result["ma_7d_usd"].std()
        assert ma_std <= raw_std

    def test_output_columns(self, multi_day_enriched):
        result = revenue_moving_avg_7d(multi_day_enriched)
        expected = {"order_date", "total_revenue_usd", "ma_7d_usd"}
        assert set(result.columns) == expected

    def test_all_values_present(self, multi_day_enriched):
        result = revenue_moving_avg_7d(multi_day_enriched)
        # min_periods=1 means no NaN
        assert result["ma_7d_usd"].isna().sum() == 0


# ===========================================================================
# Tests — revenue_by_original_currency
# ===========================================================================

class TestRevenueByCurrency:

    def test_groups_by_currency(self, enriched_df):
        result = revenue_by_original_currency(enriched_df)
        assert "currency" in result.columns
        assert result["currency"].is_unique

    def test_pct_sums_to_100(self, enriched_df):
        result = revenue_by_original_currency(enriched_df)
        total_pct = result["pct_of_total_revenue"].sum()
        assert total_pct == pytest.approx(100.0, abs=0.1)

    def test_output_columns(self, enriched_df):
        result = revenue_by_original_currency(enriched_df)
        expected = {"currency", "total_orders", "total_revenue_usd", "total_amount_original", "pct_of_total_revenue"}
        assert set(result.columns) == expected

    def test_empty_data(self):
        df = pd.DataFrame(columns=["order_id", "customer_id", "order_date", "amount", "currency", "country", "total_paid_usd"])
        result = revenue_by_original_currency(df)
        assert len(result) == 0

    def test_values_are_rounded(self, enriched_df):
        result = revenue_by_original_currency(enriched_df)
        for col in ["total_revenue_usd", "total_amount_original", "pct_of_total_revenue"]:
            for val in result[col]:
                assert val == round(val, 2)