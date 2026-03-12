import pandas as pd
import numpy as np

from main import EXCHANGE_RATES


# ---------------------------------------------------------------------------
# Payment health & conversion metrics
# ---------------------------------------------------------------------------

def payment_rate(df_orders, df_payments_agg) -> dict:
    """
    Percentage of orders that have at least one successful payment.
    Returns a dict with total_orders, paid_orders and payment_rate_pct.
    """
    total = len(df_orders)
    if total == 0:
        return {"total_orders": 0, "paid_orders": 0, "payment_rate_pct": 0.0}

    paid_order_ids = set(df_payments_agg["order_id"])
    paid = len(df_orders[df_orders["order_id"].isin(paid_order_ids)])

    return {
        "total_orders": total,
        "paid_orders": paid,
        "payment_rate_pct": round((paid / total) * 100, 2),
    }


def order_payment_gap(df_enriched) -> pd.DataFrame:
    """
    Difference between the order amount (converted to USD) and what was actually paid.
    Positive gap = underpaid, negative = overpaid.
    """
    df = df_enriched.copy()
    df["amount_usd"] = df.apply(
        lambda r: r["amount"] * EXCHANGE_RATES.get(r["currency"], 0), axis=1
    )
    df["gap_usd"] = (df["amount_usd"] - df["total_paid_usd"]).round(2)
    return df[["order_id", "customer_id", "amount_usd", "total_paid_usd", "gap_usd"]]


def payment_status_breakdown(df_payments_raw, df_orders) -> pd.DataFrame:
    """
    Count of each payment status per order_date.
    Requires the raw (pre-transform) payments joined with orders for the date.
    """
    df = df_payments_raw.copy()
    df["status"] = df["status"].str.strip().str.upper()

    # Bring in order_date so we can group by period
    df = df.merge(df_orders[["order_id", "order_date"]], on="order_id", how="inner")

    result = (
        df.groupby(["order_date", "status"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    return result


# ---------------------------------------------------------------------------
# Customer behavior metrics
# ---------------------------------------------------------------------------

def avg_ticket_over_time(df_enriched) -> pd.DataFrame:
    """
    Average ticket (revenue) per customer per date.
    Useful for spotting whether a customer's spending is trending up or down.
    """
    result = (
        df_enriched.groupby(["customer_id", "order_date"], as_index=False)
        .agg(avg_ticket_usd=("total_paid_usd", "mean"))
    )
    result["avg_ticket_usd"] = result["avg_ticket_usd"].round(2)
    return result


def customer_spend_segments(df_enriched) -> pd.DataFrame:
    """
    Segment customers into spend buckets based on their total revenue.
    Buckets: $0, $0.01–50, $50.01–200, $200+
    """
    totals = (
        df_enriched.groupby("customer_id", as_index=False)
        .agg(total_revenue_usd=("total_paid_usd", "sum"))
    )

    bins = [-0.01, 0, 50, 200, float("inf")]
    labels = ["$0 (no revenue)", "$0.01–50", "$50.01–200", "$200+"]
    totals["segment"] = pd.cut(totals["total_revenue_usd"], bins=bins, labels=labels)

    summary = (
        totals.groupby("segment", observed=False, as_index=False)
        .agg(customer_count=("customer_id", "count"))
    )
    return summary


def repurchase_rate(df_enriched) -> dict:
    """
    Percentage of customers who placed more than one order.
    """
    orders_per_customer = df_enriched.groupby("customer_id")["order_id"].nunique()
    total = len(orders_per_customer)
    if total == 0:
        return {"total_customers": 0, "one_time": 0, "recurring": 0, "repurchase_rate_pct": 0.0}

    recurring = int((orders_per_customer > 1).sum())
    one_time = total - recurring

    return {
        "total_customers": total,
        "one_time": one_time,
        "recurring": recurring,
        "repurchase_rate_pct": round((recurring / total) * 100, 2),
    }


# ---------------------------------------------------------------------------
# Temporal metrics
# ---------------------------------------------------------------------------

def revenue_by_weekday(df_enriched) -> pd.DataFrame:
    """
    Total revenue grouped by day of the week (Monday=0 .. Sunday=6).
    """
    df = df_enriched.copy()
    df["weekday"] = pd.to_datetime(df["order_date"]).dt.day_name()
    df["weekday_num"] = pd.to_datetime(df["order_date"]).dt.dayofweek

    result = (
        df.groupby(["weekday_num", "weekday"], as_index=False)
        .agg(
            total_orders=("order_id", "count"),
            total_revenue_usd=("total_paid_usd", "sum"),
        )
        .sort_values("weekday_num")
    )
    result["total_revenue_usd"] = result["total_revenue_usd"].round(2)
    return result.drop(columns=["weekday_num"]).reset_index(drop=True)


def daily_revenue_dod_change(df_enriched) -> pd.DataFrame:
    """
    Day-over-day revenue with absolute and percentage change.
    """
    daily = (
        df_enriched.groupby("order_date", as_index=False)
        .agg(total_revenue_usd=("total_paid_usd", "sum"))
        .sort_values("order_date")
        .reset_index(drop=True)
    )
    daily["total_revenue_usd"] = daily["total_revenue_usd"].round(2)
    daily["dod_change_usd"] = daily["total_revenue_usd"].diff().round(2)
    daily["dod_change_pct"] = (
        daily["total_revenue_usd"].pct_change().mul(100).round(2)
    )
    return daily


def revenue_moving_avg_7d(df_enriched) -> pd.DataFrame:
    """
    7-day rolling average of daily revenue.
    """
    daily = (
        df_enriched.groupby("order_date", as_index=False)
        .agg(total_revenue_usd=("total_paid_usd", "sum"))
        .sort_values("order_date")
        .reset_index(drop=True)
    )
    daily["total_revenue_usd"] = daily["total_revenue_usd"].round(2)
    daily["ma_7d_usd"] = (
        daily["total_revenue_usd"].rolling(window=7, min_periods=1).mean().round(2)
    )
    return daily


# ---------------------------------------------------------------------------
# Currency distribution metrics
# ---------------------------------------------------------------------------

def revenue_by_original_currency(df_enriched) -> pd.DataFrame:
    """
    Total revenue grouped by the original currency before USD conversion.
    Shows where the money actually comes from.
    """
    result = (
        df_enriched.groupby("currency", as_index=False)
        .agg(
            total_orders=("order_id", "count"),
            total_revenue_usd=("total_paid_usd", "sum"),
            total_amount_original=("amount", "sum"),
        )
    )
    result["total_revenue_usd"] = result["total_revenue_usd"].round(2)
    result["total_amount_original"] = result["total_amount_original"].round(2)

    grand_total = result["total_revenue_usd"].sum()
    if grand_total > 0:
        result["pct_of_total_revenue"] = (
            (result["total_revenue_usd"] / grand_total) * 100
        ).round(2)
    else:
        result["pct_of_total_revenue"] = 0.0

    return result