import os
import pandas as pd
import numpy as np

EXCHANGE_RATES = {
    "USD": 1.0,
    "EUR": 1.1,
    "GBP": 1.3,
}


def parse_date(value):
    """Try to parse a date string into a datetime object using common formats."""
    date_formats = ["%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"]
    for fmt in date_formats:
        try:
            return pd.to_datetime(value, format=fmt)
        except (ValueError, TypeError):
            continue
    return pd.NaT


def transform_customers(df) -> pd.DataFrame:
    """
    Transform — Customers:
    - Strip leading/trailing whitespace from text fields
    - Normalize country codes to uppercase
    - Drop records with missing country
    - Deduplicate customer_ids (keep the latest record by file order)
    - Ensure customer_id is consistently typed as string
    """
    df = df.copy()

    text_cols = df.select_dtypes(include=["object"]).columns
    for col in text_cols:
        df[col] = df[col].str.strip()

    df["country"] = df["country"].str.upper()

    df = df.dropna(subset=["country"])
    df = df[df["country"] != ""]

    df["customer_id"] = df["customer_id"].astype(str)

    df = df.drop_duplicates(subset="customer_id", keep="last")

    df = df.reset_index(drop=True)
    return df


def transform_orders(df, df_customers) -> pd.DataFrame:
    """
    Transform — Orders:
    - Normalize order_date to YYYY-MM-DD format
    - Convert amount to numeric
    - Drop orders where:
        - amount is missing or non-numeric
        - currency is not in EXCHANGE_RATES
        - customer_id does not exist in the cleaned customers table
    """
    df = df.copy()

    text_cols = df.select_dtypes(include=["object"]).columns
    for col in text_cols:
        df[col] = df[col].str.strip()

    df["customer_id"] = df["customer_id"].astype(str)

    df["order_date"] = df["order_date"].apply(parse_date)
    df = df.dropna(subset=["order_date"])
    if df.empty:
        return df.reset_index(drop=True)
    df["order_date"] = df["order_date"].dt.strftime("%Y-%m-%d")

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["amount"])

    df["currency"] = df["currency"].str.upper()
    df = df[df["currency"].isin(EXCHANGE_RATES.keys())]

    valid_customer_ids = set(df_customers["customer_id"])
    df = df[df["customer_id"].isin(valid_customer_ids)]

    df = df.reset_index(drop=True)
    return df


def transform_payments(df) -> pd.DataFrame:
    """
    Transform — Payments:
    - Normalize status to uppercase
    - Keep only payments where status == "PAID"
    - Convert paid_amount to numeric
    - Drop payments where paid_amount is missing/non-numeric or currency is invalid
    - Convert paid_amount to USD using EXCHANGE_RATES
    - Aggregate payments per order_id

    Design decision: orders with no successful payments are included with revenue = 0.
    Rationale: keeping those orders preserves accurate total_orders counts per
    country/day and allows analysis of unpaid orders (e.g. conversion funnels).
    Excluding them would distort volume metrics.
    """
    df = df.copy()

    text_cols = df.select_dtypes(include=["object"]).columns
    for col in text_cols:
        df[col] = df[col].str.strip()

    df["status"] = df["status"].str.upper()
    df = df[df["status"] == "PAID"]

    df["paid_amount"] = pd.to_numeric(df["paid_amount"], errors="coerce")
    df = df.dropna(subset=["paid_amount"])

    df["currency"] = df["currency"].str.upper()
    df = df[df["currency"].isin(EXCHANGE_RATES.keys())]

    if df.empty:
        return pd.DataFrame(columns=["order_id", "total_paid_usd"])

    df["paid_amount_usd"] = df.apply(
        lambda row: row["paid_amount"] * EXCHANGE_RATES[row["currency"]], axis=1
    )

    df_agg = (
        df.groupby("order_id", as_index=False)
        .agg(total_paid_usd=("paid_amount_usd", "sum"))
    )

    return df_agg


def join_and_enrich(df_orders, df_customers, df_payments_agg) -> pd.DataFrame:
    """
    Join & Enrich:
    - Inner join orders with customers (ensures only valid customers)
    - Left join aggregated payments onto orders (preserves orders with no payments)
    - Fill missing payment amounts with 0
    """
    df = df_orders.merge(
        df_customers[["customer_id", "country"]],
        on="customer_id",
        how="inner",
    )

    df = df.merge(df_payments_agg, on="order_id", how="left")

    df["total_paid_usd"] = df["total_paid_usd"].fillna(0.0)

    return df


def aggregate_daily_revenue_by_country(df) -> pd.DataFrame:
    """
    A. Daily revenue by country:
    - total_orders: number of orders
    - total_revenue_usd: sum of payment revenue in USD (rounded to 2 decimals)
    """
    result = (
        df.groupby(["order_date", "country"], as_index=False)
        .agg(
            total_orders=("order_id", "count"),
            total_revenue_usd=("total_paid_usd", "sum"),
        )
    )
    result["total_revenue_usd"] = result["total_revenue_usd"].round(2)
    result = result.rename(columns={"order_date": "date"})
    return result


def aggregate_customer_metrics(df) -> pd.DataFrame:
    """
    B. Customer-level metrics:
    - total_orders: number of orders per customer
    - avg_order_revenue_usd: average order revenue in USD
    - max_order_revenue_usd: maximum single-order revenue in USD
    """
    result = (
        df.groupby("customer_id", as_index=False)
        .agg(
            total_orders=("order_id", "count"),
            avg_order_revenue_usd=("total_paid_usd", "mean"),
            max_order_revenue_usd=("total_paid_usd", "max"),
        )
    )
    result["avg_order_revenue_usd"] = result["avg_order_revenue_usd"].round(2)
    result["max_order_revenue_usd"] = result["max_order_revenue_usd"].round(2)
    return result


def aggregate_country_metrics(df) -> pd.DataFrame:
    """
    C. Country-level metrics:
    - avg_order_revenue_usd: average order revenue in USD
    - total_revenue_usd: total revenue in USD
    """
    result = (
        df.groupby("country", as_index=False)
        .agg(
            avg_order_revenue_usd=("total_paid_usd", "mean"),
            total_revenue_usd=("total_paid_usd", "sum"),
        )
    )
    result["avg_order_revenue_usd"] = result["avg_order_revenue_usd"].round(2)
    result["total_revenue_usd"] = result["total_revenue_usd"].round(2)
    return result


def main():
    # 1. Extract
    customers = pd.read_csv("data/customers.csv")
    orders = pd.read_csv("data/orders.csv")
    payments = pd.read_csv("data/payments.csv")

    # 2-4. Transform
    df_customers = transform_customers(customers)
    df_orders = transform_orders(orders, df_customers)
    df_payments_agg = transform_payments(payments)

    # 5. Join & Enrich
    df_enriched = join_and_enrich(df_orders, df_customers, df_payments_agg)

    # 6. Aggregate
    daily_revenue = aggregate_daily_revenue_by_country(df_enriched)
    customer_metrics = aggregate_customer_metrics(df_enriched)
    country_metrics = aggregate_country_metrics(df_enriched)

    # Output
    print("=== Daily Revenue by Country ===")
    print(daily_revenue.to_string(index=False))
    print()

    print("=== Customer Metrics ===")
    print(customer_metrics.to_string(index=False))
    print()

    print("=== Country Metrics ===")
    print(country_metrics.to_string(index=False))

    # 7. Extra metrics
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

    # Clean raw payments for status breakdown (needs raw statuses + order dates)
    payments_clean = payments.copy()
    payments_clean["status"] = payments_clean["status"].str.strip().str.upper()

    pay_rate = payment_rate(df_orders, df_payments_agg)
    gap = order_payment_gap(df_enriched)
    status_breakdown = payment_status_breakdown(payments_clean, df_orders)
    ticket_over_time = avg_ticket_over_time(df_enriched)
    spend_segments = customer_spend_segments(df_enriched)
    repurchase = repurchase_rate(df_enriched)
    weekday_revenue = revenue_by_weekday(df_enriched)
    dod = daily_revenue_dod_change(df_enriched)
    ma7d = revenue_moving_avg_7d(df_enriched)
    currency_dist = revenue_by_original_currency(df_enriched)

    print("\n=== Payment Rate ===")
    for k, v in pay_rate.items():
        print(f"  {k}: {v}")
    print()

    print("=== Order Payment Gap (top 10 underpaid) ===")
    print(gap.sort_values("gap_usd", ascending=False).head(10).to_string(index=False))
    print()

    print("=== Payment Status Breakdown ===")
    print(status_breakdown.to_string(index=False))
    print()

    print("=== Customer Spend Segments ===")
    print(spend_segments.to_string(index=False))
    print()

    print("=== Repurchase Rate ===")
    for k, v in repurchase.items():
        print(f"  {k}: {v}")
    print()

    print("=== Revenue by Weekday ===")
    print(weekday_revenue.to_string(index=False))
    print()

    print("=== Day-over-Day Revenue Change ===")
    print(dod.to_string(index=False))
    print()

    print("=== 7-Day Moving Average ===")
    print(ma7d.to_string(index=False))
    print()

    print("=== Revenue by Original Currency ===")
    print(currency_dist.to_string(index=False))

    # Save results
    os.makedirs("output", exist_ok=True)
    daily_revenue.to_csv("output/daily_revenue_by_country.csv", index=False)
    customer_metrics.to_csv("output/customer_metrics.csv", index=False)
    country_metrics.to_csv("output/country_metrics.csv", index=False)
    gap.to_csv("output/order_payment_gap.csv", index=False)
    status_breakdown.to_csv("output/payment_status_breakdown.csv", index=False)
    ticket_over_time.to_csv("output/avg_ticket_over_time.csv", index=False)
    spend_segments.to_csv("output/customer_spend_segments.csv", index=False)
    weekday_revenue.to_csv("output/revenue_by_weekday.csv", index=False)
    dod.to_csv("output/daily_revenue_dod.csv", index=False)
    ma7d.to_csv("output/revenue_moving_avg_7d.csv", index=False)
    currency_dist.to_csv("output/revenue_by_currency.csv", index=False)

    print("\nFiles saved to output/")


if __name__ == "__main__":
    main()