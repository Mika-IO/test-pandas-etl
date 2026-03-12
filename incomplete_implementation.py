import pandas as pd
import re
from dateutil import parser


def is_numerical(value):
    return re.sub(r"[a-zA-Z0-9]+", "", str(value))


def process_date(date):
    date_formats = ["%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%Y/%m/%d"]
    processed_date = None
    for date_format in date_formats:
        try:
            processed_date = pd.to_datetime(date, format=date_format)
        except:
            pass
    return processed_date


def format_date(dt):
    try:
        return pd.to_datetime(dt, format="%d-%m-%Y")
    except:
        return pd.to_datetime(dt, format="%Y-%m-%d")


def format_date(dt):

    date_format = "%d%m-%Y"
    processed_date = pd.to_datetime(dt, format=date_format)
    return processed_date


def transform_customers(df):
    """
    Transform — Customers
    Apply the following rules to customers.csv:

    Normalize country codes to ISO-like uppercase (e.g., us → US)
    Strip leading/trailing whitespace from text fields
    If duplicate customer_ids exist:
    Keep the latest record (based on file order)
    Drop customers with missing country
    Ensure customer_id is consistently typed
    """
    df["country"] = df["country"].str.upper()
    df = df.dropna(subset=["country"])
    df["country"].str.strip()
    df = df.drop_duplicates(subset="country", keep="last")
    df["country"] = df["country"].apply(is_numerical)
    return df


def transform_orders(df, df_customers):
    """
    Apply the following rules to orders.csv:

    Normalize order_date to YYYY-MM-DD
    Invalid dates should be safely handled
    Convert amount to numeric
    Drop orders where:
    amount is missing or non-numeric
    currency is not present in EXCHANGE_RATES
    customer_id does not exist in the cleaned customers table
    """
    df["order_date"] = pd.to_datetime(format_date)
    print(df.head())
    # check. YYYY-MM-DD
    df["amount"].float()
    pass


def transform_payments(df):
    """
    Apply the following rules to payments.csv:

    Normalize status casing (e.g., paid, PAID)
    Only include payments where status == "PAID"
    Convert paid_amount to numeric
    Drop payments where:
    paid_amount is missing or non-numeric
    currency is not present in EXCHANGE_RATES
    Convert paid_amount to USD
    Aggregate payments per order before joining with orders/customers
    Note: Orders with no successful payments may either:

    be excluded, or
    included with revenue = 0
    Please explain your choice.
    """
    pass


def main():
    customers = pd.read_csv("data/customers.csv")
    df_customers = transform_customers(customers)

    orders = pd.read_csv("data/orders.csv")
    df_orders = transform_orders(orders, df_customers)
    print(orders.head())

    payments = pd.read_csv("data/payments.csv")


#   print(payments.head())


main()
