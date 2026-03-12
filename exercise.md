# ETL & Data Aggregation Exercise

## Input Files

- `data/customers.csv` — Customer master data
- `data/orders.csv` — Orders placed by customers
- `data/payments.csv` — Payments made against orders

## Exchange Rates

Use the following exchange rates to convert all monetary values to USD:
```python
EXCHANGE_RATES = {
        "USD": 1.0,
        "EUR": 1.1,
        "GBP": 1.3,
}
```

Currencies not present in this mapping should be considered invalid.

## Tasks

### 1️⃣ Extract

- Read all input CSV files.
- Parse records into appropriate Python data structures.

### 2️⃣ Transform — Customers

Apply the following rules to `customers.csv`:
- Normalize country codes to ISO-like uppercase (e.g., `us` → `US`)
- Strip leading/trailing whitespace from text fields
- If duplicate `customer_id`s exist:
    - Keep the latest record (based on file order)
- Drop customers with missing country
- Ensure `customer_id` is consistently typed

### 3️⃣ Transform — Orders

Apply the following rules to `orders.csv`:
- Normalize `order_date` to `YYYY-MM-DD`
    - Invalid dates should be safely handled
- Convert `amount` to numeric
- Drop orders where:
    - `amount` is missing or non-numeric
    - `currency` is not present in `EXCHANGE_RATES`
    - `customer_id` does not exist in the cleaned customers table

### 4️⃣ Transform — Payments

Apply the following rules to `payments.csv`:
- Normalize `status` casing (e.g., `paid`, `PAID`)
- Only include payments where `status == "PAID"`
- Convert `paid_amount` to numeric
- Drop payments where:
    - `paid_amount` is missing or non-numeric
    - `currency` is not present in `EXCHANGE_RATES`
- Convert `paid_amount` to USD
- Aggregate payments per order before joining with orders/customers

> **Note:** Orders with no successful payments may either:
> - be excluded, or
> - included with `revenue = 0`
>
> Please explain your choice.

### 5️⃣ Join & Enrich

- Join orders with customers
- Join aggregated payments with orders
- Ensure joins do not cause double counting

---

### 6️⃣ Aggregate

Produce the following metrics:

#### A. Daily revenue by country

For each date and country, compute:
- `total_orders` — number of orders
- `total_revenue_usd` — sum of payment revenue (rounded to 2 decimals)

**Schema:**  
`date,country,total_orders,total_revenue_usd`

---

#### B. Customer-level metrics

For each customer:
- Total number of orders
- Average order revenue (USD)
- Maximum single-order revenue (USD)

---

#### C. Country-level metrics

For each country:
- Average order revenue (USD)
- Total revenue (USD)
