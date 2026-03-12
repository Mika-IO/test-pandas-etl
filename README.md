# ETL Pipeline

Simple ETL pipeline that processes customers, orders and payments from CSV files, cleans everything up, converts currencies to USD, and spits out aggregated metrics.

[Exercise tasks](exercise.md)

## How it works

The pipeline reads three CSVs from `data/`, applies a bunch of cleanup rules (whitespace trimming, date normalization, currency validation, deduplication, etc.), joins them together and outputs three reports:

- **Daily revenue by country** — order count + total revenue per date/country
- **Customer metrics** — total orders, avg and max revenue per customer
- **Country metrics** — avg order revenue and total revenue per country

Exchange rates are hardcoded:

```python
EXCHANGE_RATES = {"USD": 1.0, "EUR": 1.1, "GBP": 1.3}
```

Anything outside those three currencies gets dropped.

Orders without any successful payment are kept with `revenue = 0` — this way the order counts stay accurate and you can still spot unpaid orders in the data.

## Extra metrics
 
Beyond what was asked, the pipeline also computes a set of additional metrics (in `extra_metrics.py`) to give a fuller picture of the business:
 
**Payment health** — payment rate across all orders (what % actually got paid), per-order gap between expected amount and what was paid (catches partial payments), and a breakdown of payment statuses (PAID/FAILED/REFUNDED) per day so you can spot operational issues.
 
**Customer behavior** — average ticket per customer over time (is spending going up or down?), spend-based segmentation into buckets ($0, $0-50, $50-200, $200+), and a repurchase rate (one-time vs recurring buyers).
 
**Temporal patterns** — revenue grouped by day of the week, day-over-day absolute and % change, and a 7-day rolling average to smooth out noise and show the real trend.
 
**Currency distribution** — revenue split by original currency before conversion, with % share of total. Useful for spotting market concentration.
 

## Project structure

```
├── main.py          # the etl script
├── test_main.py     
├── requirements.txt
├── data/                    # input CSVs go here
│   ├── customers.csv
│   ├── orders.csv
│   └── payments.csv
└── output/                  # created automatically on run
```

## Setup

Python 3.10+

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Usage

Drop your CSVs into `data/` and run:

```bash
python main.py
```

Output goes to stdout and gets saved to `output/`.

## Tests

```bash
pytest test_main.py -v
```
