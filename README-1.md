# LeakLens — Revenue and Margin Analysis

LeakLens is a project that helps businesses spot possible problems in product pricing, profit margins, and inventory. It combines a product dashboard with a Python machine-learning script to highlight items that may be selling at a loss and products that could run out of stock before a new delivery arrives.

The goal is to make these issues easier to find so a business can review them and decide what action to take.

## What it looks for

- **Products selling below cost:** Items whose selling or list price may not cover their unit cost.
- **Discount-related margin problems:** Products where discounts may reduce or remove the expected margin.
- **Pricing issues:** Products that may need a closer look because of a mismatch between price and cost.
- **Stockout risk:** Items whose available stock may not last until the next supplier delivery.

LeakLens highlights potential issues for review. Its results should be checked against the business's source data before making pricing, purchasing, or operational decisions.

## Project files

```text
.
├── Leaklens.html
├── README.md
├── Scored_products.csv
├── Stock_risk.csv
└── Train_model.py
```

- **`Leaklens.html`** — The dashboard interface. Open it in a modern web browser to view the dashboard.
- **`Train_model.py`** — The Python script used for the machine-learning and data-analysis workflow.
- **`Scored_products.csv`** — Product-level output, including scoring and related product information.
- **`Stock_risk.csv`** — Inventory information used to review stock coverage and possible stockout risks.
- **`README.md`** — Project overview and instructions.

The CSV column names and available results depend on the files included in this repository.

## Technologies

The project documentation identifies the following technologies:

- Python
- Pandas and NumPy for working with data
- Scikit-learn, including the Isolation Forest algorithm
- HTML for the dashboard interface

Check `Train_model.py` for the exact libraries and settings used by the current version of the project.

## Machine-learning approach

The training script uses an **Isolation Forest**, an unsupervised machine-learning algorithm that can help identify records with unusual patterns. In this project, product pricing and margin-related features are used to help rank items that may deserve further review.

An anomaly score is a signal for investigation, not proof that a product is losing money. Straightforward checks—such as comparing a product's price with its unit cost—can also help identify clear pricing issues. The two approaches can be used together: rules flag known conditions, while anomaly scoring can surface less obvious patterns.

## Inventory analysis

The stock-risk output is intended to help identify products that may run out before replenishment arrives. The analysis can consider recent or historical sales activity, available stock, and supplier lead time.

Actual stockout risk depends on the quality and freshness of the input data. Review the underlying inventory and supplier information before acting on a flag.

## Open the dashboard

1. Download or clone this repository.
2. Locate `Leaklens.html`.
3. Double-click the file, or open it with a browser such as Chrome or Microsoft Edge.

The HTML file is intended to display the dashboard locally. If any dashboard feature depends on data embedded in the HTML, keep the file as provided in the project.

## Run the Python script

You need Python installed on your computer. From a terminal opened in the project folder, install the commonly used dependencies:

```bash
pip install pandas numpy scikit-learn
```

Then run:

```bash
python Train_model.py
```

The script's actual inputs, outputs, and execution requirements are determined by its code. Check the file before running it, especially if it expects additional data files or writes output files to a particular location.

## Results and limitations

The original project notes describe an analysis of 250 product SKUs and report findings involving loss-making products, pricing issues, discounts, and stockout risk. They also report evaluation metrics for the anomaly-detection model.

Those figures are specific to the dataset and evaluation setup used for the project; they should not be treated as guaranteed results for another retailer or dataset. Before publishing exact totals or model metrics, confirm that they can be reproduced from the included data and code.

LeakLens is a decision-support project, not a replacement for checking source records or applying business judgement. Its findings are only as reliable as the data and assumptions behind them.

## Privacy and external services

The project notes describe the model as using classical machine learning rather than a large language model. Review the code and dashboard to confirm whether the current version makes any network requests or relies on external services before describing it as fully offline.

## License

No license has been specified in this repository. Unless a license is added, others should not assume they have permission to reuse, modify, or redistribute the project.

---

**LeakLens** — making product pricing, margin, and inventory risks easier to review.
