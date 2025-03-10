# Intraday Clustering for Financial State Detection

This repository contains the project **Intraday Clustering for Financial State Detection**, conducted as part of the **FIN-525: Financial Big Data** course. The project applies clustering techniques to **high-frequency financial data** from the **S&P 500 ETF** to detect **intraday market states**. The findings contribute to **algorithmic trading, risk management, and portfolio optimization**.

## 📂 Repository Structure

- **`submission/`** 📌 **(Important folder)**
  - Contains the final **report** and the **executable code** submitted to the professor.
  - The report provides detailed explanations of the **methodology, data processing, clustering techniques, and results**.
  - The executable code reproduces the analysis, including **data cleaning, correlation matrix construction, and clustering algorithms**.

- **`log returns/`**
  - Contains log return computations used in the analysis.

- **Python Scripts:**
  - `clustering_helpers.py` – Implements clustering methods such as **Louvain, Girvan-Newman, and Marsili-Giada**.
  - `correlation_helpers.py` – Constructs and refines **correlation matrices** using **eigenvalue clipping and random matrix filtering**.
  - `wrangling_helpers.py` – Handles **data cleaning and preprocessing** using **Polars** for high-performance computation.

- **Notebooks:**
  - `EDA.ipynb` – Exploratory data analysis on trade and best bid-offer data.
  - `Correlation.ipynb` – Implements correlation matrix computation and cleaning methods.
  - `Main.ipynb` – The main script for executing the full pipeline.



