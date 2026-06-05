# AI Architecture & Federated Learning Documentation

## 1. System Overview
This project implements a privacy-preserving **Mule Account Detection System** for banking networks. Because banking data contains highly sensitive Personally Identifiable Information (PII) and financial records, moving data to a centralized server for Machine Learning is a severe privacy risk. 

To solve this, the system utilizes **Federated Learning (FL)** to train an **XGBoost** classifier across multiple independent bank branches (clients). Instead of sharing customer data, the banks only share mathematical model updates.

---

## 2. Federated Learning Framework (Flower)

We utilize the **Flower (`flwr`)** framework to orchestrate the Federated Learning process. 

### Why Federated Learning?
*   **Data Privacy:** Customer transaction data never leaves the local bank's secure servers.
*   **Regulatory Compliance:** Easily complies with GDPR, RBI guidelines, and other financial data localization laws.
*   **Collaborative Intelligence:** Banks can collectively detect cross-institutional mule networks without sharing proprietary data.

### Implementation Details
*   **Clients (`fl_client.py`):** Represents individual bank branches. Each client loads its local, isolated dataset, performs feature engineering, and trains a local version of the XGBoost model.
*   **Server (`fl_server.py`):** The central aggregator. It does not see any raw data. It initializes the training process and manages the communication rounds.
*   **Custom Strategy (`MuleDetectionStrategy`):** XGBoost trees cannot be simply "averaged" like Neural Network weights. Our custom strategy securely extracts the tree structures and boosting parameters from all clients and intelligently aggregates them into a single, unified Global Model.
*   **Simulation (`main_simulation.py`):** The pipeline is configured to simulate a network of **5 independent clients** training collaboratively over **10 communication rounds**.

---

## 3. Machine Learning Model (XGBoost)

At the core of each client is an **Extreme Gradient Boosting (XGBoost)** model (`models/xgb_model.py`). 

### Why XGBoost?
*   **Tabular Data Supremacy:** Financial transactions and demographic data are heavily tabular. XGBoost consistently outperforms Deep Learning on structured, tabular datasets.
*   **Non-Linear Patterns:** It excels at finding complex, non-linear relationships (e.g., a specific combination of age, transaction velocity, and account tenure that indicates a mule).
*   **Interpretability:** XGBoost allows us to extract feature importance, making it possible to explain *why* an account was flagged (crucial for banking audits).

### Handling Imbalanced Data (SMOTE)
In banking, mule accounts are extremely rare compared to legitimate accounts (severe class imbalance). If trained normally, the model would just predict "Normal" for everyone to achieve 99% accuracy.
*   **Solution:** We implement **SMOTE (Synthetic Minority Over-sampling Technique)**. Before training, the pipeline synthesizes artificial examples of mule accounts to balance the dataset, forcing the model to learn the distinct patterns of fraudulent behavior rather than ignoring them.

### Feature Engineering (`utils/feature_engineering.py`)
Raw data is transformed into actionable intelligence before feeding it to the model. Key engineered features include:
*   **Velocity Metrics:** Total inward/outward flow (`F3799`, `F3800`) and transaction counts.
*   **Account Tenure:** Age of the account and SIM card linking (`F3889`, `F3887`).
*   **Demographic Risks:** Cross-referencing age (`F3894`) and occupation (`F3891`, e.g., "Student") with transaction volume to detect anomalies.

---

## 4. Training Workflow & Output

1.  **Initialization:** The central Server starts up and waits for clients to connect.
2.  **Local Training:** The 5 clients train their XGBoost models locally on their specific data subsets using SMOTE and feature scaling.
3.  **Aggregation:** Clients send their updated boosting parameters back to the Server.
4.  **Global Update:** The Server aggregates the updates using the `MuleDetectionStrategy` to form an improved Global Model.
5.  **Iteration:** This repeats for 10 rounds.
6.  **Finalization:** The final, highly-accurate Global Model is saved to `results/final_global_model.json`. 

### Evaluation Metrics
Because of the imbalanced nature of fraud detection, raw Accuracy is a deceptive metric. The model is strictly evaluated on:
*   **AUC-ROC (Area Under the Receiver Operating Characteristic Curve):** Measures the model's ability to rank mules higher than normal accounts.
*   **Recall (Sensitivity):** The percentage of actual mules successfully caught. *(This is prioritized heavily to ensure zero fraud escapes).*
*   **Precision:** Ensuring the false positive rate is kept manageable to reduce alert fatigue for the bank's fraud team.

---

## 5. Inference (API Deployment)
The final `final_global_model.json` is completely decoupled from the Flower framework. It is wrapped in a high-speed **FastAPI** layer (`app.py`), allowing external frontend applications (like Flutter apps or banking dashboards) to send real-time transaction data and instantly receive a Mule Probability Score and Risk Tier.
