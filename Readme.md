# Aviation Network Delay Analytics & Classification

> **Course**: INT315 - Cluster Computing  
> **Mapped Units**:
> - **Unit III**: Graph Analytics & Spark GraphX (Airports as Vertices, Flight Routes as Directed Edges, In/Out-Degrees, Hub Congestion)
> - **Unit IV**: Running SQL Queries using Spark SQL (GroupBy, Aggregate Functions, Bottleneck Isolation, SQL Wildcards)
> - **Unit V**: Spark Streaming with Apache Kafka (Producer & Consumer Messaging Pipeline, Real-time Ingestion)
> - **Unit VI**: Spark ML Programming & PySpark (StringIndexer, OneHotEncoder, VectorAssembler, Logistic Regression, Decision Tree, Confusion Matrix, FPR, AUC)

---

## 📌 1. Project Overview

This project implements a complete, end-to-end Big Data aviation network analysis and flight delay classification system built with **Apache Spark (PySpark & GraphX)** and **Apache Kafka**.

Using a comprehensive real-world dataset of commercial airline flight delays (`Airline_Delay_Cause.csv` containing over 171,000 flight log records from the US Bureau of Transportation Statistics), the project addresses two core operational challenges in aviation:
1. **Predictive Classification**: Classifying whether an arriving flight will experience a **significant delay (> 15 minutes)** using machine learning algorithms.
2. **Network Bottleneck Identification**: Modeling the aviation route topology as a directed graph to calculate **in-degrees** and **out-degrees**, pinpointing systemic bottleneck hubs and airport congestion.

---

## 📂 2. Project Directory Structure

```text
flight_delay_analytics/
│
├── data/
│   └── Airline_Delay_Cause.csv          # Real BTS commercial airline delay dataset
│
├── model/
│   ├── best_flight_delay_model/         # Saved PySpark MLlib model
│   └── model_evaluation_summary.json    # JSON report of ML evaluation metrics
│
├── src/
│   ├── __init__.py
│   ├── eda_sql.py                       # Unit IV: Spark SQL queries & bottleneck analysis
│   ├── delay_classifier.py              # Unit VI: PySpark MLlib Classification Pipeline
│   ├── graph_analytics.py               # Unit III: Python Graph Network In/Out-Degree Analysis
│   ├── AirportGraphX.scala              # Unit III: Spark GraphX Scala implementation
│   └── kafka_stream_demo.py             # Unit V: Real-time Kafka flight delay streaming
│
├── requirements.txt                     # Python dependencies
├── summary.md                           # Comprehensive execution, results & viva report
└── Readme.md                            # Main project documentation
```

---

## ⚙️ 3. Prerequisites & Installation

### Requirements
- **Python**: Version 3.8 to 3.12 (Python 3.10 recommended)
- **Java**: Java JDK 8 or 11 (required for Apache Spark)
- **Apache Spark**: Spark 3.x (pre-configured in environment)

### Install Python Dependencies
Open PowerShell or your terminal in the project directory:
```bash
pip install -r requirements.txt
```

---

## 🚀 4. How to Run the Modules

### Step 1: Spark SQL & Exploratory Data Analysis (Unit IV)
Isolates carrier bottlenecks and top congested airports using Spark SQL and DataFrame aggregations:
```bash
python src/eda_sql.py
```

### Step 2: Machine Learning Binary Classification (Unit VI)
Executes feature indexing, one-hot encoding, vector assembly, model training (Logistic Regression vs. Decision Tree), confusion matrix computation, and AUC evaluation:
```bash
python src/delay_classifier.py
```

### Step 3: Graph Network & Congestion Hub Analytics (Unit III)
Models airports as vertices and flight routes as directed edges. Computes In-Degrees, Out-Degrees, and Congestion Risk Scores:
```bash
python src/graph_analytics.py
```

#### Optional: Run Scala GraphX in Spark Shell
```bash
spark-shell
```
Inside the interactive shell:
```scala
:load src/AirportGraphX.scala
```

### Step 4: Real-Time Delay Streaming Demo (Unit V)
Simulates Kafka flight delay event publishing and real-time streaming classification:
```bash
python src/kafka_stream_demo.py
```

---

## 📊 5. Key Results Summary

### A. Carrier Bottlenecks (Spark SQL)
- **Highest Delay Per Flight**: JetBlue Airways (`B6` - 15.64 min/flight, 23.4% delay rate), Frontier Airlines (`F9` - 14.64 min/flight).
- **Primary Root Causes**:
  - Late Aircraft Propagation: **38.96%**
  - Air Carrier Internal Delay: **33.90%**
  - National Aviation System (NAS) Traffic: **21.72%**
  - Severe Weather: **5.25%**
  - Security Incidents: **0.17%**

### B. Machine Learning Performance (PySpark MLlib)

| Performance Metric | Logistic Regression | Decision Tree Classifier (Best) |
| :--- | :--- | :--- |
| **Accuracy** | 75.10% | **75.48%** |
| **Precision** | 0.00% | **51.52%** |
| **Recall (TPR)** | 0.00% | **25.88%** (Detects 322 delayed flights) |
| **False Positive Rate (FPR)** | **0.00%** | 8.08% |
| **F1-Score** | 0.0000 | **0.3446** |
| **Area Under ROC (AUC)** | 0.5188 | **0.6536** (+13.48% AUC) |

### C. Network Graph Analysis (GraphX / Graph Analytics)
- **Top Connected Hubs (Total Degree)**:
  1. Chicago O'Hare (`ORD`): Total Degree = **376** (In: 8, Out: 368)
  2. Dallas/Fort Worth (`DFW`): Total Degree = **232** (In: 7, Out: 225)
  3. Atlanta Hartsfield-Jackson (`ATL`): Total Degree = **210** (In: 8, Out: 202)
- **Highest Congestion Risk Hubs**:
  - Newark Liberty (`EWR`): Risk Score = **13.28** (Avg Arrival Delay: 16.60 min)
  - Chicago O'Hare (`ORD`): Risk Score = **11.36** (Avg Arrival Delay: 14.20 min)
  - New York LaGuardia (`LGA`): Risk Score = **11.14** (Avg Arrival Delay: 15.92 min)

---

## 📖 Detailed Report
For the complete technical breakdown, step-by-step logs, theoretical justifications, and Viva-Voce preparation, refer to:
👉 **[summary.md](file:///d:/Project-14-aug-onwards/flight_delay_analytics/summary.md)**
