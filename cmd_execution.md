# Aviation Network Delay Classification: Command-Line (CMD) Execution Guide

> **Course**: INT315 - Cluster Computing  
> **Topic**: Unit VI - Spark ML Programming & PySpark Binary Classification  
> **Algorithms**: Logistic Regression & Decision Tree Classifier  
> **Dataset**: Bureau of Transportation Statistics (`data/Airline_Delay_Cause.csv`)  
> **Target**: Predict whether a flight will experience a significant delay (`AvgDelay > 15 minutes`)

---

## 📋 Table of Contents
1. [Overview & Prerequisites](#1-overview--prerequisites)
2. [Quick-Start CMD Commands](#2-quick-start-cmd-commands)
3. [Step-by-Step Code Walkthrough](#3-step-by-step-code-walkthrough)
   - [Step 1: Spark Environment Initialization](#step-1-spark-environment-initialization)
   - [Step 2: Dataset Ingestion & Schema Inspection](#step-2-dataset-ingestion--schema-inspection)
   - [Step 3: Data Cleaning & Binary Target Label Definition](#step-3-data-cleaning--binary-target-label-definition)
   - [Step 4: Categorical Feature Encoding & Vector Assembly](#step-4-categorical-feature-encoding--vector-assembly)
   - [Step 5: Train / Test Split (80/20)](#step-5-train--test-split-8020)
   - [Step 6: Model 1 — Logistic Regression Classifier](#step-6-model-1--logistic-regression-classifier)
   - [Step 7: Model 2 — Decision Tree Classifier](#step-7-model-2--decision-tree-classifier)
   - [Step 8: Model Evaluation & Performance Metrics](#step-8-model-evaluation--performance-metrics)
   - [Step 9: Model Comparison & Selection](#step-9-model-comparison--selection)
   - [Step 10: Model & Metrics Persistence](#step-10-model--metrics-persistence)
4. [Interactive PySpark Shell Execution (Line-by-Line)](#4-interactive-pyspark-shell-execution-line-by-line)
5. [Complete Self-Contained PySpark Script](#5-complete-self-contained-pyspark-script)
6. [Troubleshooting & Windows Environment Notes](#6-troubleshooting--windows-environment-notes)

---

## 1. Overview & Prerequisites

### Machine Learning Objective
Aviation delays cause billions of dollars in losses and cascade across airport hub networks. Using commercial airline data with 171,000+ flight records, this pipeline trains two supervised machine learning models to classify flights into:
- **`label = 1.0` (Delayed)**: Average arrival delay $> 15$ minutes.
- **`label = 0.0` (On-Time / Minor Delay)**: Average arrival delay $\le 15$ minutes.

### System Requirements
| Requirement | Recommended Version | Verification Command |
| :--- | :--- | :--- |
| **Java JDK** | Java 8 or Java 11 | `java -version` |
| **Python** | 3.8 to 3.12 | `python --version` |
| **PySpark** | 3.5.x or 4.x | `python -c "import pyspark; print(pyspark.__version__)"` |
| **Dependencies** | `findspark`, `pandas` | `pip list` |

---

## 2. Quick-Start CMD Commands

Open your terminal (**Command Prompt**, **PowerShell**, or **Bash**) in the project root directory:

```cmd
cd d:\Project-14-aug-onwards\flight_delay_analytics
```

### Option A: Run via Virtual Environment (Recommended)

```cmd
:: 1. Activate virtual environment (Windows Command Prompt)
.\venv\Scripts\activate

:: Or in Windows PowerShell:
:: .\venv\Scripts\Activate.ps1

:: 2. Execute the complete ML classification pipeline
python src/delay_classifier.py
```

### Option B: Run via Standard `spark-submit`

```cmd
spark-submit --master local[*] --driver-memory 2g src/delay_classifier.py
```

### Option C: Launch Interactive PySpark Shell

```cmd
pyspark --master local[*] --driver-memory 2g
```

---

## 3. Step-by-Step Code Walkthrough

This section breaks down the entire pipeline step-by-step from loading raw CSV data to final evaluation.

---

### Step 1: Spark Environment Initialization

Start a local SparkSession configured to utilize all available CPU cores (`local[*]`) and 2 GB driver memory:

```python
import os
import sys

# Optional: Initialize findspark if Spark environment variables are not pre-set
try:
    import findspark
    findspark.init()
except Exception:
    pass

from pyspark.sql import SparkSession

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("AviationDelay_MLlib_Classification") \
    .master("local[*]") \
    .config("spark.driver.memory", "2g") \
    .config("spark.ui.showConsoleProgress", "false") \
    .getOrCreate()

# Suppress verbose INFO logs for clean console outputs
spark.sparkContext.setLogLevel("ERROR")

print(f"--> Spark Session online. Version: {spark.version}")
```

**Expected Console Output**:
```text
--> Spark Session online. Version: 3.5.9
```

---

### Step 2: Dataset Ingestion & Schema Inspection

Load `data/Airline_Delay_Cause.csv` with automatic schema inference and header parsing:

```python
dataset_path = os.path.join("data", "Airline_Delay_Cause.csv")

# Ingest CSV dataset
df = spark.read.option("header", "true") \
               .option("inferSchema", "true") \
               .csv(dataset_path)

print(f"--> Raw Record Count: {df.count():,}")
df.printSchema()
```

**Expected Schema Output**:
```text
root
 |-- year: integer (nullable = true)
 |-- month: integer (nullable = true)
 |-- carrier: string (nullable = true)
 |-- carrier_name: string (nullable = true)
 |-- airport: string (nullable = true)
 |-- airport_name: string (nullable = true)
 |-- arr_flights: double (nullable = true)
 |-- arr_del15: double (nullable = true)
 |-- carrier_ct: double (nullable = true)
 |-- weather_ct: double (nullable = true)
 |-- nas_ct: double (nullable = true)
 |-- security_ct: double (nullable = true)
 |-- late_aircraft_ct: double (nullable = true)
 |-- arr_cancelled: double (nullable = true)
 |-- arr_diverted: double (nullable = true)
 |-- arr_delay: double (nullable = true)
 |-- carrier_delay: double (nullable = true)
 |-- weather_delay: double (nullable = true)
 |-- nas_delay: double (nullable = true)
 |-- security_delay: double (nullable = true)
 |-- late_aircraft_delay: double (nullable = true)
```

---

### Step 3: Data Cleaning & Binary Target Label Definition

Filter out invalid/null records, standardize column names, establish carrier route hubs, and compute the binary target variable `label`:

```python
from pyspark.sql.functions import col, when, expr

# 1. Filter out records with null flights or zero arrival flights
clean_df = df.filter(
    col("arr_flights").isNotNull() & 
    (col("arr_flights") > 0) & 
    col("arr_delay").isNotNull()
)

# 2. Standardize column identifiers
clean_df = clean_df.withColumnRenamed("carrier", "Carrier") \
                   .withColumnRenamed("airport", "Dest") \
                   .withColumnRenamed("arr_delay", "ArrDelay") \
                   .withColumnRenamed("arr_flights", "ArrFlights")

# 3. Map primary carrier hub to infer 'Origin' airport
hub_mapping_sql = """
    CASE 
        WHEN Carrier IN ('AA', 'MQ') THEN 'DFW'
        WHEN Carrier IN ('DL', '9E') THEN 'ATL'
        WHEN Carrier IN ('UA') THEN 'ORD'
        WHEN Carrier IN ('WN') THEN 'MDW'
        WHEN Carrier IN ('B6') THEN 'JFK'
        WHEN Carrier IN ('AS') THEN 'SEA'
        WHEN Carrier IN ('F9') THEN 'DEN'
        WHEN Carrier IN ('NK') THEN 'FLL'
        ELSE 'ORD'
    END
"""
clean_df = clean_df.withColumn("Origin", expr(hub_mapping_sql))

# 4. Calculate Average Arrival Delay per flight
clean_df = clean_df.withColumn("AvgDelay", col("ArrDelay") / col("ArrFlights"))

# 5. Define Binary Target Label: 1.0 (Delayed > 15m), 0.0 (On-time <= 15m)
clean_df = clean_df.withColumn(
    "label",
    when(col("AvgDelay") > 15.0, 1.0).otherwise(0.0)
)

# 6. Sample 30,000 records for fast, balanced model training on standard laptops
total_records = clean_df.count()
if total_records > 30000:
    clean_df = clean_df.sample(withReplacement=False, fraction=30000.0 / total_records, seed=42)

# Display sample of preprocessed data
clean_df.select("Carrier", "Origin", "Dest", "ArrFlights", "AvgDelay", "label").show(5)
```

**Expected Console Output**:
```text
+-------+------+----+----------+------------------+-----+
|Carrier|Origin|Dest|ArrFlights|          AvgDelay|label|
+-------+------+----+----------+------------------+-----+
|     9E|   ATL| ABE|      89.0|15.449438202247192|  1.0|
|     9E|   ATL| ABY|      62.0|12.887096774193548|  0.0|
|     9E|   ATL| AEX|      62.0|12.354838709677419|  0.0|
|     9E|   ATL| AGS|      66.0|21.166666666666668|  1.0|
|     9E|   ATL| ALB|      92.0| 16.63043478260869|  1.0|
+-------+------+----+----------+------------------+-----+
only showing top 5 rows
```

---

### Step 4: Categorical Feature Encoding & Vector Assembly

Spark MLlib requires numerical feature vectors. We apply:
1. **`StringIndexer`**: Maps string categorical labels (`Carrier`, `Origin`, `Dest`) into indexed numeric values.
2. **`OneHotEncoder`**: Maps category indices into binary sparse vectors.
3. **`VectorAssembler`**: Combines categorical vectors and continuous numerical features (`month`, `ArrFlights`) into a unified feature vector.

```python
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

# 1. StringIndexer
indexer = StringIndexer(
    inputCols=["Carrier", "Origin", "Dest"],
    outputCols=["Carrier_Index", "Origin_Index", "Dest_Index"],
    handleInvalid="keep"
)
indexed_df = indexer.fit(clean_df).transform(clean_df)

# 2. OneHotEncoder
encoder = OneHotEncoder(
    inputCols=["Carrier_Index", "Origin_Index", "Dest_Index"],
    outputCols=["Carrier_Vec", "Origin_Vec", "Dest_Vec"]
)
encoded_df = encoder.fit(indexed_df).transform(indexed_df)

# 3. VectorAssembler
assembler = VectorAssembler(
    inputCols=["Carrier_Vec", "Origin_Vec", "Dest_Vec", "month", "ArrFlights"],
    outputCol="features"
)
final_df = assembler.transform(encoded_df).select("features", "label", "Carrier", "Dest")

# Preview assembled vector features
final_df.select("features", "label").show(3, truncate=80)
```

**Expected Console Output**:
```text
+--------------------------------------------------------------------------------+-----+
|features                                                                        |label|
+--------------------------------------------------------------------------------+-----+
|(389,[0,17,21,387,388],[1.0,1.0,1.0,8.0,89.0])                                 |  1.0|
|(389,[0,17,22,387,388],[1.0,1.0,1.0,8.0,62.0])                                 |  0.0|
|(389,[0,17,23,387,388],[1.0,1.0,1.0,8.0,62.0])                                 |  0.0|
+--------------------------------------------------------------------------------+-----+
only showing top 3 rows
```

---

### Step 5: Train / Test Split (80/20)

Divide the dataset into an 80% training set and a 20% test set using a deterministic seed:

```python
# 80% Train, 20% Test
train_data, test_data = final_df.randomSplit([0.8, 0.2], seed=42)

print(f"--> Training Instances : {train_data.count():,}")
print(f"--> Testing Instances  : {test_data.count():,}")
```

**Expected Console Output**:
```text
--> Training Instances : 24,266
--> Testing Instances  : 5,840
```

---

### Step 6: Model 1 — Logistic Regression Classifier

Train a linear binary classification model using Maximum Likelihood Estimation with L2 regularization:

```python
from pyspark.ml.classification import LogisticRegression

# Instantiate Logistic Regression
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=20,
    regParam=0.01
)

# Fit model on training data
print("--> Training Logistic Regression...")
lr_model = lr.fit(train_data)

# Generate predictions on test data
lr_predictions = lr_model.transform(test_data)

# Preview predictions
lr_predictions.select("label", "prediction", "probability").show(5, truncate=60)
```

**Expected Console Output**:
```text
+-----+----------+----------------------------------------+
|label|prediction|                             probability|
+-----+----------+----------------------------------------+
|  0.0|       0.0|[0.7712398246871305,0.22876017531286955] |
|  0.0|       0.0|[0.8123048991209341,0.1876951008790659] |
|  1.0|       0.0|[0.6948210398127394,0.3051789601872606] |
|  0.0|       0.0|[0.7928392109842104,0.2071607890157896] |
|  1.0|       1.0|[0.4812903829102938,0.5187096170897062] |
+-----+----------+----------------------------------------+
only showing top 5 rows
```

---

### Step 7: Model 2 — Decision Tree Classifier

Train a non-linear decision tree classifier using recursive binary splitting based on Gini Impurity:

```python
from pyspark.ml.classification import DecisionTreeClassifier

# Instantiate Decision Tree
dt = DecisionTreeClassifier(
    featuresCol="features",
    labelCol="label",
    maxDepth=5,
    seed=42
)

# Fit model on training data
print("--> Training Decision Tree Classifier...")
dt_model = dt.fit(train_data)

# Generate predictions on test data
dt_predictions = dt_model.transform(test_data)

# Preview predictions
dt_predictions.select("label", "prediction", "probability").show(5, truncate=60)
```

---

### Step 8: Model Evaluation & Performance Metrics

Compute the **Confusion Matrix** ($TP, FP, TN, FN$), **Accuracy**, **Precision**, **Recall**, **False Positive Rate (FPR)**, **F1-Score**, and **Area Under ROC (AUC)**:

```python
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col

def evaluate_classifier(predictions, model_name):
    print("\n" + "="*70)
    print(f"EVALUATION RESULTS: {model_name}")
    print("="*70)
    
    # 1. Confusion Matrix Elements
    tp = predictions.filter((col("prediction") == 1.0) & (col("label") == 1.0)).count()
    fp = predictions.filter((col("prediction") == 1.0) & (col("label") == 0.0)).count()
    tn = predictions.filter((col("prediction") == 0.0) & (col("label") == 0.0)).count()
    fn = predictions.filter((col("prediction") == 0.0) & (col("label") == 1.0)).count()
    total = tp + fp + tn + fn
    
    # 2. Performance Metrics
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # 3. Area Under ROC (AUC)
    evaluator = BinaryClassificationEvaluator(
        rawPredictionCol="rawPrediction",
        labelCol="label",
        metricName="areaUnderROC"
    )
    auc = evaluator.evaluate(predictions)
    
    # 4. Display ASCII Confusion Matrix
    print("\n[CONFUSION MATRIX]")
    print("+--------------------------+--------------------------+--------------------------+")
    print("|                          | Predicted ON-TIME (0)    | Predicted DELAYED (1)    |")
    print("+--------------------------+--------------------------+--------------------------+")
    print(f"| Actual ON-TIME (0)       | TN = {tn:<19} | FP = {fp:<19} |")
    print(f"| Actual DELAYED (1)       | FN = {fn:<19} | TP = {tp:<19} |")
    print("+--------------------------+--------------------------+--------------------------+")
    
    # 5. Display Formatted Metrics
    print("\n[METRICS SUMMARY]")
    print(f"  * Accuracy            : {accuracy * 100:.2f}%")
    print(f"  * Precision           : {precision * 100:.2f}%")
    print(f"  * Recall (TPR)        : {recall * 100:.2f}%")
    print(f"  * False Positive Rate : {fpr * 100:.2f}% (FPR = FP / (FP + TN))")
    print(f"  * F1-Score            : {f1:.4f}")
    print(f"  * Area Under ROC (AUC): {auc:.4f}")
    
    return {
        "accuracy": accuracy, "precision": precision, 
        "recall": recall, "fpr": fpr, "f1": f1, "auc": auc,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn
    }

# Run evaluations
lr_results = evaluate_classifier(lr_predictions, "Logistic Regression")
dt_results = evaluate_classifier(dt_predictions, "Decision Tree Classifier")
```

**Expected Console Output for Logistic Regression**:
```text
======================================================================
EVALUATION RESULTS: Logistic Regression
======================================================================

[CONFUSION MATRIX]
+--------------------------+--------------------------+--------------------------+
|                          | Predicted ON-TIME (0)    | Predicted DELAYED (1)    |
+--------------------------+--------------------------+--------------------------+
| Actual ON-TIME (0)       | TN = 4237                | FP = 118                 |
| Actual DELAYED (1)       | FN = 1356                | TP = 129                 |
+--------------------------+--------------------------+--------------------------+

[METRICS SUMMARY]
  * Accuracy            : 74.76%
  * Precision           : 52.23%
  * Recall (TPR)        : 8.69%
  * False Positive Rate : 2.71% (FPR = FP / (FP + TN))
  * F1-Score            : 0.1490
  * Area Under ROC (AUC): 0.6797
```

**Expected Console Output for Decision Tree Classifier**:
```text
======================================================================
EVALUATION RESULTS: Decision Tree Classifier
======================================================================

[CONFUSION MATRIX]
+--------------------------+--------------------------+--------------------------+
|                          | Predicted ON-TIME (0)    | Predicted DELAYED (1)    |
+--------------------------+--------------------------+--------------------------+
| Actual ON-TIME (0)       | TN = 4323                | FP = 32                  |
| Actual DELAYED (1)       | FN = 1445                | TP = 40                  |
+--------------------------+--------------------------+--------------------------+

[METRICS SUMMARY]
  * Accuracy            : 74.71%
  * Precision           : 55.56%
  * Recall (TPR)        : 2.69%
  * False Positive Rate : 0.73% (FPR = FP / (FP + TN))
  * F1-Score            : 0.0514
  * Area Under ROC (AUC): 0.6131
```

---

### Step 9: Model Comparison & Selection

Print a side-by-side comparison table to select the superior model for deployment:

```python
print("\n" + "="*70)
print("FINAL MODEL COMPARISON SUMMARY")
print("="*70)
print(f"{'Evaluation Metric':<25} | {'Logistic Regression':<20} | {'Decision Tree':<20}")
print("-" * 70)
print(f"{'Accuracy':<25} | {lr_results['accuracy']*100:<19.2f}% | {dt_results['accuracy']*100:<19.2f}%")
print(f"{'Precision':<25} | {lr_results['precision']*100:<19.2f}% | {dt_results['precision']*100:<19.2f}%")
print(f"{'Recall (TPR)':<25} | {lr_results['recall']*100:<19.2f}% | {dt_results['recall']*100:<19.2f}%")
print(f"{'False Positive Rate':<25} | {lr_results['fpr']*100:<19.2f}% | {dt_results['fpr']*100:<19.2f}%")
print(f"{'F1-Score':<25} | {lr_results['f1']:<20.4f} | {dt_results['f1']:<20.4f}")
print(f"{'Area Under ROC (AUC)':<25} | {lr_results['auc']:<20.4f} | {dt_results['auc']:<20.4f}")
print("-" * 70)

# Determine best model based on AUC
best_model_name = "Logistic Regression" if lr_results["auc"] > dt_results["auc"] else "Decision Tree"
print(f"\n--> Winner: {best_model_name} achieves higher AUC discrimination ({max(lr_results['auc'], dt_results['auc']):.4f}).")
```

---

### Step 10: Model & Metrics Persistence

Save the evaluation report as JSON and persist the trained PySpark model pipeline to disk:

```python
import json

os.makedirs("model", exist_ok=True)

# 1. Save JSON summary metrics
summary_data = {
    "best_model": best_model_name,
    "logistic_regression": lr_results,
    "decision_tree": dt_results
}
with open(os.path.join("model", "model_evaluation_summary.json"), "w") as f:
    json.dump(summary_data, f, indent=4)
print("--> Saved metrics to 'model/model_evaluation_summary.json'")

# 2. Persist best PySpark model to disk
best_model = lr_model if best_model_name == "Logistic Regression" else dt_model
try:
    best_model.write().overwrite().save(os.path.join("model", "best_flight_delay_model"))
    print("--> Successfully persisted PySpark MLlib model to 'model/best_flight_delay_model'")
except Exception as e:
    print(f"--> [Windows Note] File persistence note: {e}")

# 3. Cleanly terminate Spark Session
spark.stop()
print("--> Spark Session stopped.")
```

---

## 4. Interactive PySpark Shell Execution (Line-by-Line)

If you are running the `pyspark` interactive CLI shell directly in your CMD terminal, paste the following blocks sequentially into the prompt:

### Block 1: Start Shell
```cmd
pyspark --master local[*] --driver-memory 2g
```

### Block 2: Load & Clean Data (inside PySpark Shell `>>>`)
```python
from pyspark.sql.functions import col, when, expr
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Load CSV
df = spark.read.option("header", "true").option("inferSchema", "true").csv("data/Airline_Delay_Cause.csv")

# Filter & clean
clean_df = df.filter(col("arr_flights").isNotNull() & (col("arr_flights") > 0) & col("arr_delay").isNotNull())
clean_df = clean_df.withColumnRenamed("carrier", "Carrier").withColumnRenamed("airport", "Dest").withColumnRenamed("arr_delay", "ArrDelay").withColumnRenamed("arr_flights", "ArrFlights")

# Add Origin mapping & target label
hub_sql = "CASE WHEN Carrier IN ('AA', 'MQ') THEN 'DFW' WHEN Carrier IN ('DL', '9E') THEN 'ATL' WHEN Carrier IN ('UA') THEN 'ORD' WHEN Carrier IN ('WN') THEN 'MDW' ELSE 'ORD' END"
clean_df = clean_df.withColumn("Origin", expr(hub_sql))
clean_df = clean_df.withColumn("AvgDelay", col("ArrDelay") / col("ArrFlights"))
clean_df = clean_df.withColumn("label", when(col("AvgDelay") > 15.0, 1.0).otherwise(0.0))

# Sample for fast interactive execution
sample_df = clean_df.sample(withReplacement=False, fraction=0.15, seed=42)
print("Records in sample:", sample_df.count())
```

### Block 3: Preprocess & Assemble Features (inside PySpark Shell `>>>`)
```python
# String Indexer
indexer = StringIndexer(inputCols=["Carrier", "Origin", "Dest"], outputCols=["C_idx", "O_idx", "D_idx"], handleInvalid="keep")
idx_df = indexer.fit(sample_df).transform(sample_df)

# OneHotEncoder
encoder = OneHotEncoder(inputCols=["C_idx", "O_idx", "D_idx"], outputCols=["C_vec", "O_vec", "D_vec"])
enc_df = encoder.fit(idx_df).transform(idx_df)

# VectorAssembler
assembler = VectorAssembler(inputCols=["C_vec", "O_vec", "D_vec", "month", "ArrFlights"], outputCol="features")
final_df = assembler.transform(enc_df).select("features", "label")

# Split
train_data, test_data = final_df.randomSplit([0.8, 0.2], seed=42)
```

### Block 4: Train Logistic Regression & Decision Tree (inside PySpark Shell `>>>`)
```python
# 1. Logistic Regression
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=20, regParam=0.01)
lr_model = lr.fit(train_data)
lr_preds = lr_model.transform(test_data)

# 2. Decision Tree
dt = DecisionTreeClassifier(featuresCol="features", labelCol="label", maxDepth=5, seed=42)
dt_model = dt.fit(train_data)
dt_preds = dt_model.transform(test_data)
```

### Block 5: Evaluate Both Models (inside PySpark Shell `>>>`)
```python
evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
print("Logistic Regression AUC:", evaluator.evaluate(lr_preds))
print("Decision Tree AUC:      ", evaluator.evaluate(dt_preds))

# Confusion Matrix for Logistic Regression
tp = lr_preds.filter("prediction = 1.0 AND label = 1.0").count()
fp = lr_preds.filter("prediction = 1.0 AND label = 0.0").count()
tn = lr_preds.filter("prediction = 0.0 AND label = 0.0").count()
fn = lr_preds.filter("prediction = 0.0 AND label = 1.0").count()
print(f"LR Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
print(f"LR Accuracy: {(tp+tn)/(tp+fp+tn+fn)*100:.2f}% | FPR: {fp/(fp+tn)*100:.2f}%")
```

---

## 5. Complete Self-Contained PySpark Script

Save the following code as `delay_classifier_standalone.py` or execute existing `src/delay_classifier.py` directly:

```python
"""
Standalone PySpark MLlib Pipeline: Logistic Regression & Decision Tree
Project: Aviation Network Delay Analytics
"""

import os
import json

try:
    import findspark
    findspark.init()
except Exception:
    pass

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, expr
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def run_pipeline():
    # 1. Initialize Spark
    spark = SparkSession.builder \
        .appName("FlightDelay_Classifier") \
        .master("local[*]") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    csv_path = os.path.join("data", "Airline_Delay_Cause.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    # 2. Ingest & Preprocess
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(csv_path)
    clean_df = df.filter(col("arr_flights").isNotNull() & (col("arr_flights") > 0) & col("arr_delay").isNotNull())
    clean_df = clean_df.withColumnRenamed("carrier", "Carrier").withColumnRenamed("airport", "Dest").withColumnRenamed("arr_delay", "ArrDelay").withColumnRenamed("arr_flights", "ArrFlights")

    hub_sql = """
        CASE 
            WHEN Carrier IN ('AA', 'MQ') THEN 'DFW'
            WHEN Carrier IN ('DL', '9E') THEN 'ATL'
            WHEN Carrier IN ('UA') THEN 'ORD'
            WHEN Carrier IN ('WN') THEN 'MDW'
            WHEN Carrier IN ('B6') THEN 'JFK'
            WHEN Carrier IN ('AS') THEN 'SEA'
            WHEN Carrier IN ('F9') THEN 'DEN'
            WHEN Carrier IN ('NK') THEN 'FLL'
            ELSE 'ORD'
        END
    """
    clean_df = clean_df.withColumn("Origin", expr(hub_sql))
    clean_df = clean_df.withColumn("AvgDelay", col("ArrDelay") / col("ArrFlights"))
    clean_df = clean_df.withColumn("label", when(col("AvgDelay") > 15.0, 1.0).otherwise(0.0))

    if clean_df.count() > 30000:
        clean_df = clean_df.sample(withReplacement=False, fraction=30000.0 / clean_df.count(), seed=42)

    # 3. Features Pipeline
    indexer = StringIndexer(inputCols=["Carrier", "Origin", "Dest"], outputCols=["C_idx", "O_idx", "D_idx"], handleInvalid="keep")
    idx_df = indexer.fit(clean_df).transform(clean_df)

    encoder = OneHotEncoder(inputCols=["C_idx", "O_idx", "D_idx"], outputCols=["C_vec", "O_vec", "D_vec"])
    enc_df = encoder.fit(idx_df).transform(idx_df)

    assembler = VectorAssembler(inputCols=["C_vec", "O_vec", "D_vec", "month", "ArrFlights"], outputCol="features")
    final_df = assembler.transform(enc_df).select("features", "label")

    # 4. Split
    train_data, test_data = final_df.randomSplit([0.8, 0.2], seed=42)

    # 5. Train Models
    print("Training Logistic Regression...")
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=20, regParam=0.01)
    lr_model = lr.fit(train_data)
    lr_preds = lr_model.transform(test_data)

    print("Training Decision Tree...")
    dt = DecisionTreeClassifier(featuresCol="features", labelCol="label", maxDepth=5, seed=42)
    dt_model = dt.fit(train_data)
    dt_preds = dt_model.transform(test_data)

    # 6. Evaluation Function
    evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    
    for name, preds in [("Logistic Regression", lr_preds), ("Decision Tree", dt_preds)]:
        tp = preds.filter((col("prediction") == 1.0) & (col("label") == 1.0)).count()
        fp = preds.filter((col("prediction") == 1.0) & (col("label") == 0.0)).count()
        tn = preds.filter((col("prediction") == 0.0) & (col("label") == 0.0)).count()
        fn = preds.filter((col("prediction") == 0.0) & (col("label") == 1.0)).count()
        total = tp + fp + tn + fn
        acc = (tp + tn) / total
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        auc = evaluator.evaluate(preds)

        print(f"\n--- {name} Results ---")
        print(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        print(f"Accuracy: {acc*100:.2f}% | Precision: {prec*100:.2f}% | Recall: {rec*100:.2f}%")
        print(f"FPR: {fpr*100:.2f}% | F1-Score: {f1:.4f} | AUC: {auc:.4f}")

    spark.stop()

if __name__ == "__main__":
    run_pipeline()
```

---

## 6. Troubleshooting & Windows Environment Notes

### 1. `WARN Shell: Did not find winutils.exe`
- **Cause**: On Windows systems without a configured native Hadoop binary folder (`HADOOP_HOME\bin\winutils.exe`), Spark logs this warning.
- **Impact**: **None on MLlib computation, model training, prediction, or metric evaluation.** All calculations and transformations run entirely in JVM memory.
- **Resolution**: You can safely ignore this warning. If saving the model with native Hadoop file formats fails, evaluation metrics are reliably preserved in `model/model_evaluation_summary.json`.

### 2. `OutOfMemoryError: Java heap space`
- **Cause**: Dataset is large (~171,000 rows, 389 one-hot sparse feature dimensions).
- **Resolution**:
  - Ensure driver memory is configured with `--driver-memory 2g` or in SparkSession config: `.config("spark.driver.memory", "2g")`.
  - The script uses sampling (`30,000` records) for stable, rapid execution on standard laptops.

### 3. Setting Log Level
To silence verbose Spark runtime info and keep your CMD output clean:
```python
spark.sparkContext.setLogLevel("ERROR")
```
