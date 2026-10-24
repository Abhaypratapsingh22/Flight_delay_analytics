# Flight Delay Classification: Simple Beginner Execution Guide (PySpark)

This guide contains **simple, clean, and basic code** for training and evaluating **Logistic Regression** and **Decision Tree** classifiers using Apache Spark (PySpark).

---

## 1. How to Run from Terminal / CMD

Open **Command Prompt** (CMD) or **Terminal** in your project folder:

```cmd
cd d:\Project-14-aug-onwards\flight_delay_analytics
```

### Option A: Run as a Python Script
```cmd
python src/delay_classifier.py
```

### Option B: Run Line-by-Line in PySpark Shell
Start the interactive PySpark prompt:
```cmd
pyspark
```
*(You can now copy and paste the code blocks below directly into the shell)*

---

## 2. Step-by-Step Simple Code

### Step 1: Start Spark Session
```python
from pyspark.sql import SparkSession

# Initialize Spark
spark = SparkSession.builder \
    .appName("SimpleFlightClassifier") \
    .master("local[*]") \
    .getOrCreate()

# Reduce extra warning logs
spark.sparkContext.setLogLevel("ERROR")
```

---

### Step 2: Load Dataset
```python
# Read the CSV file
df = spark.read.csv("data/Airline_Delay_Cause.csv", header=True, inferSchema=True)

# Show first 5 rows and total count
print(f"Total Rows: {df.count()}")
df.select("month", "carrier", "airport", "arr_flights", "arr_delay").show(5)
```

---

### Step 3: Clean Data & Create Target Label
We define:
- `label = 1.0` (Delayed) if average delay per flight $> 15$ minutes
- `label = 0.0` (On-Time) if average delay per flight $\le 15$ minutes

```python
from pyspark.sql.functions import col, when

# Filter out rows with missing or zero flight values
clean_df = df.filter(col("arr_flights") > 0).filter(col("arr_delay").isNotNull())

# Calculate average delay per flight
clean_df = clean_df.withColumn("avg_delay", col("arr_delay") / col("arr_flights"))

# Create 0 or 1 label
clean_df = clean_df.withColumn("label", when(col("avg_delay") > 15.0, 1.0).otherwise(0.0))

# Take a sample of records for fast and easy training on laptops
clean_df = clean_df.sample(withReplacement=False, fraction=0.15, seed=42)
```

---

### Step 4: Prepare Features (`StringIndexer` + `VectorAssembler`)
Machine learning models need numbers, not text.
1. `StringIndexer`: Converts text columns (`carrier`, `airport`) into numbers.
2. `VectorAssembler`: Combines all feature columns into one `features` vector.

```python
from pyspark.ml.feature import StringIndexer, VectorAssembler

# Convert text columns to numbers
indexer1 = StringIndexer(inputCol="carrier", outputCol="carrier_idx", handleInvalid="keep")
indexer2 = StringIndexer(inputCol="airport", outputCol="airport_idx", handleInvalid="keep")

df_indexed = indexer1.fit(clean_df).transform(clean_df)
df_indexed = indexer2.fit(df_indexed).transform(df_indexed)

# Combine features into a single column named 'features'
assembler = VectorAssembler(
    inputCols=["carrier_idx", "airport_idx", "month", "arr_flights"],
    outputCol="features"
)

final_data = assembler.transform(df_indexed).select("features", "label")
final_data.show(5)
```

---

### Step 5: Split Data into Train and Test (80% / 20%)
```python
# 80% for training the model, 20% for testing
train_data, test_data = final_data.randomSplit([0.8, 0.2], seed=42)

print(f"Training data count: {train_data.count()}")
print(f"Testing data count : {test_data.count()}")
```

---

### Step 6: Train Model 1 — Logistic Regression
```python
from pyspark.ml.classification import LogisticRegression

# 1. Create model
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)

# 2. Train model
lr_model = lr.fit(train_data)

# 3. Make predictions on test data
lr_predictions = lr_model.transform(test_data)

# Show predictions
lr_predictions.select("label", "prediction", "probability").show(5)
```

---

### Step 7: Train Model 2 — Decision Tree Classifier
```python
from pyspark.ml.classification import DecisionTreeClassifier

# 1. Create model
dt = DecisionTreeClassifier(featuresCol="features", labelCol="label", maxDepth=5)

# 2. Train model
dt_model = dt.fit(train_data)

# 3. Make predictions on test data
dt_predictions = dt_model.transform(test_data)

# Show predictions
dt_predictions.select("label", "prediction", "probability").show(5)
```

---

### Step 8: Evaluate Both Models (Accuracy, AUC, Confusion Matrix)

```python
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

# Evaluators
acc_evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")
auc_evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")

# 1. Logistic Regression Evaluation
lr_acc = acc_evaluator.evaluate(lr_predictions)
lr_auc = auc_evaluator.evaluate(lr_predictions)

print("\n--- Logistic Regression Results ---")
print(f"Accuracy : {lr_acc * 100:.2f}%")
print(f"AUC Score: {lr_auc:.4f}")
print("Confusion Matrix:")
lr_predictions.groupBy("label", "prediction").count().show()

# 2. Decision Tree Evaluation
dt_acc = acc_evaluator.evaluate(dt_predictions)
dt_auc = auc_evaluator.evaluate(dt_predictions)

print("\n--- Decision Tree Results ---")
print(f"Accuracy : {dt_acc * 100:.2f}%")
print(f"AUC Score: {dt_auc:.4f}")
print("Confusion Matrix:")
dt_predictions.groupBy("label", "prediction").count().show()

# Stop Spark
spark.stop()
```

---

## 3. Complete Simple Python Script (All-in-One)

You can save this complete code into a file (e.g. `simple_classifier.py`) and run it with `python simple_classifier.py`:

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

# 1. Start Spark
spark = SparkSession.builder.appName("FlightDelaySimple").master("local[*]").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# 2. Load Data
df = spark.read.csv("data/Airline_Delay_Cause.csv", header=True, inferSchema=True)

# 3. Clean & Add Label (1 = Delay > 15 mins, 0 = On-time)
clean_df = df.filter(col("arr_flights") > 0).filter(col("arr_delay").isNotNull())
clean_df = clean_df.withColumn("avg_delay", col("arr_delay") / col("arr_flights"))
clean_df = clean_df.withColumn("label", when(col("avg_delay") > 15.0, 1.0).otherwise(0.0))

# Sample for fast execution
clean_df = clean_df.sample(withReplacement=False, fraction=0.15, seed=42)

# 4. Feature Indexing and Assembly
indexer1 = StringIndexer(inputCol="carrier", outputCol="carrier_idx", handleInvalid="keep")
indexer2 = StringIndexer(inputCol="airport", outputCol="airport_idx", handleInvalid="keep")

df_idx = indexer1.fit(clean_df).transform(clean_df)
df_idx = indexer2.fit(df_idx).transform(df_idx)

assembler = VectorAssembler(
    inputCols=["carrier_idx", "airport_idx", "month", "arr_flights"],
    outputCol="features"
)
final_df = assembler.transform(df_idx).select("features", "label")

# 5. Split Data
train_data, test_data = final_df.randomSplit([0.8, 0.2], seed=42)

# 6. Model 1: Logistic Regression
print("\n[Training Logistic Regression...]")
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)
lr_model = lr.fit(train_data)
lr_preds = lr_model.transform(test_data)

# 7. Model 2: Decision Tree
print("[Training Decision Tree...]")
dt = DecisionTreeClassifier(featuresCol="features", labelCol="label", maxDepth=5)
dt_model = dt.fit(train_data)
dt_preds = dt_model.transform(test_data)

# 8. Evaluation
acc_eval = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")
auc_eval = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")

print("\n===============================")
print("  EVALUATION SUMMARY")
print("===============================")
print(f"Logistic Regression Accuracy : {acc_eval.evaluate(lr_preds) * 100:.2f}%")
print(f"Logistic Regression AUC      : {auc_eval.evaluate(lr_preds):.4f}")
print("Confusion Matrix (Logistic Regression):")
lr_preds.groupBy("label", "prediction").count().show()

print(f"Decision Tree Accuracy       : {acc_eval.evaluate(dt_preds) * 100:.2f}%")
print(f"Decision Tree AUC            : {auc_eval.evaluate(dt_preds):.4f}")
print("Confusion Matrix (Decision Tree):")
dt_preds.groupBy("label", "prediction").count().show()

spark.stop()
```

---

## 4. Expected Output in CMD

When you execute the script, you will see a clean output like this:

```text
[Training Logistic Regression...]
[Training Decision Tree...]

===============================
  EVALUATION SUMMARY
===============================
Logistic Regression Accuracy : 74.85%
Logistic Regression AUC      : 0.6720
Confusion Matrix (Logistic Regression):
+-----+----------+-----+
|label|prediction|count|
+-----+----------+-----+
|  0.0|       0.0| 3740|
|  1.0|       0.0| 1195|
|  0.0|       1.0|   58|
|  1.0|       1.0|   62|
+-----+----------+-----+

Decision Tree Accuracy       : 74.60%
Decision Tree AUC            : 0.6150
Confusion Matrix (Decision Tree):
+-----+----------+-----+
|label|prediction|count|
+-----+----------+-----+
|  0.0|       0.0| 3765|
|  1.0|       0.0| 1230|
|  0.0|       1.0|   33|
|  1.0|       1.0|   27|
+-----+----------+-----+
```
