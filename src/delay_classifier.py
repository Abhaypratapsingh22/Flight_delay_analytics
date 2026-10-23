"""
================================================================================
Aviation Network Delay Analytics - Binary Classification with PySpark MLlib
Mapped Unit: Unit VI (Logistic Regression, Decision Tree, Confusion Matrix, AUC)
Course: INT315 - Cluster Computing
================================================================================
Description:
    This script builds a machine learning pipeline using PySpark MLlib:
    1. Preprocessing: Converts Carrier, Origin, and Destination into indexed
       and one-hot encoded features using StringIndexer and OneHotEncoder.
    2. Feature Assembly: Bundles all features into a single VectorAssembler vector.
    3. Binary Classification Target:
       - Delayed (1) : Average arrival delay > 15 minutes
       - On-time (0) : Average arrival delay <= 15 minutes
    4. Model Training: Trains Logistic Regression and Decision Tree Classifier.
    5. Evaluation & Performance Matrix:
       - Confusion Matrix (TP, FP, TN, FN)
       - False Positive Rate (FPR)
       - Accuracy, Precision, Recall, F1-Score
       - Area Under ROC (AUC)
    6. Model Persistence: Saves the trained model to disk in the 'model/' directory.
================================================================================
"""

import os
import sys

# ------------------------------------------------------------------------------
# STEP 1: INITIALIZE SPARK ENVIRONMENT
# ------------------------------------------------------------------------------
try:
    import findspark
    findspark.init()
except Exception as e:
    print(f"[INFO] findspark initialization: {e}")

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, round, expr
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator


def create_spark_session():
    """
    Creates and returns a local SparkSession configured for MLlib workloads.
    """
    print("\n" + "="*80)
    print("STEP 1: INITIALIZING PYSPARK SESSION FOR MACHINE LEARNING")
    print("="*80)
    spark = SparkSession.builder \
        .appName("AviationDelay_MLlib_Classification") \
        .master("local[*]") \
        .config("spark.driver.memory", "2g") \
        .config("spark.ui.showConsoleProgress", "false") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    print(f"--> Spark Session online. Version: {spark.version}")
    return spark


def prepare_data(spark, csv_path):
    """
    STEP 2: INGEST AND PREPROCESS DATA
    - Ingests flight logs
    - Maps carrier primary hubs to establish Origin -> Destination routes
    - Defines binary target label:
        1 = Significant delay (> 15 minutes)
        0 = On-time / Minor delay (<= 15 minutes)
    """
    print("\n" + "="*80)
    print("STEP 2: INGESTING FLIGHT LOGS & PREPARING TARGET LABELS")
    print("="*80)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    # Read CSV
    df = spark.read.option("header", "true") \
                   .option("inferSchema", "true") \
                   .csv(csv_path)

    # Filter out missing records
    clean_df = df.filter(
        col("arr_flights").isNotNull() & 
        (col("arr_flights") > 0) & 
        col("arr_delay").isNotNull()
    )

    # Standardize column names
    clean_df = clean_df.withColumnRenamed("carrier", "Carrier") \
                       .withColumnRenamed("airport", "Dest") \
                       .withColumnRenamed("arr_delay", "ArrDelay") \
                       .withColumnRenamed("arr_flights", "ArrFlights")

    # Map Carrier Hubs to generate 'Origin' airport for each carrier
    # (e.g. DL -> ATL, AA -> DFW, UA -> ORD, WN -> MDW, B6 -> JFK, AS -> SEA)
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

    # Calculate average delay per flight
    clean_df = clean_df.withColumn("AvgDelay", col("ArrDelay") / col("ArrFlights"))

    # Define binary classification target label:
    # 1 = Delayed (> 15 minutes)
    # 0 = On-time (<= 15 minutes)
    clean_df = clean_df.withColumn(
        "label",
        when(col("AvgDelay") > 15.0, 1.0).otherwise(0.0)
    )

    total_count = clean_df.count()
    delayed_count = clean_df.filter(col("label") == 1.0).count()
    ontime_count = total_count - delayed_count

    print(f"--> Total flight records loaded: {total_count:,}")
    print(f"--> Target Label Distribution:")
    print(f"    - On-Time   (label = 0) : {ontime_count:,} ({ontime_count / total_count * 100:.2f}%)")
    print(f"    - Delayed   (label = 1) : {delayed_count:,} ({delayed_count / total_count * 100:.2f}%)")

    # For faster execution and stable training on standard laptops,
    # sample 30,000 records if dataset is very large
    if total_count > 30000:
        print("\n[INFO] Sampling 30,000 records for fast, beginner-friendly model training...")
        clean_df = clean_df.sample(withReplacement=False, fraction=30000.0 / total_count, seed=42)

    return clean_df


def build_feature_pipeline(df):
    """
    STEP 3: CATEGORICAL ENCODING & VECTOR ASSEMBLER
    Converts Carrier, Origin, and Destination into indexed and one-hot encoded
    vectors using StringIndexer and OneHotEncoder, then combines with numerical features.
    """
    print("\n" + "="*80)
    print("STEP 3: FEATURE ENGINEERING (StringIndexer + OneHotEncoder + VectorAssembler)")
    print("="*80)

    # 1. StringIndexer: Assigns a unique integer index to each category
    print("--> Applying StringIndexer to ['Carrier', 'Origin', 'Dest']...")
    indexer = StringIndexer(
        inputCols=["Carrier", "Origin", "Dest"],
        outputCols=["Carrier_Index", "Origin_Index", "Dest_Index"],
        handleInvalid="keep"
    )
    indexed_df = indexer.fit(df).transform(df)

    # 2. OneHotEncoder: Converts category indices into binary one-hot vectors
    print("--> Applying OneHotEncoder to categorical index columns...")
    encoder = OneHotEncoder(
        inputCols=["Carrier_Index", "Origin_Index", "Dest_Index"],
        outputCols=["Carrier_Vec", "Origin_Vec", "Dest_Vec"]
    )
    encoded_df = encoder.fit(indexed_df).transform(indexed_df)

    # 3. VectorAssembler: Combines all feature columns into a single vector
    print("--> Assembling all features into 'features' vector...")
    assembler = VectorAssembler(
        inputCols=["Carrier_Vec", "Origin_Vec", "Dest_Vec", "month", "ArrFlights"],
        outputCol="features"
    )
    final_df = assembler.transform(encoded_df).select("features", "label", "Carrier", "Dest")

    return final_df


def evaluate_model(predictions, model_name):
    """
    STEP 4: COMPUTE PERFORMANCE MATRIX
    Calculates:
    - Confusion Matrix (TP, FP, TN, FN)
    - False Positive Rate (FPR)
    - True Positive Rate (Recall)
    - Precision, Accuracy, F1-Score
    - Area Under ROC (AUC)
    """
    print("\n" + "-"*80)
    print(f"PERFORMANCE EVALUATION: {model_name}")
    print("-"*80)

    # 1. Compute Confusion Matrix Elements
    tp = predictions.filter((col("prediction") == 1.0) & (col("label") == 1.0)).count()
    fp = predictions.filter((col("prediction") == 1.0) & (col("label") == 0.0)).count()
    tn = predictions.filter((col("prediction") == 0.0) & (col("label") == 0.0)).count()
    fn = predictions.filter((col("prediction") == 0.0) & (col("label") == 1.0)).count()
    total = tp + fp + tn + fn

    # 2. Compute Metrics using Beginner Formulas
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # 3. Compute AUC using PySpark BinaryClassificationEvaluator
    roc_evaluator = BinaryClassificationEvaluator(
        rawPredictionCol="rawPrediction",
        labelCol="label",
        metricName="areaUnderROC"
    )
    auc = roc_evaluator.evaluate(predictions)

    # 4. Display Confusion Matrix in Beginner-Readable Table (ASCII compatible)
    print("\n[CONFUSION MATRIX]")
    print("+--------------------------+--------------------------+--------------------------+")
    print("|                          | Predicted ON-TIME (0)    | Predicted DELAYED (1)    |")
    print("+--------------------------+--------------------------+--------------------------+")
    print(f"| Actual ON-TIME (0)       | TN = {tn:<19} | FP = {fp:<19} |")
    print(f"| Actual DELAYED (1)       | FN = {fn:<19} | TP = {tp:<19} |")
    print("+--------------------------+--------------------------+--------------------------+")

    # 5. Display Evaluation Metrics
    print("\n[PERFORMANCE METRICS]")
    print(f"  - Accuracy            : {accuracy * 100:.2f}%")
    print(f"  - Precision           : {precision * 100:.2f}%")
    print(f"  - Recall (TPR)        : {recall * 100:.2f}%")
    print(f"  - False Positive Rate : {fpr * 100:.2f}% (FPR = FP / (FP + TN))")
    print(f"  - F1-Score            : {f1:.4f}")
    print(f"  - Area Under ROC (AUC): {auc:.4f}")

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "f1": f1,
        "auc": auc,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn
    }


def train_and_evaluate_models(data_df, model_save_dir):
    """
    STEP 5: TRAIN LOGISTIC REGRESSION AND DECISION TREE CLASSIFIERS
    Splits data 80/20, trains both models, compares results, and saves the best model.
    """
    print("\n" + "="*80)
    print("STEP 5: SPLITTING DATA (80% Train, 20% Test) & MODEL TRAINING")
    print("="*80)

    train_data, test_data = data_df.randomSplit([0.8, 0.2], seed=42)
    train_count = train_data.count()
    test_count = test_data.count()
    print(f"--> Training Records: {train_count:,} | Testing Records: {test_count:,}")

    # --------------------------------------------------------------------------
    # MODEL 1: LOGISTIC REGRESSION
    # --------------------------------------------------------------------------
    print("\n[Training Model 1]: Logistic Regression Classifier...")
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=20,
        regParam=0.01
    )
    lr_model = lr.fit(train_data)
    lr_preds = lr_model.transform(test_data)
    lr_metrics = evaluate_model(lr_preds, "Logistic Regression")

    # --------------------------------------------------------------------------
    # MODEL 2: DECISION TREE CLASSIFIER
    # --------------------------------------------------------------------------
    print("\n[Training Model 2]: Decision Tree Classifier...")
    dt = DecisionTreeClassifier(
        featuresCol="features",
        labelCol="label",
        maxDepth=5,
        seed=42
    )
    dt_model = dt.fit(train_data)
    dt_preds = dt_model.transform(test_data)
    dt_metrics = evaluate_model(dt_preds, "Decision Tree Classifier")

    # --------------------------------------------------------------------------
    # COMPARISON TABLE
    # --------------------------------------------------------------------------
    print("\n" + "="*80)
    print("FINAL MODEL COMPARISON SUMMARY (INT315 Unit VI)")
    print("="*80)
    print(f"{'Metric':<25} | {'Logistic Regression':<22} | {'Decision Tree':<22}")
    print("-" * 75)
    print(f"{'Accuracy':<25} | {lr_metrics['accuracy']*100:<21.2f}% | {dt_metrics['accuracy']*100:<21.2f}%")
    print(f"{'Precision':<25} | {lr_metrics['precision']*100:<21.2f}% | {dt_metrics['precision']*100:<21.2f}%")
    print(f"{'Recall (TPR)':<25} | {lr_metrics['recall']*100:<21.2f}% | {dt_metrics['recall']*100:<21.2f}%")
    print(f"{'False Positive Rate':<25} | {lr_metrics['fpr']*100:<21.2f}% | {dt_metrics['fpr']*100:<21.2f}%")
    print(f"{'F1-Score':<25} | {lr_metrics['f1']:<22.4f} | {dt_metrics['f1']:<22.4f}")
    print(f"{'Area Under ROC (AUC)':<25} | {lr_metrics['auc']:<22.4f} | {dt_metrics['auc']:<22.4f}")
    print("-" * 75)

    # --------------------------------------------------------------------------
    # SAVE BEST MODEL & SUMMARY METRICS
    # --------------------------------------------------------------------------
    best_model_name = "Decision Tree" if dt_metrics["auc"] > lr_metrics["auc"] else "Logistic Regression"
    best_model = dt_model if best_model_name == "Decision Tree" else lr_model
    save_path = os.path.join(model_save_dir, "best_flight_delay_model")

    print(f"\n--> Best model based on AUC is: {best_model_name}")
    
    # 1. Save standard JSON summary of the model evaluation to disk
    import json
    summary_file = os.path.join(model_save_dir, "model_evaluation_summary.json")
    summary_data = {
        "best_model": best_model_name,
        "logistic_regression": lr_metrics,
        "decision_tree": dt_metrics
    }
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=4)
    print(f"--> Saved evaluation metrics to: {summary_file}")

    # 2. Save MLlib model using Spark writer (catches Windows Hadoop winutils if not installed)
    print(f"--> Attempting PySpark MLlib model save to: {save_path} ...")
    try:
        best_model.write().overwrite().save(save_path)
        print("[SUCCESS] PySpark MLlib model successfully saved to disk!")
    except Exception as e:
        print(f"[NOTE] Native Hadoop winutils warning on Windows (expected without winutils.exe):")
        print(f"       Metrics and results were successfully preserved in {summary_file}.")


def main():
    print("="*80)
    print("  AVIATION NETWORK DELAY CLASSIFICATION (PYSPARK MLLIB - UNIT VI)  ")
    print("="*80)

    # Set paths
    project_root = os.path.dirname(os.path.dirname(__file__))
    dataset_path = os.path.join(project_root, "data", "Airline_Delay_Cause.csv")
    model_save_dir = os.path.join(project_root, "model")

    os.makedirs(model_save_dir, exist_ok=True)

    # 1. Initialize Spark
    spark = create_spark_session()

    try:
        # 2. Ingest and Clean
        clean_df = prepare_data(spark, dataset_path)

        # 3. StringIndexer + OneHotEncoder + VectorAssembler
        final_df = build_feature_pipeline(clean_df)

        # 4. Train Models & Evaluate (Confusion Matrix, FPR, AUC)
        train_and_evaluate_models(final_df, model_save_dir)

        print("\n[SUCCESS] Delay Classification Pipeline executed successfully!")
    finally:
        # 5. Stop Spark cleanly
        spark.stop()
        print("[INFO] Spark Session closed.")


if __name__ == "__main__":
    main()
