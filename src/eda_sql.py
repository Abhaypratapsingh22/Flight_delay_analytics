"""
================================================================================
Aviation Network Delay Analytics - Spark SQL & Data Preprocessing
Mapped Unit: Unit IV (Spark SQL, GroupBy, Aggregate Functions, SQL Wildcards)
Course: INT315 - Cluster Computing
================================================================================
Description:
    This script loads the airline delay dataset into Apache Spark,
    performs data cleaning, and runs Spark SQL queries to discover:
    1. Systemic bottlenecks by carrier (groupBy Carrier + agg avg ArrDelay)
    2. Top congested arrival airports (hubs)
    3. Primary root causes of delays (Weather, Carrier, Security, NAS, Late Aircraft)
    4. Spark SQL queries using wildcards (LIKE)
================================================================================
"""

import os
import sys

# ------------------------------------------------------------------------------
# STEP 1: INITIALIZE SPARK ENVIRONMENT
# ------------------------------------------------------------------------------
# We use 'findspark' so that Python can automatically find where Apache Spark
# is installed on your computer without complicated environment path setups.
try:
    import findspark
    findspark.init()
except Exception as e:
    print(f"[INFO] findspark initialization: {e}")

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, sum, round, when, expr

def create_spark_session():
    """
    Creates and returns a local SparkSession.
    'local[*]' means Spark will use all available CPU cores on your laptop.
    """
    print("\n[Step 1] Starting Apache Spark Session...")
    spark = SparkSession.builder \
        .appName("AviationDelay_EDA_SQL") \
        .master("local[*]") \
        .config("spark.ui.showConsoleProgress", "false") \
        .getOrCreate()
    
    # Set log level to WARN to reduce unnecessary console messages
    spark.sparkContext.setLogLevel("ERROR")
    print(f"--> Spark Session created successfully! (Spark Version: {spark.version})")
    return spark


def load_and_clean_data(spark, csv_path):
    """
    STEP 2: LOAD DATASET AND CLEAN
    Reads 'Airline_Delay_Cause.csv', cleans null values,
    and standardizes column names.
    """
    print(f"\n[Step 2] Loading dataset from: {csv_path}")
    
    # Check if the file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found at: {csv_path}")

    # Read CSV with Spark DataFrame API
    # header=True means the first row contains column headers
    # inferSchema=True automatically detects integers, doubles, and strings
    df = spark.read.option("header", "true") \
                   .option("inferSchema", "true") \
                   .csv(csv_path)

    total_raw_rows = df.count()
    print(f"--> Total raw rows in dataset: {total_raw_rows:,}")

    # Clean the dataset:
    # 1. Filter out records where arr_flights is null or <= 0
    # 2. Filter out records where arr_delay is null
    clean_df = df.filter(
        col("arr_flights").isNotNull() & 
        (col("arr_flights") > 0) & 
        col("arr_delay").isNotNull()
    )

    # Standardize column names as requested:
    # 'carrier' -> 'Carrier'
    # 'arr_delay' -> 'ArrDelay'
    # 'airport' -> 'Airport'
    # 'arr_flights' -> 'ArrFlights'
    # 'arr_del15' -> 'ArrDel15'
    clean_df = clean_df.withColumnRenamed("carrier", "Carrier") \
                       .withColumnRenamed("arr_delay", "ArrDelay") \
                       .withColumnRenamed("airport", "Airport") \
                       .withColumnRenamed("arr_flights", "ArrFlights") \
                       .withColumnRenamed("arr_del15", "ArrDel15")

    # Add calculated columns for beginner clarity:
    # avg_delay_per_flight = ArrDelay / ArrFlights (in minutes)
    clean_df = clean_df.withColumn("AvgDelayPerFlight", round(col("ArrDelay") / col("ArrFlights"), 2))
    
    # delay_rate = percentage of flights delayed >= 15 minutes
    clean_df = clean_df.withColumn("DelayRatePercent", round((col("ArrDel15") / col("ArrFlights")) * 100, 2))

    print(f"--> Cleaned rows ready for analysis: {clean_df.count():,}")
    return clean_df


def run_sql_analysis(spark, df):
    """
    STEP 3: RUN SPARK SQL QUERIES
    Registers the DataFrame as a temporary SQL table 'flights'
    and runs SQL queries to isolate systemic bottlenecks.
    """
    print("\n[Step 3] Registering DataFrame as SQL Temporary View: 'flights'...")
    df.createOrReplaceTempView("flights")

    print("\n" + "="*80)
    print("ANALYSIS 1: GROUP BY CARRIER & AVERAGE ARRIVAL DELAY")
    print("Objective: Isolate systemic bottlenecks across airline carriers")
    print("="*80)
    
    # 1. Using DataFrame API: groupBy("Carrier").agg(avg("ArrDelay"))
    print("\n[Method A - PySpark DataFrame API]: groupBy('Carrier').agg(avg('ArrDelay'))")
    carrier_df_api = df.groupBy("Carrier") \
                       .agg(
                           round(avg("ArrDelay"), 2).alias("AvgTotalDelayMinutes"),
                           round(avg("AvgDelayPerFlight"), 2).alias("AvgDelayPerFlightMinutes"),
                           sum("ArrFlights").alias("TotalFlights"),
                           sum("ArrDel15").alias("TotalDelayedFlights")
                       ) \
                       .orderBy(col("AvgDelayPerFlightMinutes").desc())
    carrier_df_api.show(10, truncate=False)

    # 2. Using Pure Spark SQL query
    print("\n[Method B - Pure Spark SQL Query]:")
    query1 = """
        SELECT 
            Carrier,
            carrier_name AS CarrierName,
            COUNT(*) AS NumberOfRecords,
            ROUND(AVG(ArrDelay), 2) AS AvgArrDelay,
            ROUND(AVG(AvgDelayPerFlight), 2) AS AvgMinutesPerFlight,
            ROUND(AVG(DelayRatePercent), 2) AS AvgDelayRatePercent
        FROM flights
        GROUP BY Carrier, carrier_name
        ORDER BY AvgMinutesPerFlight DESC
        LIMIT 10
    """
    spark.sql(query1).show(10, truncate=False)

    print("\n" + "="*80)
    print("ANALYSIS 2: TOP CONGESTED AIRPORTS (AIRPORT BOTTLENECKS)")
    print("Objective: Identify high-risk airports where flights suffer worst delays")
    print("="*80)
    query2 = """
        SELECT 
            Airport,
            airport_name AS AirportName,
            SUM(ArrFlights) AS TotalArrivingFlights,
            SUM(ArrDel15) AS TotalDelayedFlights,
            ROUND(AVG(AvgDelayPerFlight), 2) AS AvgDelayPerFlightMinutes,
            ROUND((SUM(ArrDel15) / SUM(ArrFlights)) * 100, 2) AS OverallDelayPercentage
        FROM flights
        GROUP BY Airport, airport_name
        HAVING SUM(ArrFlights) >= 5000
        ORDER BY AvgDelayPerFlightMinutes DESC
        LIMIT 10
    """
    spark.sql(query2).show(10, truncate=False)

    print("\n" + "="*80)
    print("ANALYSIS 3: PRIMARY DELAY CAUSES (ROOT CAUSE BREAKDOWN)")
    print("Objective: Determine what causes the delays (Weather vs Carrier vs Security etc.)")
    print("="*80)
    query3 = """
        SELECT 
            ROUND(SUM(carrier_delay) / SUM(ArrDelay) * 100, 2) AS CarrierDelaySharePct,
            ROUND(SUM(weather_delay) / SUM(ArrDelay) * 100, 2) AS WeatherDelaySharePct,
            ROUND(SUM(nas_delay) / SUM(ArrDelay) * 100, 2) AS NasAirTrafficSharePct,
            ROUND(SUM(security_delay) / SUM(ArrDelay) * 100, 2) AS SecurityDelaySharePct,
            ROUND(SUM(late_aircraft_delay) / SUM(ArrDelay) * 100, 2) AS LateAircraftSharePct
        FROM flights
    """
    spark.sql(query3).show(truncate=False)

    print("\n" + "="*80)
    print("ANALYSIS 4: SPARK SQL WILDCARDS DEMONSTRATION (Unit IV Syllabus)")
    print("Objective: Find major 'International' airports experiencing heavy delays")
    print("="*80)
    query4 = """
        SELECT 
            Airport,
            airport_name AS AirportName,
            ROUND(AVG(AvgDelayPerFlight), 2) AS AvgDelayMinutes
        FROM flights
        WHERE airport_name LIKE '%International%'
        GROUP BY Airport, airport_name
        HAVING AVG(AvgDelayPerFlight) > 15.0
        ORDER BY AvgDelayMinutes DESC
        LIMIT 10
    """
    spark.sql(query4).show(10, truncate=False)


def main():
    print("="*80)
    print("  AVIATION NETWORK DELAY ANALYTICS - SPARK SQL (INT315 UNIT IV)  ")
    print("="*80)

    # Path to dataset
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "Airline_Delay_Cause.csv")

    # 1. Initialize Spark
    spark = create_spark_session()

    try:
        # 2. Load and Clean
        df = load_and_clean_data(spark, dataset_path)

        # 3. Run Analysis
        run_sql_analysis(spark, df)

        print("\n[SUCCESS] Spark SQL Preprocessing & EDA completed successfully!")
    finally:
        # 4. Always stop SparkSession cleanly
        spark.stop()
        print("[INFO] Spark Session closed.")


if __name__ == "__main__":
    main()
