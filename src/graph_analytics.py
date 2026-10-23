"""
================================================================================
Aviation Network Delay Analytics - Graph Network Analysis & Congestion Hubs
Mapped Unit: Unit III (Graph Analytics, Network Degrees, Bottleneck Hubs)
Course: INT315 - Cluster Computing
================================================================================
Description:
    This script models the aviation network as a directed graph:
    1. Vertices: Airports (e.g., ATL, ORD, DFW, JFK, LAX)
    2. Directed Edges: Flight routes from Origin -> Destination
    3. Graph Metrics:
       - In-Degree: Number of incoming routes/flights landing at an airport
       - Out-Degree: Number of outgoing routes/flights departing from an airport
       - Hub Congestion Score: Identifies high-risk airport bottlenecks
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
from pyspark.sql.functions import col, count, sum, avg, round, desc, expr


def create_spark_session():
    """
    Creates and returns a local SparkSession.
    """
    print("\n" + "="*80)
    print("STEP 1: INITIALIZING SPARK SESSION FOR GRAPH NETWORK ANALYTICS")
    print("="*80)
    spark = SparkSession.builder \
        .appName("Aviation_Graph_Analytics") \
        .master("local[*]") \
        .config("spark.ui.showConsoleProgress", "false") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    print(f"--> Spark Session online. Version: {spark.version}")
    return spark


def build_flight_network(spark, csv_path):
    """
    STEP 2: MODEL AIRPORTS AS VERTICES AND ROUTES AS DIRECTED EDGES
    Extracts origin-destination pairs and flight metrics.
    """
    print("\n" + "="*80)
    print("STEP 2: CONSTRUCTING DIRECTED AVIATION GRAPH (Vertices & Edges)")
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

    # Map Carrier Hubs to generate 'Origin' airport for each route
    hub_mapping_sql = """
        CASE 
            WHEN carrier IN ('AA', 'MQ') THEN 'DFW'
            WHEN carrier IN ('DL', '9E') THEN 'ATL'
            WHEN carrier IN ('UA') THEN 'ORD'
            WHEN carrier IN ('WN') THEN 'MDW'
            WHEN carrier IN ('B6') THEN 'JFK'
            WHEN carrier IN ('AS') THEN 'SEA'
            WHEN carrier IN ('F9') THEN 'DEN'
            WHEN carrier IN ('NK') THEN 'FLL'
            ELSE 'ORD'
        END
    """
    network_df = clean_df.withColumn("Origin", expr(hub_mapping_sql)) \
                         .withColumnRenamed("airport", "Dest") \
                         .withColumnRenamed("arr_flights", "ArrFlights") \
                         .withColumnRenamed("arr_delay", "ArrDelay") \
                         .withColumnRenamed("arr_del15", "ArrDel15")

    # Group into unique Directed Edges: (Origin -> Dest)
    routes_edges = network_df.groupBy("Origin", "Dest") \
                             .agg(
                                 sum("ArrFlights").alias("TotalRouteFlights"),
                                 sum("ArrDel15").alias("DelayedRouteFlights"),
                                 round(avg(col("ArrDelay") / col("ArrFlights")), 2).alias("AvgRouteDelay")
                             )

    total_routes = routes_edges.count()
    print(f"--> Total Directed Flight Routes (Edges): {total_routes:,}")
    print("\n[Sample Directed Edges (Origin -> Destination)]:")
    routes_edges.show(5, truncate=False)

    return routes_edges, network_df


def compute_degrees_and_congestion(spark, routes_edges, network_df):
    """
    STEP 3: COMPUTE IN-DEGREE, OUT-DEGREE, AND CONGESTION HUBS
    - In-Degree: How many routes feed into an airport (Arrival Hub Pressure)
    - Out-Degree: How many routes depart from an airport (Departure Hub Pressure)
    - Pinpoints high-risk congestion hubs!
    """
    print("\n" + "="*80)
    print("STEP 3: CALCULATING NODE IN-DEGREES AND OUT-DEGREES")
    print("="*80)

    # 1. Compute In-Degree (Destination Airport connectivity)
    # How many distinct origin airports fly into this destination
    in_degrees = routes_edges.groupBy("Dest") \
                             .agg(
                                 count("Origin").alias("InDegree_Routes"),
                                 sum("TotalRouteFlights").alias("TotalIncomingFlights")
                             ) \
                             .withColumnRenamed("Dest", "Airport")

    # 2. Compute Out-Degree (Origin Airport connectivity)
    # How many distinct destination airports are served from this origin hub
    out_degrees = routes_edges.groupBy("Origin") \
                              .agg(
                                  count("Dest").alias("OutDegree_Routes"),
                                  sum("TotalRouteFlights").alias("TotalOutgoingFlights")
                              ) \
                              .withColumnRenamed("Origin", "Airport")

    # 3. Join In-Degree and Out-Degree
    full_degrees = in_degrees.join(out_degrees, on="Airport", how="full_outer") \
                             .fillna(0)

    # Calculate Total Degree = In-Degree + Out-Degree
    full_degrees = full_degrees.withColumn(
        "TotalDegree",
        col("InDegree_Routes") + col("OutDegree_Routes")
    )

    print("\n[AIRPORT NETWORK DEGREE DISTRIBUTION (Top Hubs by Connectivity)]:")
    full_degrees.orderBy(desc("TotalDegree")).show(10, truncate=False)

    # 4. Pinpoint High-Risk Congestion Hubs
    # Combine airport delay metrics with in-degree to score congestion risk
    print("\n" + "="*80)
    print("STEP 4: PINPOINTING HIGH-RISK CONGESTION HUBS")
    print("="*80)

    airport_delays = network_df.groupBy("Dest") \
                               .agg(
                                   round(avg(col("ArrDelay") / col("ArrFlights")), 2).alias("AvgArrivalDelayMinutes"),
                                   sum("ArrFlights").alias("TotalFlightsHandled"),
                                   sum("ArrDel15").alias("TotalDelayedFlights")
                               ) \
                               .withColumnRenamed("Dest", "Airport")

    congestion_hubs = full_degrees.join(airport_delays, on="Airport", how="inner")

    # High-Risk Congestion Score = (InDegree * AvgArrivalDelayMinutes) / 10
    congestion_hubs = congestion_hubs.withColumn(
        "CongestionRiskScore",
        round((col("InDegree_Routes") * col("AvgArrivalDelayMinutes")) / 10.0, 2)
    )

    print("\n[TOP 10 HIGH-RISK CONGESTION HUBS]:")
    congestion_hubs.orderBy(desc("CongestionRiskScore")) \
                   .select("Airport", "InDegree_Routes", "OutDegree_Routes", "TotalFlightsHandled", "AvgArrivalDelayMinutes", "CongestionRiskScore") \
                   .show(10, truncate=False)


def main():
    print("="*80)
    print("  AVIATION NETWORK ANALYSIS - GRAPH ANALYTICS (INT315 UNIT III)  ")
    print("="*80)

    project_root = os.path.dirname(os.path.dirname(__file__))
    dataset_path = os.path.join(project_root, "data", "Airline_Delay_Cause.csv")

    spark = create_spark_session()

    try:
        routes_edges, network_df = build_flight_network(spark, dataset_path)
        compute_degrees_and_congestion(spark, routes_edges, network_df)
        print("\n[SUCCESS] Aviation Graph Analytics completed successfully!")
    finally:
        spark.stop()
        print("[INFO] Spark Session closed.")


if __name__ == "__main__":
    main()
