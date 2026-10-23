/*
 * ============================================================================
 * Aviation Network Delay Analytics - Spark GraphX in Scala
 * Mapped Unit: Unit III (Graph Analytics, GraphX Operations, In/Out Degrees)
 * Course: INT315 - Cluster Computing
 * ============================================================================
 * Description:
 *   This Scala script demonstrates how to model an aviation network in Spark GraphX:
 *   1. Vertices: Airports (Airport ID, Airport Code, Airport City)
 *   2. Directed Edges: Flight Routes between airports (srcId -> dstId, Flight Volume)
 *   3. GraphX Operations:
 *      - graph.inDegrees  : Incoming flights (Arrival congestion)
 *      - graph.outDegrees : Outgoing flights (Hub departure volume)
 *      - Pinpoints high-risk congestion hubs
 *
 * How to Run:
 *   Launch spark-shell:
 *     spark-shell
 *   Then load this file:
 *     :load src/AirportGraphX.scala
 * ============================================================================
 */

import org.apache.spark._
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD

object AirportGraphXApp {

  def main(args: Array[String]): Unit = {

    // Step 1: Initialize Spark Context (provided automatically in spark-shell as 'sc')
    val conf = new SparkConf().setAppName("AirportGraphXAnalytics").setMaster("local[*]")
    val sc = SparkContext.getOrCreate(conf)
    sc.setLogLevel("ERROR")

    println("\n" + "=" * 75)
    println("  AVIATION NETWORK GRAPH ANALYTICS USING SPARK GRAPHX (SCALA)  ")
    println("=" * 75)

    // ------------------------------------------------------------------------
    // STEP 2: DEFINE AIRPORT VERTICES (VertexId, (AirportCode, CityName))
    // ------------------------------------------------------------------------
    println("\n[Step 1] Creating Airport Vertices...")
    val airportVertices: RDD[(VertexId, (String, String))] = sc.parallelize(Array(
      (1L, ("ATL", "Atlanta - Hartsfield Jackson")),
      (2L, ("ORD", "Chicago - O'Hare")),
      (3L, ("DFW", "Dallas/Fort Worth")),
      (4L, ("DEN", "Denver International")),
      (5L, ("LAX", "Los Angeles International")),
      (6L, ("JFK", "New York - JFK")),
      (7L, ("SFO", "San Francisco International")),
      (8L, ("SEA", "Seattle-Tacoma")),
      (9L, ("MCO", "Orlando International")),
      (10L, ("EWR", "Newark Liberty International"))
    ))

    // ------------------------------------------------------------------------
    // STEP 3: DEFINE FLIGHT ROUTE EDGES (srcAirportId -> dstAirportId, FlightCount)
    // ------------------------------------------------------------------------
    println("[Step 2] Creating Flight Route Directed Edges...")
    val flightEdges: RDD[Edge[Int]] = sc.parallelize(Array(
      Edge(1L, 2L, 540),   // ATL -> ORD
      Edge(1L, 3L, 480),   // ATL -> DFW
      Edge(1L, 5L, 390),   // ATL -> LAX
      Edge(1L, 6L, 410),   // ATL -> JFK
      Edge(1L, 9L, 620),   // ATL -> MCO
      Edge(2L, 1L, 530),   // ORD -> ATL
      Edge(2L, 3L, 460),   // ORD -> DFW
      Edge(2L, 4L, 510),   // ORD -> DEN
      Edge(2L, 7L, 380),   // ORD -> SFO
      Edge(2L, 10L, 450),  // ORD -> EWR
      Edge(3L, 1L, 470),   // DFW -> ATL
      Edge(3L, 2L, 450),   // DFW -> ORD
      Edge(3L, 4L, 420),   // DFW -> DEN
      Edge(3L, 5L, 490),   // DFW -> LAX
      Edge(4L, 2L, 500),   // DEN -> ORD
      Edge(4L, 5L, 430),   // DEN -> LAX
      Edge(4L, 7L, 390),   // DEN -> SFO
      Edge(4L, 8L, 350),   // DEN -> SEA
      Edge(5L, 1L, 380),   // LAX -> ATL
      Edge(5L, 6L, 420),   // LAX -> JFK
      Edge(5L, 7L, 580),   // LAX -> SFO
      Edge(5L, 8L, 410),   // LAX -> SEA
      Edge(6L, 1L, 400),   // JFK -> ATL
      Edge(6L, 5L, 410),   // JFK -> LAX
      Edge(6L, 7L, 320),   // JFK -> SFO
      Edge(7L, 2L, 370),   // SFO -> ORD
      Edge(7L, 5L, 590),   // SFO -> LAX
      Edge(8L, 4L, 340),   // SEA -> DEN
      Edge(8L, 5L, 400),   // SEA -> LAX
      Edge(8L, 7L, 360),   // SEA -> SFO
      Edge(9L, 1L, 610),   // MCO -> ATL
      Edge(9L, 6L, 350),   // MCO -> JFK
      Edge(10L, 2L, 440)   // EWR -> ORD
    ))

    // ------------------------------------------------------------------------
    // STEP 4: CONSTRUCT GRAPHX GRAPH
    // ------------------------------------------------------------------------
    val defaultAirport = ("UNKNOWN", "Unknown Airport")
    val airportGraph = Graph(airportVertices, flightEdges, defaultAirport)

    println(s"--> Total Airport Vertices : ${airportGraph.vertices.count()}")
    println(s"--> Total Flight Routes    : ${airportGraph.edges.count()}")

    // ------------------------------------------------------------------------
    // STEP 5: CALCULATE IN-DEGREES (Arrival Congestion)
    // ------------------------------------------------------------------------
    println("\n" + "-" * 75)
    println("AIRPORT IN-DEGREES (Incoming Routes -> Arrival Congestion)")
    println("-" * 75)

    // Join inDegrees with airport vertices to get airport names
    val inDegreeWithNames = airportGraph.inDegrees
      .join(airportGraph.vertices)
      .map { case (id, (inDeg, (code, city))) => (code, city, inDeg) }
      .sortBy(_._3, ascending = false)

    println(f"${"Airport Code"}%-15s | ${"City / Airport Name"}%-35s | ${"In-Degree"}%10s")
    println("-" * 75)
    inDegreeWithNames.collect().foreach { case (code, city, inDeg) =>
      println(f"$code%-15s | $city%-35s | $inDeg%10d")
    }

    // ------------------------------------------------------------------------
    // STEP 6: CALCULATE OUT-DEGREES (Departure Hub Pressure)
    // ------------------------------------------------------------------------
    println("\n" + "-" * 75)
    println("AIRPORT OUT-DEGREES (Outgoing Routes -> Departure Hub Capacity)")
    println("-" * 75)

    val outDegreeWithNames = airportGraph.outDegrees
      .join(airportGraph.vertices)
      .map { case (id, (outDeg, (code, city))) => (code, city, outDeg) }
      .sortBy(_._3, ascending = false)

    println(f"${"Airport Code"}%-15s | ${"City / Airport Name"}%-35s | ${"Out-Degree"}%10s")
    println("-" * 75)
    outDegreeWithNames.collect().foreach { case (code, city, outDeg) =>
      println(f"$code%-15s | $city%-35s | $outDeg%10d")
    }

    // ------------------------------------------------------------------------
    // STEP 7: TOTAL DEGREE & PINPOINT HIGH-RISK CONGESTION HUBS
    // ------------------------------------------------------------------------
    println("\n" + "-" * 75)
    println("TOP CONGESTION HUBS (Total Network Connectivity = In + Out Degrees)")
    println("-" * 75)

    val totalDegrees = airportGraph.degrees
      .join(airportGraph.vertices)
      .map { case (id, (deg, (code, city))) => (code, city, deg) }
      .sortBy(_._3, ascending = false)

    println(f"${"Airport Code"}%-15s | ${"City / Airport Name"}%-35s | ${"Total Degree"}%12s")
    println("-" * 75)
    totalDegrees.collect().foreach { case (code, city, deg) =>
      println(f"$code%-15s | $city%-35s | $deg%12d")
    }

    println("\n[CONCLUSION] Airports with highest total degrees (ATL, ORD, LAX, DFW, SFO)")
    println("             are systemic bottleneck hubs prone to cascading flight delays.")
    println("=" * 75)
  }
}

// When loaded in spark-shell, invoke main directly:
AirportGraphXApp.main(Array())
