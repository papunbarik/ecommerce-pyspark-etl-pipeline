"""
Olist E-Commerce Multi-Channel ETL & Window Revenue Pipeline
Engine: Apache Spark (PySpark 3.4.1)
Storage Target: Google Cloud BigQuery / Partitioned Parquet
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, sum as _sum, avg, count, when, lag,
    to_date, round as _round, coalesce, lit
)
from pyspark.sql.window import Window
from pyspark.sql.types import DecimalType

def init_spark() -> SparkSession:
    return SparkSession.builder \
        .appName("OlistRevenueAttributionETL") \
        .config("spark.sql.shuffle.partitions", "4") \
        .getOrCreate()

def run_pipeline():
    spark = init_spark()
    
    # 1. Ingestion: Raw Order Headers & Payment Records
    orders_df = spark.read.csv("data/olist_orders_dataset.csv", header=True, inferSchema=True)
    payments_df = spark.read.csv("data/olist_order_payments_dataset.csv", header=True, inferSchema=True)
    
    # 2. Cleansing & Deduplication
    clean_orders = orders_df \
        .filter(col("order_status") == "delivered") \
        .dropDuplicates(["order_id"]) \
        .withColumn("order_purchase_date", to_date(col("order_purchase_timestamp"))) \
        .withColumn("customer_id", coalesce(col("customer_id"), lit("GUEST_CHECKOUT")))
        
    clean_payments = payments_df \
        .filter(col("payment_value") > 0) \
        .dropDuplicates(["order_id", "payment_sequential"])
        
    # 3. Join & Dimensional Attribution
    joined_df = clean_orders.join(clean_payments, on="order_id", how="inner") \
        .select(
            col("order_id"),
            col("customer_id"),
            col("order_purchase_date"),
            col("payment_type").alias("channel"),
            col("payment_value").cast(DecimalType(10, 2)).alias("revenue_brl")
        )
        
    # 4. Window Aggregation: Daily Channel Totals & 7-Day Rolling Revenue
    daily_channel_agg = joined_df.groupBy("channel", "order_purchase_date") \
        .agg(
            _round(_sum("revenue_brl"), 2).alias("daily_revenue"),
            count("order_id").alias("transaction_count")
        )
        
    window_spec = Window.partitionBy("channel").orderBy("order_purchase_date").rowsBetween(-6, 0)
    
    final_df = daily_channel_agg.withColumn(
        "rolling_7d_revenue",
        _round(_sum("daily_revenue").over(window_spec), 2)
    )
    
    print("ETL Transformation complete. Row sample:")
    final_df.show(5, truncate=False)
    
    # 5. Partitioned Parquet Sink (Emulating BigQuery / GCS export)
    final_df.write \
        .mode("overwrite") \
        .partitionBy("channel") \
        .parquet("output/olist_daily_channel_revenue")
        
    spark.stop()

if __name__ == "__main__":
    run_pipeline()
