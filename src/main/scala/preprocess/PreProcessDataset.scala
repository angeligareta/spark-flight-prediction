package preprocess

import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * Object with methods to preprocess data
  */
object PreProcessDataset {
  def start(spark: SparkSession, dataset: DataFrame): DataFrame = {
    // Drop columns that the exercise required.
    val columnsToDrop = Array(
      "ArrTime",
      "ActualElapsedTime",
      "AirTime",
      "TaxiIn",
      "Diverted",
      "CarrierDelay",
      "WeatherDelay",
      "NASDelay",
      "SecurityDelay",
      "LateAircraftDelay"
    );
    var preProcessDataset = dataset.drop(columnsToDrop: _*);

    // Import implicits to use $
    import spark.implicits._

    // Cast columns that are string to correct format
    preProcessDataset = preProcessDataset
      .withColumn("DepTime", $"DepTime" cast "Int")
      .withColumn("CRSElapsedTime", $"CRSElapsedTime" cast "Int")
      .withColumn("ArrDelay", $"ArrDelay" cast "Int")
      .withColumn("DepDelay", $"DepDelay" cast "Int")
      .withColumn("DepDelay", $"DepDelay" cast "Int")

    // TODO: More preProcessing
    return preProcessDataset
  }
}
