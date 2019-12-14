package preprocess

import org.apache.spark.sql.DataFrame

/**
  * Object with methods to preprocess data
  */
object PreProcessDataset {
  def start(dataset: DataFrame): DataFrame = {
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

    // TODO: More preprocessing

    return preProcessDataset
  }
}
