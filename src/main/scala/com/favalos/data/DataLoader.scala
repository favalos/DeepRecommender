package com.favalos.data

import java.io.File

import org.apache.mxnet.{DataIter, IO}

import scala.util.Random

object DataLoader {

  def saveToFile(data: Iterable[(String,List[Array[String]])], movieIdMap: Map[String, Int], filename: String): Unit = {

    val p = new java.io.PrintWriter(new File(filename))

    data.foreach {
      case(_, arr) => {
        val row = Array.fill[Float](movieIdMap.size)(0)
        arr.foreach(entry => row(movieIdMap(entry(1))) = entry(2).toFloat)
        p.append(row.mkString(",") + System.lineSeparator())
      }
    }

    p.flush()
    p.close()
  }

  def convertMovieLens(fileName: String, trainPercent: Int = 70, validPercent: Int = 15, testPercent: Int = 15 ): Unit = {

    val outputName = "Data_Tensor"

    assert(trainPercent + validPercent + testPercent == 100, "Sum of percentages must be equal to 100.")

    val dataRaw = scala.io.Source.fromInputStream(DataLoader.getClass.getResourceAsStream(fileName))
    val splittedDataList = dataRaw.getLines().drop(1).map(_.split(",")).toList

    val movieIdMap = splittedDataList.map(_(1)).toSet.zipWithIndex.toMap[String, Int]

    val splittedData = splittedDataList.groupBy(_(0))

    val (trainData, data) = Random.shuffle(splittedData).splitAt( splittedData.size * trainPercent / 100)
    val (validData, testData) = data.splitAt(data.size * validPercent /(validPercent + testPercent))

    saveToFile(trainData, movieIdMap, outputName + ".train" )
    saveToFile(validData, movieIdMap, outputName + ".valid" )
    saveToFile(testData, movieIdMap, outputName + ".test")

  }

  def loadIterCSV(filename: String, shape: String, batchSize: String = "128"): DataIter = {

    val dataIter = IO.CSVIter(Map("data_csv" -> filename,
      "data_shape" -> shape,
      "batch_size" -> batchSize))

    dataIter
  }


}
