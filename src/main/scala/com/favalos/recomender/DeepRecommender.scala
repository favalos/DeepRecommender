package com.favalos.recomender

import com.favalos.autoencoder.{AutoEncoderModule, SymbolBuilder}
import com.favalos.data.DataLoader
import org.apache.mxnet.optimizer.{SGD}
import org.apache.mxnet.{Context, MSE, Shape, Symbol}

object DeepRecommender {

  def loss(output: Symbol, label: Symbol, batchSize: Int): Symbol = {

    val c = Symbol.zeros(Shape(batchSize,1))
    val mask = Symbol.broadcast_greater()()(Map("lhs" -> label, "rhs" -> c))
    val sum = Symbol.sum_axis()()(Map("data" -> mask, "axis" -> 1,"keepdims" -> 1))
    val mse = Symbol.square("loss")()(Map("data" -> (output - label)))
    Symbol.broadcast_div()()(Map("lhs" -> (mse * mask), "rhs" -> sum))
  }

  def trainRecommender() = {

    val epoch = 50;
    val ctx = Context.cpu()
    val dataIter = DataLoader.loadIterCSV("Data_Tensor.train", "(9066)")

    val encoderDims = Some(Array(9066, 128, 256, 256))
    val decoderDims = Some(Array(256, 128, 9066))
    val dropOut = Some(0.65F)

    val dataSymbol = Symbol.Variable("data")

    val decoderSymbol = SymbolBuilder.buildDeepEncoder(dataSymbol, encoderDims, decoderDims, dropOut)

    val module = new AutoEncoderModule(decoderSymbol,
                          ctx,
                          dataIter.provideData,
                          Map("label" -> dataIter.provideData("data")),
                          new SGD(learningRate = 0.00001F, momentum = 0.9F),
                          loss)

    val mse = new MSE()
    for(i <- 0 until epoch) {

      mse.reset()
      dataIter.reset()

      while (dataIter.hasNext) {

        val dataBatch = dataIter.next()
        module.forward(dataBatch.data, dataBatch.data)
        module.backward()
        module.updateGradients()

        mse.update(dataBatch.data, module.getOutputs())

      }

      if(i > 0 && i % 10 == 0) {
        module.saveCheckpoint("DeepRecommender", i)
      }

      println(s"Iteration: ${i} MSE metric is: ${mse.get._2(0)}")

    }

    module.saveCheckpoint("DeepRecommender", epoch)
  }


  def main(args: Array[String]): Unit = {

    DataLoader.convertMovieLens("ratings.csv", trainPercent = 90, validPercent = 5, testPercent = 5)

    trainRecommender()

  }

}
