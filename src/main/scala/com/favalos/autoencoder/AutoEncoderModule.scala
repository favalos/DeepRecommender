package com.favalos.autoencoder

import org.apache.mxnet.Model
import org.apache.mxnet.{Context, NDArray, Optimizer, Shape, Symbol, Xavier}

class AutoEncoderModule(val symbol: Symbol, ctx: Context,
                        val dataShape: Map[String, Shape],
                        val labelShape: Map[String, Shape],
                        val optimizer: Optimizer,
                        val lossFunction: (Symbol, Symbol, Int) => Symbol,
                        trainable: Boolean = true) {


  val initializer = new Xavier()

  val except = Set("data", "label")
  val label = Symbol.Variable("label")

  val loss = Symbol.MakeLoss("loss")()(Map("data" -> lossFunction(symbol, label, dataShape("data")(0))))

  val (decInputShape, _, _) = loss.inferShape(dataShape ++ labelShape)

  val (argsDict, gradDict, paramGrads) = {

    val argsDict = loss.listArguments().zip(decInputShape.map(s => NDArray.empty(s, ctx))).toMap

    val gradDict = symbol.listArguments()
      .zip(decInputShape).filterNot(v => except.contains(v._1)).map {
      case (key, shape) => key -> NDArray.empty(shape, ctx)
    }.toMap

    argsDict.foreach {
      case (key, ndArray) =>
        if (!except.contains(key))
          initializer.initWeight(key, ndArray)
    }

    val paramGrads = gradDict.toList.zipWithIndex.map {
      case ((key, grad), idx) => (idx, key, grad, optimizer.createState(idx, argsDict(key)))
    }

    (argsDict, gradDict, paramGrads)
  }

  val executor = loss.bind(ctx, argsDict, gradDict)

  def forward(data: IndexedSeq[NDArray], label: IndexedSeq[NDArray]) = {

    this.argsDict("data").set(data(0))
    this.argsDict("label").set(label(0))
    this.executor.forward(isTrain = trainable)
  }

  def backward() = {
    this.executor.backward()
  }

  def updateGradients() = {
    this.paramGrads.foreach {
      case (idx, key, grad, state) =>
        optimizer.update(idx, this.argsDict(key), grad, state)
    }
  }

  def getOutputs(): Array[NDArray] = {
    this.executor.outputs
  }

  def saveCheckpoint(prefix: String, epoch: Int) = {

    Model.saveCheckpoint(prefix, epoch, symbol, argsDict, Map.empty[String,NDArray])
  }
}

