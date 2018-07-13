package com.favalos.autoencoder

import org.apache.mxnet.Symbol

object SymbolBuilder {

  def buildDeepEncoder(input: Symbol, encoderDims: Option[Array[Int]],
                       decoderDims: Option[Array[Int]], dropoutVal: Option[Float] = None): Symbol = {

    val encoderSym = encoder(input, encoderDims)
    val dropoutSym = dropout("enc", encoderSym, dropoutVal)
    val decoderSym = decoder(dropoutSym, decoderDims)

    decoderSym
  }

  def encoder(input: Symbol, dimsOpt: Option[Array[Int]], activation: String = "relu"): Symbol = {

    dimsOpt match {
      case Some(dims) => {
        val encoderNetwork = createNetwork("enc", dims, input, activation)
        encoderNetwork
      }
      case _ => throw new RuntimeException("Missing dims for encoding.")
    }
  }

  def decoder(input: Symbol, dimsOpt: Option[Array[Int]], activation: String = "relu"): Symbol = {

    dimsOpt match {
      case Some(dims) => {
        val decoderNetwork = createNetwork("dec", dims, input, activation)
        decoderNetwork
      }
      case _ => throw new RuntimeException("Missing dims for decoding.")
    }
  }

  def dropout(prefix: String, input: Symbol, percent: Option[Float]) : Symbol = {

    percent match {
      case Some(p) =>
        Symbol.Dropout(s"${prefix}_dropout")()(Map("data" -> input, "p" -> p))
      case _ =>
        input
    }
  }

  private[autoencoder] def createNetwork(prefix: String, dims: Array[Int],
                                         input: Symbol, activation: String): Symbol = {

    val network = dims.zip(1 to dims.length).foldLeft(input){

      ( layer, idx ) =>
        val fc = Symbol.FullyConnected(s"${prefix}_dl${idx._2}")()(Map("data" -> layer, "num_hidden" -> idx._1))
        val act = Symbol.Activation(s"${prefix}_act${idx._2}")()(Map("data" -> fc, "act_type" -> activation))
        act
    }

    network
  }

}
