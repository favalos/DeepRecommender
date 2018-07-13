name := "DeepRecomender"

version := "0.1"

scalaVersion := "2.11.12"

resolvers += Resolver.mavenLocal

libraryDependencies ++= Seq(
  "org.apache.mxnet" % "mxnet-full_2.11-osx-x86_64-cpu" % "1.2.0-SNAPSHOT"
)