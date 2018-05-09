# blayze

A fast and flexible Naive Bayes implementation for the JVM written in Kotlin.

 * Fully supports the online learning paradigm, in which data, and even new features, are added as they become available.
 * Reasonably fast and memory efficient. We've trained a document classifier with tens of thousands of classes on hundreds of thousands of documents, and ironed out most of the hot-spots.
 * Models and data structures are immutable such that they are concurrency friendly.
 * Efficient serialization and deserialization using protobuf.
 * Missing and unknown features at prediction time are properly handled.
 * Minimal dependencies.
  
## Usage

```kotlin
val model = Model().batchAdd(listOf(Update(
                        inputs = Inputs( // Supports multiple feature types: text, categorical and gaussian.
                                text = mapOf( 
                                        "subject" to "Attention, is it true?", //features are named.
                                        "body" to "Good day dear beneficiary. This is Secretary to president of Benin republic is writing this email ..." // multiple features of the same type have different names  
                                ),
                                categorical = mapOf(
                                        "sender" to "WWW.@galaxy.ocn.ne.jp"
                                )
                        ),
                        outcome = "spam" // the outcome, in this case spam.
                )
        ))
        
val predictions: Map<Outcome, Double> = model.predict(Inputs(/*...*/)) // e.g. {"spam": 0.624, "ham": 0.376}
```

## Built With
 * [Kotlin](https://kotlinlang.org/) - Language
 * [Maven](https://maven.apache.org/) - Dependency Management
 * [Protocol Buffers](https://developers.google.com/protocol-buffers/) - Serialization
 
## Versioning

We use [SemVer](http://semver.org/) for versioning.

## Authors

 * [Rasmus Berg Palm](https://github.com/rasmusbergpalm)
 * [Fuyang Liu](https://github.com/liufuyang)
 * [Lasse Reedtz](https://github.com/lre)
