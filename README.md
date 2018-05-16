# blayze

A fast and flexible Naive Bayes implementation for the JVM written in Kotlin.

 * Fully supports the online learning paradigm, in which data, and even new features, are added as they become available.
 * Reasonably fast and memory efficient. We've trained a document classifier with tens of thousands of classes on hundreds of thousands of documents, and ironed out most of the hot-spots.
 * Models and data structures are immutable such that they are concurrency friendly.
 * Efficient serialization and deserialization using protobuf.
 * Missing and unknown features at prediction time are properly handled.
 * Minimal dependencies.
  
## Usage

Get the latest artifact from [maven central](https://search.maven.org/#search%7Cga%7C1%7Cg%3A%22com.tradeshift%22%20a%3A%22blayze%22) 

````java
Model model = new Model().batchAdd(Lists.newArrayList(new Update( //Models are immutable
                new Inputs( // Supports multiple feature types
                        ImmutableMap.of( //Text features
                                "subject", "Attention, is it true?", //features are named.
                                "body", "Good day dear beneficiary. This is Secretary to president of Benin republic is writing this email ..." // multiple features of the same type have different names
                        ),
                        ImmutableMap.of(  //Categorical features
                                "sender", "WWW.@galaxy.ocn.ne.jp"
                        ),
                        ImmutableMap.of(  //Gaussian features
                                "n_words", 482.
                        )
                ),
                "spam" // the outcome, in this case spam.
        )
));

Map<String, Double> predictions = model.predict(new Inputs(/*...*/));// e.g. {"spam": 0.624, "ham": 0.376}
````

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
