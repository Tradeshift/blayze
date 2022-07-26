# blayze

A fast and flexible Bayesian Naive Bayes implementation for the JVM written in Kotlin.

* Fully supports the online learning paradigm, in which data, and even new features, are added as they become available.
* Reasonably fast and memory efficient. We've trained a document classifier with tens of thousands of classes on hundreds of thousands of documents, and ironed out most of the hot-spots.
* Naturally works with few samples, by integrating out the uncertainty on estimated parameters.
* Models and data structures are immutable such that they are concurrency friendly.
* Efficient serialization and deserialization using protobuf.
* Missing and unknown features at prediction time are properly handled.
* Minimal dependencies.

## Usage

Get the latest artifact from [maven central](https://search.maven.org/#search%7Cga%7C1%7Cg%3A%22com.tradeshift%22%20a%3A%22blayze%22)

````java
//Java 9
Model model = new Model().batchAdd(List.of(new Update( //Models are immutable
    new Inputs( // Supports multiple feature types
    Map.of( //Text features
    "subject", "Attention, is it true?", //features are named.
    "body", "Good day dear beneficiary. This is Secretary to president of Benin republic is writing this email ..." // multiple features of the same type have different names
    ),
    Map.of( //Categorical features
    "sender", "WWW.@galaxy.ocn.ne.jp"
    ),
    Map.of( //Gaussian features
    "n_words", 482.
    )
    ),
    "spam" // the outcome, in this case spam.
    )));

    Map<String, Double> predictions = model.predict(new Inputs(/*...*/));// e.g. {"spam": 0.624, "ham": 0.376}
````

## Built With
* [Kotlin](https://kotlinlang.org/) - Language
* [Maven](https://maven.apache.org/) - Dependency Management
* [Protocol Buffers](https://developers.google.com/protocol-buffers/) - Serialization

## Versioning

We use [SemVer](http://semver.org/) for versioning.

## Release a new version
- [Create a Sonatype account](https://issues.sonatype.org/secure/Signup!default.jspa)
    - The created username and password will be referred to as `<sonatype_user>` and `<sonatype_pwd`, respectively.
- Create a Sonatype Jira ticket of type `Publishing Support` requesting access to `com.tradeshift.blayze`.
  It must be approved by an existing user with write access. It can take a couple of days before access is granted.
- Create a PR that updates the version in `pom.xml` along with the code changes. Merge it to `master` or `v4`
  once it is approved.
- Generate a gpg key: `gpg --gen-key`
- List gpg keys: `gpg --list-keys`
- Extract the key id of the previously generated gpg key. It will be referred to as `<gpg_key_id>` from now
- Encrypt `<sonatype_pwd>` using `mvn --encrypt-password`. The encrypted value is referred to as `<sonatype_pwd_enc>`
- Create a new server in `~/.m2/settings.xml`
```
<settings>
    <servers>
        <server>
            <id>ossrh-blayze</id>
            <username><sonatype_user></username>
            <password><sonatype_pwd_enc></password>
        </server>      
    </servers>
</settings>
```
- Run `mvn clean deploy -P release -Dgpg.keyname=<gpg_key_id>`
- For further details, check [Sonatype documentation](https://central.sonatype.org/publish/publish-guide/)

## Backwards compatibility
We publish security updates for major version `4.x.x` (branch `v4`) as well as `6.x.x` (branch `master`)

## Authors

* [Rasmus Berg Palm](https://github.com/rasmusbergpalm)
* [Fuyang Liu](https://github.com/liufuyang)
* [Lasse Reedtz](https://github.com/lre)
