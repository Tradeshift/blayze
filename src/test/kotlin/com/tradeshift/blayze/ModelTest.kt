import com.tradeshift.blayze.Model
import com.tradeshift.blayze.Protos
import com.tradeshift.blayze.collection.Counter
import com.tradeshift.blayze.dto.Inputs
import com.tradeshift.blayze.dto.Update
import com.tradeshift.blayze.features.Categorical
import com.tradeshift.blayze.features.Multinomial
import com.tradeshift.blayze.features.Text
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.streams.toList


class ModelTest {

    @Test
    fun categorical_features_can_be_serialized_and_deserialized() {
        val bytes = model.toProto().toByteArray()
        val reconstructed = Model.fromProto(Protos.Model.parseFrom(bytes))

        val suggestions = reconstructed.predict(Inputs(categorical = mapOf(Pair("user", "ole"))))

        assertEquals(setOf("p", "n"), suggestions.keys)

        assertEquals(0.333333333, suggestions["p"]!!, 0.0000001)
        assertEquals(0.666666666, suggestions["n"]!!, 0.0000001)
    }

    @Test
    fun text_features_can_be_serialized_and_deserialized() {
        val bytes = model.toProto().toByteArray()
        val reconstructed = Model.fromProto(Protos.Model.parseFrom(bytes))

        val suggestions = reconstructed.predict(Inputs(text = mapOf(Pair("q", "awesome awesome awesome ok"))))

        // [[0.9268899 0.0731101]] from sklearn
        assertEquals(setOf("p", "n"), suggestions.keys)

        assertEquals(setOf("p", "n"), suggestions.keys)

        assertEquals(0.9268899, suggestions["p"]!!, 0.0000001)
        assertEquals(0.0731101, suggestions["n"]!!, 0.0000001)
    }


    @Test(expected = IllegalArgumentException::class)
    fun fails_to_deserialize_proto_with_different_version() {
        val proto = model.toProto()
        val differentVersionProto = Protos.Model.newBuilder(proto).setModelVersion(proto.modelVersion - 1).build()
        Model.fromProto(differentVersionProto)
    }

    @Test
    fun gaussian_features_can_be_serialized_and_deserialized() {
        val model = Model().batchAdd(
                listOf(
                        Update(Inputs(gaussian = mapOf(Pair("age", 20.0))), "eur"),
                        Update(Inputs(gaussian = mapOf(Pair("age", 30.0))), "eur"),
                        Update(Inputs(gaussian = mapOf(Pair("age", 40.0))), "usd"),
                        Update(Inputs(gaussian = mapOf(Pair("age", 50.0))), "usd"),
                        Update(Inputs(gaussian = mapOf(Pair("age", 40.0))), "eur")
                ))

        val bytes = model.toProto().toByteArray()
        val reconstructed = Model.fromProto(Protos.Model.parseFrom(bytes))

        val predictions = reconstructed.predict(Inputs(gaussian = mapOf(Pair("age", 23.0))))

        assertEquals(predictions.keys, setOf("usd", "eur"))

        var pEUR = 0.0312254 // https://www.wolframalpha.com/input/?i=normalpdf(23.0,+30.0,+10.0)
        var pUSD = 0.000446108 // https://www.wolframalpha.com/input/?i=normalpdf(23.0,+45.0,+7.0710678118654755)

        pUSD = pUSD * (2.0 / 5.0)
        pEUR = pEUR * (3.0 / 5.0)

        pUSD = pUSD / (pUSD + pEUR)
        pEUR = 1 - pUSD

        assertEquals(pUSD, predictions["usd"]!!, 0.000001)
        assertEquals(pEUR, predictions["eur"]!!, 0.000001)
    }

    @Test
    fun adding_empty_batches_does_not_delete_features() {
        var model = Model(textFeatures = mapOf("q" to Text(Multinomial(includeFeatureProbability = 0.5, pseudoCount = 0.01))))
        model = model.batchAdd(listOf())

        assertEquals(setOf("q"), model.toProto().textFeaturesMap.keys)
    }

    @Test
    fun adding_batches_without_a_feature_does_not_delete_that_feature() {
        var model = Model()
        model = model.add(Update(Inputs(text = mapOf("f1" to "foo bar baz")), "p"))
        model = model.add(Update(Inputs(categorical = mapOf("f2" to "map")), "p"))
        model = model.add(Update(Inputs(gaussian = mapOf("f3" to 23.3)), "p"))

        assertEquals(setOf("f1"), model.toProto().textFeaturesMap.keys)
        assertEquals(setOf("f2"), model.toProto().categoricalFeaturesMap.keys)
        assertEquals(setOf("f3"), model.toProto().gaussianFeaturesMap.keys)
    }

    @Test
    fun text_parameters_are_saved_when_serialized() {
        var model = Model(textFeatures = mapOf("q" to Text(Multinomial(includeFeatureProbability = 0.5, pseudoCount = 0.01))))
        model = model.batchAdd(listOf(
                Update(Inputs(text = mapOf("q" to "foo bar baz")), "p"),
                Update(Inputs(text = mapOf("q" to "zap foo foo")), "p")
        )).add(Update(Inputs(text = mapOf("q" to "map zap zee")), "n"))

        val bytes = model.toProto().toByteArray()
        val reconstructed = Model.fromProto(Protos.Model.parseFrom(bytes))
        val delegate = reconstructed.toProto().textFeaturesMap["q"]!!.delegate
        assertEquals(0.01, delegate.pseudoCount, 0.0)
        assertEquals(0.5, delegate.includeFeatureProbability, 0.0)
    }

    @Test
    fun can_fit_20newsgroup() {
        val train = newsgroup("20newsgroup_train.txt")
        val model = Model(textFeatures = mapOf("q" to Text(Multinomial(pseudoCount = 0.01)))).batchAdd(train)

        val test = newsgroup("20newsgroup_test.txt")
        val acc = test
                .parallelStream()
                .map {
                    if (it.outcome == model.predict(it.inputs).maxBy { it.value }?.key) {
                        1.0
                    } else {
                        0.0
                    }
                }
                .toList()
                .average()

        assertTrue(acc > 0.65) // sklearn MultinomialNB with a CountVectorizer gets ~0.646
    }


    @Test
    fun can_batch_add_gaussian_features() {
        val model = Model().batchAdd(
                listOf(
                        Update(Inputs(gaussian = mapOf(Pair("age", 20.0))), "eur"),
                        Update(Inputs(gaussian = mapOf(Pair("age", 30.0))), "eur"),
                        Update(Inputs(gaussian = mapOf(Pair("age", 40.0))), "usd")
                )).batchAdd(
                listOf(
                        Update(Inputs(gaussian = mapOf(Pair("age", 50.0))), "usd"),
                        Update(Inputs(gaussian = mapOf(Pair("age", 40.0))), "eur")
                )
        )
        val predictions = model.predict(Inputs(gaussian = mapOf(Pair("age", 23.0))))

        assertEquals(predictions.keys, setOf("usd", "eur"))

        var pEUR = 0.0312254 // https://www.wolframalpha.com/input/?i=normalpdf(23.0,+30.0,+10.0)
        var pUSD = 0.000446108 // https://www.wolframalpha.com/input/?i=normalpdf(23.0,+45.0,+7.0710678118654755)

        pUSD = pUSD * (2.0 / 5.0)
        pEUR = pEUR * (3.0 / 5.0)

        pUSD = pUSD / (pUSD + pEUR)
        pEUR = 1 - pUSD

        assertEquals(pUSD, predictions["usd"]!!, 0.000001)
        assertEquals(pEUR, predictions["eur"]!!, 0.000001)
    }

    @Test
    fun can_batch_add_categorical_features() {
        val model = Model().batchAdd(
                listOf(
                        Update(Inputs(categorical = mapOf(Pair("user", "alice"))), "usd"),
                        Update(Inputs(categorical = mapOf(Pair("user", "alice"))), "eur")

                )).batchAdd(
                listOf(
                        Update(Inputs(categorical = mapOf(Pair("user", "bob"))), "usd")
                )
        )
        val predictions = model.predict(Inputs(categorical = mapOf(Pair("user", "alice"))))

        assertEquals(predictions.keys, setOf("usd", "eur"))

        val nUsers = 2.0
        val nUSDUsers = 2.0
        val nEURUsers = 1.0
        val nAliceUSD = 1.0
        val nAliceEUR = 1.0
        val nUSD = 2.0
        val nEUR = 1.0
        val nPseudoCount = 1.0
        val logUSD = nUSD / (nUSD + nEUR) * ((nAliceUSD + nPseudoCount) / (nUSDUsers + nUsers * nPseudoCount))
        val logEUR = nEUR / (nUSD + nEUR) * ((nAliceEUR + nPseudoCount) / (nEURUsers + nUsers * nPseudoCount))
        val pUSD = logUSD / (logUSD + logEUR)
        val pEUR = 1 - pUSD

        assertEquals(pUSD, predictions["usd"]!!, 0.000001)
        assertEquals(pEUR, predictions["eur"]!!, 0.000001)
    }

    @Test
    fun can_batch_add_text_features() {
        val model = Model(textFeatures = mapOf("q" to Text(Multinomial()))).batchAdd(
                listOf(
                        Update(Inputs(text = mapOf(Pair("q", "foo bar baz"))), "positive"),
                        Update(Inputs(text = mapOf(Pair("q", "foo foo bar baz zap zoo"))), "negative")

                )).batchAdd(
                listOf(
                        Update(Inputs(text = mapOf(Pair("q", "map pap mee zap"))), "negative")
                )
        )
        val predictions = model.predict(Inputs(text = mapOf(Pair("q", "foo"))))

        assertEquals(predictions.keys, setOf("positive", "negative"))

        val nUniqueWords = 8.0
        val nPositiveWords = 3.0
        val nNegativeWords = 10.0
        val nFooPositive = 1.0
        val nFooNegative = 2.0
        val nPositive = 1.0
        val nNegative = 2.0
        val nPseudoCount = 1.0
        val up = nPositive / (nPositive + nNegative) * ((nFooPositive + nPseudoCount) / (nPositiveWords + nUniqueWords * nPseudoCount))
        val un = nNegative / (nPositive + nNegative) * ((nFooNegative + nPseudoCount) / (nNegativeWords + nUniqueWords * nPseudoCount))
        val p = up / (up + un)
        val n = 1 - p

        assertEquals(p, predictions["positive"]!!, 0.000001)
        assertEquals(n, predictions["negative"]!!, 0.000001)
    }

    @Test
    fun single_categorical_feature_give_correct_probability() {
        val suggestions = model.predict(Inputs(categorical = mapOf(Pair("user", "ole"))))

        /*
        Expected probabilities:

        P(p) = 1/4
        P(n) = 3/4
        From https://en.wikipedia.org/wiki/Categorical_distribution#Posterior_predictive_distribution
        and p has seen {ole} and n has seen {ole, bob, ada}

        posterior P(ole | p) = (1 + 1) / (1 + 3) = 1 / 2
        posterior P(ole | n) = (1 + 1) / (3 + 3) = 1 / 3

        P(p | ole) = P(p) * P(ole | p) / Norm = (1/4 * 1/2) / ((1/4 * 1/2) + ((3/4) * (1/3))) = 1/3
        P(n | ole) = P(n) * P(ole | n) / Norm = (3/4 * 1/3) / ((1/4 * 1/2) + ((3/4) * (1/3))) = 2/3
        */
        assertEquals(setOf("p", "n"), suggestions.keys)

        assertEquals(setOf("p", "n"), suggestions.keys)

        assertEquals(0.333333333, suggestions["p"]!!, 0.0000001)
        assertEquals(0.666666666, suggestions["n"]!!, 0.0000001)
    }

    @Test
    fun multiple_categorical_features_give_correct_probabilities() {
        val suggestions = model.predict(Inputs(categorical = mapOf(Pair("user", "ole"), Pair("contry", "dystopia"))))

        val contyPosP = (0.0 + 1.0) / (1.0 + 2.0)
        val contryPosN = (1.0 + 1.0) / (1.0 + 2.0)

        val norm = (0.333333333 * contyPosP) + (0.666666666 * contryPosN)
        val pos = (0.333333333 * contyPosP) / norm
        val neg = (0.666666666 * contryPosN) / norm

        assertEquals(setOf("p", "n"), suggestions.keys)

        assertEquals(setOf("p", "n"), suggestions.keys)

        assertEquals(pos, suggestions["p"]!!, 0.0000001)
        assertEquals(neg, suggestions["n"]!!, 0.0000001)
    }

    @Test
    fun single_text_feature_gives_correct_probabilities() {
        val suggestions = model.predict(Inputs(text = mapOf(Pair("q", "awesome awesome awesome ok"))))

        // [[0.9268899 0.0731101]] from sklearn
        assertEquals(setOf("p", "n"), suggestions.keys)

        assertEquals(setOf("p", "n"), suggestions.keys)

        assertEquals(0.9268899, suggestions["p"]!!, 0.0000001)
        assertEquals(0.0731101, suggestions["n"]!!, 0.0000001)
    }

    @Test
    fun multiple_text_features_gives_correct_probability() {
        val suggestions = model.predict(Inputs(text = mapOf(Pair("q", "awesome ok"), Pair("other_q", "awesome awesome"))))

        assertEquals(setOf("p", "n"), suggestions.keys)

        assertEquals(setOf("p", "n"), suggestions.keys)

        assertEquals(0.9268899, suggestions["p"]!!, 0.0000001)
        assertEquals(0.0731101, suggestions["n"]!!, 0.0000001)
    }

    @Test
    fun multiple_feature_types_are_considered() {
        val suggestions = model.predict(
                Inputs(
                        text = mapOf(Pair("q", "awesome ok"), Pair("other_q", "awesome awesome")),
                        categorical = mapOf(Pair("user", "ole"))
                )
        )

        assertEquals(setOf("p", "n"), suggestions.keys)
        assertEquals(setOf("p", "n"), suggestions.keys)

        val categoricalPosteriorP = 0.5
        val categoricalPosteriorN = 0.33333333

        val textPriorAndPosteriorP = 0.9268899
        val textPriorAndPosteriorN = 0.0731101

        val norm = (categoricalPosteriorP * textPriorAndPosteriorP) + (categoricalPosteriorN * textPriorAndPosteriorN)

        val pos = (categoricalPosteriorP * textPriorAndPosteriorP) / norm
        val neg = (categoricalPosteriorN * textPriorAndPosteriorN) / norm

        assertEquals(pos, suggestions["p"]!!, 0.0000001)
        assertEquals(neg, suggestions["n"]!!, 0.0000001)
    }

    @Test
    fun empty_features_default_to_prior() {
        val suggestions = model.predict(Inputs())

        assertEquals(setOf("p", "n"), suggestions.keys)

        assertEquals(0.25, suggestions["p"]!!, 0.0000001)
        assertEquals(0.75, suggestions["n"]!!, 0.0000001)
    }

    @Test
    fun features_with_only_stopwords() { //this and that are stopwords and are removed from the inputs
        val textInputA = Inputs(text = mapOf("feature_A" to "this this this", "feature_B" to "that that that"))
        val textInputB = Inputs(text = mapOf("feature_A" to "that that that", "feature_B" to "this this this"))

        val model = Model()
                .batchAdd(
                        listOf(
                                Update(textInputA, "A"),
                                Update(textInputA, "A"),
                                Update(textInputA, "A"),
                                Update(textInputB, "B"),
                                Update(textInputB, "B")
                        )
                )
        val prediction = model.predict(textInputA)

        // So it falls back to the prior
        assertEquals(3.0 / 5.0, prediction["A"]!!, 0.0000001)
        assertEquals(2.0 / 5.0, prediction["B"]!!, 0.0000001)
    }

    @Test
    fun unseen_features_default_to_prior() {
        val suggestions = model.predict(Inputs(
                mapOf(Pair("q", "k k k k k k k k k k k k k k k k k")),
                mapOf(Pair("user", "notseen"))
        ))
        assertEquals(setOf("p", "n"), suggestions.keys)

        assertEquals(0.25, suggestions["p"]!!, 0.0000001)
        assertEquals(0.75, suggestions["n"]!!, 0.0000001)
    }

    @Test
    fun special_chars_are_replaced_by_space() {
        val suggestions = model.predict(Inputs(text = mapOf(Pair("q", "awesome.!!    awesome;;;awesome \t\n ok"))))

        // [[0.9268899 0.0731101]] from sklearn
        assertEquals(setOf("p", "n"), suggestions.keys)

        assertEquals(setOf("p", "n"), suggestions.keys)

        assertEquals(0.9268899, suggestions["p"]!!, 0.0000001)
        assertEquals(0.0731101, suggestions["n"]!!, 0.0000001)
    }

    private val model: Model
        get() {
            val priorCounts = mapOf(
                    Pair("p", 1),
                    Pair("n", 3)
            )

            val textFeatures = mapOf(
                    Pair(
                            "q",
                            Text(
                                    Multinomial(
                                            1.0,
                                            1.0
                                    ).batchUpdate(
                                            listOf(
                                                    "p" to Counter(mapOf("awesome" to 7, "terrible" to 3, "ok" to 19)),
                                                    "n" to Counter(mapOf("awesome" to 2, "terrible" to 13, "ok" to 21))
                                            )
                                    )
                            )

                    ),
                    Pair(
                            "other_q",
                            Text(
                                    Multinomial(
                                            1.0,
                                            1.0
                                    ).batchUpdate(
                                            listOf(
                                                    "p" to Counter(mapOf("awesome" to 7, "terrible" to 3, "ok" to 19)),
                                                    "n" to Counter(mapOf("awesome" to 2, "terrible" to 13, "ok" to 21))
                                            )
                                    )
                            )
                    ))

            val categoricalFeatures = mapOf(Pair(
                    "user",
                    Categorical(
                            Multinomial(
                                    1.0,
                                    1.0
                            ).batchUpdate(
                                    listOf(
                                            "p" to Counter("ole"),
                                            "n" to Counter("ole", "bob", "ada")
                                    )
                            )
                    )
            ),
                    Pair(
                            "contry",
                            Categorical(
                                    Multinomial(
                                            1.0,
                                            1.0
                                    ).batchUpdate(
                                            listOf(
                                                    "p" to Counter("utopia"),
                                                    "n" to Counter("dystopia")
                                            )
                                    )
                            )
                    ))

            return Model(priorCounts, textFeatures, categoricalFeatures)
        }


    fun newsgroup(fname: String): List<Update> {
        val lines = this::class.java.getResource(fname).readText(Charsets.UTF_8).split("\n")
        val updates = mutableListOf<Update>()

        for (line in lines) {
            val split = line.split(" ".toRegex(), 2).toTypedArray()
            val outcome = split[0]
            var f = Inputs()
            if (split.size == 2) { //some are legit empty
                f = Inputs(mapOf("q" to split[1]))
            }
            updates.add(Update(f, outcome))
        }
        return updates
    }

}


