import com.tradeshift.blayze.Model
import com.tradeshift.blayze.Protos
import com.tradeshift.blayze.collection.Counter
import com.tradeshift.blayze.dto.Inputs
import com.tradeshift.blayze.dto.Update
import com.tradeshift.blayze.features.Categorical
import com.tradeshift.blayze.features.Gaussian
import com.tradeshift.blayze.features.Multinomial
import com.tradeshift.blayze.features.Text
import io.mockk.every
import io.mockk.mockk
import io.mockk.verify
import org.junit.Assert.*
import org.junit.Test
import kotlin.streams.toList


class ModelTest {

    private val priorCounts = mapOf(
            Pair("p", 1),
            Pair("n", 3)
    )
    private val textFeatures = mapOf(
            "q" to Text(Multinomial(
                    1.0,
                    1.0
            ).batchUpdate(listOf(
                    "p" to Counter(mapOf("awesome" to 7, "terrible" to 3, "ok" to 19)),
                    "n" to Counter(mapOf("awesome" to 2, "terrible" to 13, "ok" to 21))
            ))),
            "other_q" to Text(Multinomial(
                    1.0,
                    1.0
            ).batchUpdate(listOf(
                    "p" to Counter(mapOf("awesome" to 7, "terrible" to 3, "ok" to 19)),
                    "n" to Counter(mapOf("awesome" to 2, "terrible" to 13, "ok" to 21))
            )))
    )

    private val categoricalFeatures = mapOf(
            "user" to Categorical(
                    Multinomial(
                            1.0,
                            1.0
                    ).batchUpdate(
                            listOf(
                                    "p" to Counter("ole"),
                                    "n" to Counter("ole", "bob", "ada")
                            )
                    )
            ),
            "country" to
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
    )

    val model = Model(priorCounts, textFeatures, categoricalFeatures)

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
        val inputs = Inputs(text = mapOf(Pair("q", "awesome awesome awesome ok")))
        val expected = model.predict(inputs)

        val bytes = model.toProto().toByteArray()
        val reconstructed = Model.fromProto(Protos.Model.parseFrom(bytes))
        val actual = reconstructed.predict(inputs)

        assertEquals(expected, actual)
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

        val reconstructed = Model.fromProto(Protos.Model.parseFrom(model.toProto().toByteArray()))

        val inputs = Inputs(gaussian = mapOf(Pair("age", 23.0)))
        val expected = model.predict(inputs)
        val actual = reconstructed.predict(inputs)

        assertEquals(expected, actual)
    }

    @Test
    fun outcomes_of_gaussian_features_with_less_than_two_samples_does_not_dominate() {
        val model = Model().batchAdd(
                listOf(
                        Update(Inputs(
                                gaussian = mapOf(Pair("weather.degree", 15.0))),
                                "t-shirt"),
                        Update(Inputs(
                                gaussian = mapOf(Pair("weather.degree", 19.0))),
                                "t-shirt"),
                        Update(Inputs(
                                gaussian = mapOf(Pair("weather.degree", 16.0))),
                                "t-shirt"),
                        Update(Inputs(
                                gaussian = mapOf(Pair("weather.degree", 21.0))),
                                "t-shirt"),
                        Update(Inputs(
                                gaussian = mapOf(Pair("weather.degree", -10.0))),
                                "sweater")
                )
        )

        val predictions = model.predict(Inputs(
                gaussian = mapOf(Pair("weather.degree", 14.0))
        ))

        val actual = predictions["t-shirt"]!!
        assertEquals(0.8, actual, 1e-6)
    }

    @Test
    fun adding_a_gaussian_feature_does_not_break_classifier() {
        val model = Model().batchAdd(
                listOf(
                        Update(Inputs(categorical = mapOf("weather.label" to "warm")), "t-shirt"),
                        Update(Inputs(categorical = mapOf("weather.label" to "warm")), "t-shirt"),
                        Update(Inputs(categorical = mapOf("weather.label" to "warm")), "t-shirt"),
                        Update(Inputs(categorical = mapOf("weather.label" to "warm")), "t-shirt"),
                        Update(Inputs(
                                categorical = mapOf("weather.label" to "cold"),
                                gaussian = mapOf(Pair("weather.degree", -10.0))
                        ),
                                "sweater")
                )
        )
        val p1 = model.predict(Inputs(
                categorical = mapOf("weather.label" to "warm"))
        )
        val a1 = p1["t-shirt"]!!
        assertTrue("expected $a1 > 0.9", a1 > 0.9)

        val p2 = model.predict(Inputs(
                categorical = mapOf("weather.label" to "warm"),
                gaussian = mapOf("weather.degree" to 22.0)
        ))
        val a2 = p2["t-shirt"]!!
        assertTrue("expected $a2 > 0.9", a2 > 0.9)
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

        var pEUR = 0.02782119452355812 // d = np.array([20, 30, 40]); n=d.size; scipy.stats.t.pdf(23.0, n, d.mean(), d.var()*(n+1)/n))
        var pUSD = 0.002837354211344904 // d = np.array([40, 50]); n=d.size; scipy.stats.t.pdf(23.0, n, d.mean(), d.var()*(n+1)/n))

        pUSD = pUSD * (2.0 / 5.0)
        pEUR = pEUR * (3.0 / 5.0)

        pUSD = pUSD / (pUSD + pEUR)
        pEUR = 1 - pUSD

        assertEquals(pUSD, predictions["usd"]!!, 0.000001)
        assertEquals(pEUR, predictions["eur"]!!, 0.000001)
    }

    @Test
    fun can_batch_add_categorical_features() {
        val model = Model(categoricalFeatures = mapOf("user" to Categorical(pseudoCount = 1.0))).batchAdd(
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
        val model = Model().batchAdd(
                listOf(
                        Update(Inputs(text = mapOf(Pair("q", "foo bar baz"))), "p"),
                        Update(Inputs(text = mapOf(Pair("q", "foo foo bar baz zap zoo"))), "n")

                )).batchAdd(
                listOf(
                        Update(Inputs(text = mapOf(Pair("q", "map pap mee zap"))), "n")
                )
        )
        val actual = model.predict(Inputs(text = mapOf(Pair("q", "foo"))))

        val f = Text().batchUpdate(listOf(
                "p" to "foo bar baz",
                "n" to "foo foo bar baz zap zoo",
                "n" to "map pap mee zap"
        ))

        val pq = f.logPosteriorPredictive(setOf("p", "n"), "foo")

        val up = Math.exp(Math.log(1.0) + pq["p"]!!)
        val un = Math.exp(Math.log(2.0) + pq["n"]!!)

        val pp = up / (up + un)
        val pn = 1.0 - pp

        assertEquals(pp, actual["p"]!!, 1e-6)
        assertEquals(pn, actual["n"]!!, 1e-6)
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
        val suggestions = model.predict(Inputs(categorical = mapOf(Pair("user", "ole"), Pair("country", "dystopia"))))

        val contyPosP = (0.0 + 1.0) / (1.0 + 2.0)
        val countryPosN = (1.0 + 1.0) / (1.0 + 2.0)

        val norm = (0.333333333 * contyPosP) + (0.666666666 * countryPosN)
        val pos = (0.333333333 * contyPosP) / norm
        val neg = (0.666666666 * countryPosN) / norm

        assertEquals(setOf("p", "n"), suggestions.keys)

        assertEquals(setOf("p", "n"), suggestions.keys)

        assertEquals(pos, suggestions["p"]!!, 0.0000001)
        assertEquals(neg, suggestions["n"]!!, 0.0000001)
    }

    @Test
    fun single_text_feature_gives_correct_probabilities() {
        val suggestions = model.predict(Inputs(text = mapOf(Pair("q", "awesome awesome awesome ok"))))

        val pq = textFeatures["q"]!!.logPosteriorPredictive(setOf("p", "n"), "awesome awesome awesome ok")

        val up = Math.exp(Math.log(priorCounts["p"]!!.toDouble()) + pq["p"]!!)
        val un = Math.exp(Math.log(priorCounts["n"]!!.toDouble()) + pq["n"]!!)

        val pp = up / (up + un)
        val pn = 1.0 - pp

        assertEquals(pp, suggestions["p"]!!, 1e-6)
        assertEquals(pn, suggestions["n"]!!, 1e-6)
    }

    @Test
    fun multiple_text_features_gives_correct_probability() {
        val suggestions = model.predict(Inputs(text = mapOf(Pair("q", "awesome ok"), Pair("other_q", "awesome awesome"))))

        val pq = textFeatures["q"]!!.logPosteriorPredictive(setOf("p", "n"), "awesome ok")
        val pq2 = textFeatures["other_q"]!!.logPosteriorPredictive(setOf("p", "n"), "awesome awesome")

        val up = Math.exp(Math.log(priorCounts["p"]!!.toDouble()) + pq["p"]!! + pq2["p"]!!)
        val un = Math.exp(Math.log(priorCounts["n"]!!.toDouble()) + pq["n"]!! + pq2["n"]!!)

        val pp = up / (up + un)
        val pn = 1.0 - pp

        assertEquals(pp, suggestions["p"]!!, 1e-6)
        assertEquals(pn, suggestions["n"]!!, 1e-6)
    }

    @Test
    fun multiple_feature_types_are_considered() {
        val suggestions = model.predict(
                Inputs(
                        text = mapOf(Pair("q", "awesome ok"), Pair("other_q", "awesome awesome")),
                        categorical = mapOf(Pair("user", "ole"))
                )
        )

        val pq = textFeatures["q"]!!.logPosteriorPredictive(setOf("p", "n"), "awesome ok")
        val pq2 = textFeatures["other_q"]!!.logPosteriorPredictive(setOf("p", "n"), "awesome awesome")
        val pq3 = categoricalFeatures["user"]!!.logPosteriorPredictive(setOf("p", "n"), "ole")

        val up = Math.exp(Math.log(priorCounts["p"]!!.toDouble()) + pq["p"]!! + pq2["p"]!! + pq3["p"]!!)
        val un = Math.exp(Math.log(priorCounts["n"]!!.toDouble()) + pq["n"]!! + pq2["n"]!! + pq3["n"]!!)

        val pp = up / (up + un)
        val pn = 1.0 - pp

        assertEquals(pp, suggestions["p"]!!, 1e-6)
        assertEquals(pn, suggestions["n"]!!, 1e-6)
    }

    @Test
    fun empty_features_default_to_prior() {
        val suggestions = model.predict(Inputs())

        assertEquals(setOf("p", "n"), suggestions.keys)

        assertEquals(0.25, suggestions["p"]!!, 0.0000001)
        assertEquals(0.75, suggestions["n"]!!, 0.0000001)
    }

    @Test
    fun unseen_features_are_ignored_at_prediction_time() {
        val suggestions = model.predict(Inputs(
                mapOf(Pair("q2", "k k k k k k k k k k k k k k k k k")),
                mapOf(Pair("user2", "notseen"))
        ))
        assertEquals(setOf("p", "n"), suggestions.keys)

        assertEquals(0.25, suggestions["p"]!!, 0.0000001)
        assertEquals(0.75, suggestions["n"]!!, 0.0000001)
    }

    @Test
    fun withParameters_blows_up_if_parameters_are_given_for_feature_that_does_not_exist() {
        try {
            Model().withParameters(Model.Parameters(text = mapOf("foo" to Multinomial.Parameters())))
            fail("Expected IllegalArgumentException")
        } catch (e: IllegalArgumentException) {
            assertEquals("No feature named 'foo'", e.message)
        }
    }

    @Test
    fun predict_with_null_parameters_call_features_with_null_parameters() {
        val text = mockk<Text>()
        every { text.logPosteriorPredictive(any(), any(), any()) } returns mapOf("o1" to -1.0, "o2" to -2.0)

        val categorical = mockk<Categorical>()
        every { categorical.logPosteriorPredictive(any(), any(), any()) } returns mapOf("o1" to -1.0, "o2" to -2.0)

        val gaussian = mockk<Gaussian>()
        every { gaussian.logPosteriorPredictive(any(), any(), any()) } returns mapOf("o1" to -1.0, "o2" to -2.0)

        val m = Model(mapOf("o1" to 1, "o2" to 1), mapOf("foo" to text), mapOf("bar" to categorical), mapOf("baz" to gaussian), 0)
        m.predict(Inputs(mapOf("foo" to "yes yes"), mapOf("bar" to "nope"), mapOf("baz" to 1.0)))

        verify { text.logPosteriorPredictive(setOf("o1", "o2"), "yes yes", null) }
        verify { categorical.logPosteriorPredictive(setOf("o1", "o2"), "nope", null) }
        verify { gaussian.logPosteriorPredictive(setOf("o1", "o2"), 1.0, null) }
    }

    @Test
    fun predict_with_non_null_parameters_call_features_with_given_parameters() {
        val text1 = mockk<Text>()
        every { text1.logPosteriorPredictive(any(), any(), any()) } returns mapOf("o1" to -1.0, "o2" to -2.0)

        val text2 = mockk<Text>()
        every { text2.logPosteriorPredictive(any(), any(), any()) } returns mapOf("o1" to -1.0, "o2" to -2.0)

        val categorical = mockk<Categorical>()
        every { categorical.logPosteriorPredictive(any(), any(), any()) } returns mapOf("o1" to -1.0, "o2" to -2.0)

        val gaussian = mockk<Gaussian>()
        every { gaussian.logPosteriorPredictive(any(), any(), any()) } returns mapOf("o1" to -1.0, "o2" to -2.0)

        val m = Model(mapOf("o1" to 1, "o2" to 1), mapOf("foo1" to text1, "foo2" to text2), mapOf("bar" to categorical), mapOf("baz" to gaussian), 0)
        m.predict(Inputs(mapOf("foo1" to "yes yes", "foo2" to "no no"), mapOf("bar" to "nope"), mapOf("baz" to 1.0)), Model.Parameters(
                0,
                mapOf("foo1" to Multinomial.Parameters(0.51, 0.52)),
                mapOf("bar" to Multinomial.Parameters(0.31, 0.32)),
                mapOf("baz" to Gaussian.Parameters(1.0, 1, 2.0, 2))
        ))

        verify { text1.logPosteriorPredictive(setOf("o1", "o2"), "yes yes", Multinomial.Parameters(0.51, 0.52)) }
        verify { text2.logPosteriorPredictive(setOf("o1", "o2"), "no no", null) }
        verify { categorical.logPosteriorPredictive(setOf("o1", "o2"), "nope", Multinomial.Parameters(0.31, 0.32)) }
        verify { gaussian.logPosteriorPredictive(setOf("o1", "o2"), 1.0, Gaussian.Parameters(1.0, 1, 2.0, 2)) }
    }

    @Test
    fun batchAdd_with_null_parameters_call_features_with_null_parameters() {
        val text = mockk<Text>()
        every { text.batchUpdate(any(), any()) } returns text

        val categorical = mockk<Categorical>()
        every { categorical.batchUpdate(any(), any()) } returns categorical

        val gaussian = mockk<Gaussian>()
        every { gaussian.batchUpdate(any(), any()) } returns gaussian

        val m = Model(mapOf("o1" to 1, "o2" to 1), mapOf("foo" to text), mapOf("bar" to categorical), mapOf("baz" to gaussian), 0)
        m.batchAdd(listOf(Update(Inputs(mapOf("foo" to "yes yes"), mapOf("bar" to "nope"), mapOf("baz" to 1.0)), "o1")))

        verify { text.batchUpdate(listOf("o1" to "yes yes"), null) }
        verify { categorical.batchUpdate(listOf("o1" to "nope"), null) }
        verify { gaussian.batchUpdate(listOf("o1" to 1.0), null) }
    }

    @Test
    fun batchAdd_with_non_null_parameters_call_features_with_given_parameters() {
        val text1 = mockk<Text>()
        every { text1.batchUpdate(any(), any()) } returns text1

        val text2 = mockk<Text>()
        every { text2.batchUpdate(any(), any()) } returns text2

        val categorical = mockk<Categorical>()
        every { categorical.batchUpdate(any(), any()) } returns categorical

        val gaussian = mockk<Gaussian>()
        every { gaussian.batchUpdate(any(), any()) } returns gaussian

        val m = Model(mapOf("o1" to 1, "o2" to 1), mapOf("foo1" to text1, "foo2" to text2), mapOf("bar" to categorical), mapOf("baz" to gaussian), 0)
        m.batchAdd(listOf(Update(Inputs(mapOf("foo1" to "yes yes", "foo2" to "yes no"), mapOf("bar" to "nope"), mapOf("baz" to 1.0)), "o1")), Model.Parameters(
                0,
                mapOf("foo1" to Multinomial.Parameters(0.51, 0.52)),
                mapOf("bar" to Multinomial.Parameters(0.31, 0.32)),
                mapOf("baz" to Gaussian.Parameters(1.0, 1, 2.0, 2))
        ))

        verify { text1.batchUpdate(listOf("o1" to "yes yes"), Multinomial.Parameters(0.51, 0.52)) }
        verify { text2.batchUpdate(listOf("o1" to "yes no"), null) }
        verify { categorical.batchUpdate(listOf("o1" to "nope"), Multinomial.Parameters(0.31, 0.32)) }
        verify { gaussian.batchUpdate(listOf("o1" to 1.0), Gaussian.Parameters(1.0, 1, 2.0, 2)) }
    }

}


