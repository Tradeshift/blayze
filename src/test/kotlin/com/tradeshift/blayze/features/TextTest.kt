package com.tradeshift.blayze.features

import com.tradeshift.blayze.collection.Counter
import io.mockk.mockk
import io.mockk.verify
import org.junit.Assert.assertEquals
import org.junit.Test

class TextTest {

    private val outcomes = setOf("p", "n")
    private val multinomial = Multinomial(pseudoCount = 1.0).batchUpdate(
            listOf(
                    "p" to Counter(mapOf("awesome" to 7, "terrible" to 3, "ok" to 19)),
                    "n" to Counter(mapOf("awesome" to 2, "terrible" to 13, "ok" to 21))
            )
    )

    @Test
    fun splits_text_and_delegates_to_multinomial() {
        val text = Text(multinomial)
        val expected = multinomial.logPosteriorPredictive(outcomes, Counter("awesome", "awesome", "awesome", "ok"))
        val actual = text.logPosteriorPredictive(outcomes, "awesome awesome awesome ok")

        assertEquals(expected, actual)
    }

    @Test
    fun batch_update_sums_seen_words() {
        val mult = Multinomial(pseudoCount = 1.0).batchUpdate(listOf(
                "p" to Counter(mapOf("awesome" to 7, "ok" to 19, "terrible" to 3)),
                "n" to Counter(mapOf("awesome" to 2, "ok" to 21, "terrible" to 13))

        ))
        val expected = mult.logPosteriorPredictive(outcomes, Counter("awesome", "awesome", "awesome", "ok"))

        val text = Text(pseudoCount = 1.0)
                .batchUpdate(listOf(
                        "p" to "awesome awesome awesome awesome awesome awesome awesome",
                        "p" to "ok ok ok ok ok ok ok ok ok ok terrible terrible terrible ",
                        "n" to "awesome awesome ok ok ok ok ok ok ok ok ok ok ok ok ok ok ok"))
                .batchUpdate(listOf(
                        "p" to "ok ok ok ok ok ok ok ok ok",
                        "n" to "ok ok ok ok ok ok terrible terrible terrible terrible",
                        "n" to "terrible terrible terrible terrible terrible terrible terrible terrible terrible"))


        val actual = text.logPosteriorPredictive(outcomes, "awesome awesome awesome ok")

        assertEquals(expected, actual)
    }

    @Test
    fun test_withParameters_sets_parameters() {
        val parameters = Multinomial.Parameters(0.0892275415, 0.7363151583)
        val text = Text().withParameters(parameters)
        assertEquals(text.delegate.defaultParams, parameters)
    }

    @Test
    fun test_logPosteriorPredictive_delegates_parameters_to_multinomial() {
        val mult = mockk<Multinomial>(relaxed = true)
        Text(mult).logPosteriorPredictive(setOf("foo", "bar"), "baz", Multinomial.Parameters(0.32, 0.23))
        verify { mult.logPosteriorPredictive(setOf("foo", "bar"), Counter("baz"), Multinomial.Parameters(0.32, 0.23)) }
    }

    @Test
    fun test_logPosteriorPredictive_delegates_null_parameters_to_multinomial() {
        val mult = mockk<Multinomial>(relaxed = true)
        Text(mult).logPosteriorPredictive(setOf("foo", "bar"), "baz", null)
        verify { mult.logPosteriorPredictive(setOf("foo", "bar"), Counter("baz"), null) }
    }

    @Test
    fun test_batchadd_delegates_parameters_to_multinomial() {
        val mult = mockk<Multinomial>(relaxed = true)
        Text(mult).batchUpdate(listOf("baz" to "foo"), Multinomial.Parameters(0.32, 0.23))
        verify { mult.batchUpdate(listOf("baz" to Counter("foo")), Multinomial.Parameters(0.32, 0.23)) }
    }

    @Test
    fun test_batchadd_delegates_null_parameters_to_multinomial() {
        val mult = mockk<Multinomial>(relaxed = true)
        Text(mult).batchUpdate(listOf("baz" to "foo"), null)
        verify { mult.batchUpdate(listOf("baz" to Counter("foo")), null) }
    }
}
