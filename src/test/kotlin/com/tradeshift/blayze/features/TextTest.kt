package com.tradeshift.blayze.features

import com.tradeshift.blayze.collection.Counter
import org.junit.Assert.assertEquals
import org.junit.Test

class TextTest {

    private val outcomes = setOf("p", "n")
    private val multinomial = Multinomial(pseudoCount = 1.0).batchUpdate(
            listOf(
                    "p" to Counter(mapOf("awesome" to 7, "terrible" to 3, "ok" to 19)),
                    "n" to Counter(mapOf("awesome" to 2, "terrible" to 13, "ok" to 21))
            )
            , null)

    @Test
    fun splits_text_and_delegates_to_multinomial() {
        val text = Text(multinomial)
        val expected = multinomial.logPosteriorPredictive(outcomes, Counter("awesome", "awesome", "awesome", "ok"), null)
        val actual = text.logPosteriorPredictive(outcomes, "awesome awesome awesome ok", null)

        assertEquals(expected, actual)
    }

    @Test
    fun stop_words_are_removed() {
        val text = Text(multinomial)
        val expected = text.logPosteriorPredictive(outcomes, "awesome awesome awesome ok", null)
        val actual = text.logPosteriorPredictive(outcomes, "the the a he awesome awesome awesome ok", null)

        assertEquals(expected, actual)
    }

    @Test
    fun special_chars_are_replaced_by_space() {
        val text = Text(multinomial)
        val expected = text.logPosteriorPredictive(outcomes, "awesome awesome ok", null)
        val actual = text.logPosteriorPredictive(outcomes, "awesome!;;\nawesome!!    \t !.,!ok", null)

        assertEquals(expected, actual)
    }

    @Test
    fun batch_update_sums_seen_words() {
        val mult = Multinomial(pseudoCount = 1.0).batchUpdate(listOf(
                "p" to Counter(mapOf("awesome" to 7, "ok" to 19, "terrible" to 3)),
                "n" to Counter(mapOf("awesome" to 2, "ok" to 21, "terrible" to 13))

        ), null)
        val expected = mult.logPosteriorPredictive(outcomes, Counter("awesome", "awesome", "awesome", "ok"), null)

        val text = Text(pseudoCount = 1.0)
                .batchUpdate(listOf(
                        "p" to "awesome awesome awesome awesome awesome awesome awesome",
                        "p" to "ok ok ok ok ok ok ok ok ok ok terrible terrible terrible ",
                        "n" to "awesome awesome ok ok ok ok ok ok ok ok ok ok ok ok ok ok ok"), null)
                .batchUpdate(listOf(
                        "p" to "ok ok ok ok ok ok ok ok ok",
                        "n" to "ok ok ok ok ok ok terrible terrible terrible terrible",
                        "n" to "terrible terrible terrible terrible terrible terrible terrible terrible terrible"), null)


        val actual = text.logPosteriorPredictive(outcomes, "awesome awesome awesome ok", null)

        assertEquals(expected, actual)
    }

}
