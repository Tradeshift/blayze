package com.tradeshift.blayze.features

import com.tradeshift.blayze.collection.Counter
import org.junit.Assert.assertEquals
import org.junit.Test
import kotlin.math.pow

class TextTest {

    private val multinomial = Multinomial().batchUpdate(
            listOf(
                    "p" to Counter(mapOf("awesome" to 7, "terrible" to 3, "ok" to 19)),
                    "n" to Counter(mapOf("awesome" to 2, "terrible" to 13, "ok" to 21))
            )
    )

    @Test
    fun return_right_log_proberbility() {
        val text = Text(multinomial)
        val pP = text.logProbability(setOf("p"), "awesome awesome awesome ok")["p"]!!
        val pN = text.logProbability(setOf("n"), "awesome awesome awesome ok")["n"]!!

        /*
         P(awesome | p) = 7/29, P(ok | p) = 19/29, P(awesome awesome awesome ok | p) = ((7+1)/(29+3))^3 * (19+1)/(29+3)
         P(awesome | n) = 2/36, P(ok | p) = 21/36, P(awesome awesome awesome ok | p) = ((2+1)/(36+3))^3 * (21+1)/(36+3)
         */
        assertEquals((8 / 32.0).pow(3) * 20 / 32.0, Math.exp(pP), 0.0000001)
        assertEquals((3 / 39.0).pow(3) * 22 / 39.0, Math.exp(pN), 0.0000001)
    }

    @Test
    fun test_batch_update() {
        val text = Text(Multinomial())
                .batchUpdate(listOf(
                        "p" to "awesome awesome awesome awesome awesome awesome awesome",
                        "p" to "ok ok ok ok ok ok ok ok ok ok terrible terrible terrible ",
                        "n" to "awesome awesome ok ok ok ok ok ok ok ok ok ok ok ok ok ok ok"))
                .batchUpdate(listOf(
                        "p" to "ok ok ok ok ok ok ok ok ok",
                        "n" to "ok ok ok ok ok ok terrible terrible terrible terrible",
                        "n" to "terrible terrible terrible terrible terrible terrible terrible terrible terrible"))

        val pP = text.logProbability(setOf("p"), "awesome awesome awesome ok")["p"]!!
        val pN = text.logProbability(setOf("n"), "awesome awesome awesome ok")["n"]!!

        assertEquals((8 / 32.0).pow(3) * 20 / 32.0, Math.exp(pP), 0.0000001)
        assertEquals((3 / 39.0).pow(3) * 22 / 39.0, Math.exp(pN), 0.0000001)
    }

    @Test
    fun return_zero_if_not_seen() {
        val text = Text(multinomial)
        val pP = text.logProbability(setOf("p"), "notseen")["p"]!!
        val pN = text.logProbability(setOf("n"), "notseen")["n"]!!

        assertEquals(0.0, pP, 0.0000001)
        assertEquals(0.0, pN, 0.0000001)
    }
}
