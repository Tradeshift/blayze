package com.tradeshift.blayze.features

import com.tradeshift.blayze.collection.tableOf
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.pow

class TextTest {

    private val multinomial = Multinomial(
            countTable = tableOf(
                    "p" to "awesome" to 7,
                    "p" to "terrible" to 3,
                    "p" to "ok" to 19,
                    "n" to "awesome" to 2,
                    "n" to "terrible" to 13,
                    "n" to "ok" to 21
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

    @Test
    fun test_wordcounter_bigram() {
        val result = Text.WordCounter.countWords("aa bb cc dd", useBigram = true)
        assertEquals(setOf("aa", "bb", "cc", "dd", "aa bb", "bb cc", "cc dd"), result.keys)
    }

    @Test
    fun test_text_created_from_old_text_keep_useBigram_setting() {
        val text = Text(multinomial, useBigram = true)
        val toProto = text.batchUpdate(listOf()).toProto()  // verify the return of batchUpdate() has used the old useBigram setting
        assertTrue(toProto.useBigram)
        assertTrue(Text.fromProto(toProto).toProto().useBigram) // verify toProto and fromProto has used the old useBigram setting
    }
}
