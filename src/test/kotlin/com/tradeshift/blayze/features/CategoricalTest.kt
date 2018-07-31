package com.tradeshift.blayze.features

import com.tradeshift.blayze.collection.Counter
import org.junit.Assert.assertEquals
import org.junit.Test

class CategoricalTest {

    private val multinomial = Multinomial().batchUpdate(
            listOf(
                    "p" to Counter("ole"),
                    "n" to Counter("ole", "bob", "ada")
            )
    )

    @Test
    fun return_right_log_probability() {
        val categorical = Categorical(multinomial)
        val P_p = categorical.logProbability(setOf("p"), "ole")["p"]!!
        val P_n = categorical.logProbability(setOf("n"), "ole")["n"]!!

        /*
        posterior P(ole | p) = (1 + 1) / (1 + 3) = 1 / 2
        posterior P(ole | n) = (1 + 1) / (3 + 3) = 1 / 3
        */

        assertEquals(1 / 2.0, Math.exp(P_p), 0.0000001)
        assertEquals(1 / 3.0, Math.exp(P_n), 0.0000001)
    }

    @Test
    fun test_batch_update() {
        val categorical = Categorical()
                .batchUpdate(listOf("p" to "ole", "n" to "ole"))
                .batchUpdate(listOf("n" to "bob", "n" to "ada"))

        val pP = categorical.logProbability(setOf("p"), "ole")["p"]!!
        val pN = categorical.logProbability(setOf("n"), "ole")["n"]!!

        assertEquals(1 / 2.0, Math.exp(pP), 0.0000001)
        assertEquals(1 / 3.0, Math.exp(pN), 0.0000001)
    }

    @Test
    fun unseen_feature_gives_zero_logprobability() {
        val categorical = Categorical(multinomial)
        val pP = categorical.logProbability(setOf("p"), "notseen")["p"]!!
        val pN = categorical.logProbability(setOf("n"), "notseen")["n"]!!

        assertEquals(0.0, pP, 0.0000001)
        assertEquals(0.0, pN, 0.0000001)
    }
}
