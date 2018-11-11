package com.tradeshift.blayze.features

import com.tradeshift.blayze.collection.Counter
import org.junit.Assert.assertEquals
import org.junit.Test

class CategoricalTest {

    private val multinomial = Multinomial(pseudoCount = 1.0).batchUpdate(
            listOf(
                    "p" to Counter("ole"),
                    "n" to Counter("ole", "bob", "ada")
            )
    )

    @Test
    fun return_right_log_probability() {
        val categorical = Categorical(multinomial)
        val P_p = categorical.logPosteriorPredictive(setOf("p"), "ole")["p"]!!
        val P_n = categorical.logPosteriorPredictive(setOf("n"), "ole")["n"]!!

        /*
        posterior P(ole | p) = (1 + 1) / (1 + 3) = 1 / 2
        posterior P(ole | n) = (1 + 1) / (3 + 3) = 1 / 3
        */

        assertEquals(1 / 2.0, Math.exp(P_p), 0.0000001)
        assertEquals(1 / 3.0, Math.exp(P_n), 0.0000001)
    }

    @Test
    fun test_batch_update() {
        val categorical = Categorical(pseudoCount = 1.0)
                .batchUpdate(listOf("p" to "ole", "n" to "ole"))
                .batchUpdate(listOf("n" to "bob", "n" to "ada"))

        val pP = categorical.logPosteriorPredictive(setOf("p"), "ole")["p"]!!
        val pN = categorical.logPosteriorPredictive(setOf("n"), "ole")["n"]!!

        assertEquals(1 / 2.0, Math.exp(pP), 0.0000001)
        assertEquals(1 / 3.0, Math.exp(pN), 0.0000001)
    }

}
