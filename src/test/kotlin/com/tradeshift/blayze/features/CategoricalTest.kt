package com.tradeshift.blayze.features

import com.tradeshift.blayze.collection.Counter
import io.mockk.mockk
import io.mockk.verify
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

    @Test
    fun test_withParameters_sets_parameters() {
        val parameters = Multinomial.Parameters(0.0892275415, 0.7363151583)
        val categorical = Categorical().withParameters(parameters)

        assertEquals(categorical.delegate.defaultParams, parameters)
    }
    
    @Test
    fun test_logPosteriorPredictive_delegates_parameters_to_multinomial() {
        val mult = mockk<Multinomial>(relaxed = true)
        Categorical(mult).logPosteriorPredictive(setOf("foo", "bar"), "baz", Multinomial.Parameters(0.32, 0.23))
        verify { mult.logPosteriorPredictive(setOf("foo", "bar"), Counter("baz"), Multinomial.Parameters(0.32, 0.23)) }
    }

    @Test
    fun test_logPosteriorPredictive_delegates_null_parameters_to_multinomial() {
        val mult = mockk<Multinomial>(relaxed = true)
        Categorical(mult).logPosteriorPredictive(setOf("foo", "bar"), "baz", null)
        verify { mult.logPosteriorPredictive(setOf("foo", "bar"), Counter("baz"), null) }
    }

    @Test
    fun test_batchadd_delegates_parameters_to_multinomial() {
        val mult = mockk<Multinomial>(relaxed = true)
        Categorical(mult).batchUpdate(listOf("baz" to "foo"), Multinomial.Parameters(0.32, 0.23))
        verify { mult.batchUpdate(listOf("baz" to Counter("foo")), Multinomial.Parameters(0.32, 0.23)) }
    }

    @Test
    fun test_batchadd_delegates_null_parameters_to_multinomial() {
        val mult = mockk<Multinomial>(relaxed = true)
        Categorical(mult).batchUpdate(listOf("baz" to "foo"), null)
        verify { mult.batchUpdate(listOf("baz" to Counter("foo")), null) }
    }

}
