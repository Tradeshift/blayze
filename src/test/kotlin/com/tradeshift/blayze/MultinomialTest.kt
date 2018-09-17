package com.tradeshift.blayze

import com.tradeshift.blayze.collection.Counter
import com.tradeshift.blayze.features.Multinomial
import com.tradeshift.blayze.features.MutableMultinomial
import org.junit.Assert.assertEquals
import org.junit.Test

class MultinomialTest {

    @Test
    fun test_empty_multinomial() {
        val empty = Multinomial(1.0, 1.0)
        val logProb = empty.logProbability(setOf("p", "n"), Counter("a", "b", "c"))
        assertEquals(0.0, logProb["p"])
        assertEquals(0.0, logProb["n"])
    }

    @Test
    fun test_logprob() {
        val logProb = multinomial.logProbability(setOf("p", "n"), Counter("a", "b"))

        assertEquals(2, logProb.size)
        // log(1 + 1) - log(2 + 3 * 1) + log(1 + 1) - log(2 + 3 * 1)
        assertEquals(-1.832, logProb["p"]!!, 0.001)
        // log(0 + 1) - log(2 + 3 * 1) + log(1 + 1) - log(2 + 3 * 1)
        assertEquals(-2.525, logProb["n"]!!, 0.001)
    }

    @Test
    fun request_outcome_subset() {
        val logProb = multinomial.logProbability(setOf("p"), Counter("a", "b"))

        assertEquals(1, logProb.size)
        // log(1 + 1) - log(2 + 3 * 1) + log(1 + 1) - log(2 + 3 * 1)
        assertEquals(-1.832, logProb["p"]!!, 0.001)
    }

    @Test
    fun request_outcome_intersection() {
        val logProb = multinomial.logProbability(setOf("p", "x"), Counter("a", "b"))

        assertEquals(2, logProb.size)
        // log(1 + 1) - log(2 + 3 * 1) + log(1 + 1) - log(2 + 3 * 1)
        assertEquals(-1.832, logProb["p"]!!, 0.001)
        // log(0 + 1) - log(0 + 3 * 1) + log(0 + 1) - log(0 + 3 * 1)
        assertEquals(-2.197, logProb["x"]!!, 0.001)
    }

    private val multinomial: Multinomial
        get() {
            val mutable = MutableMultinomial(1.0, 1.0)
            mutable.batchUpdate(
                    listOf(
                            "p" to Counter("a", "b"),
                            "n" to Counter("b", "c")
                    ))
            return mutable.toFeature()
        }

}