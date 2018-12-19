package com.tradeshift.blayze.features

import com.tradeshift.blayze.collection.Counter
import org.junit.Assert.assertEquals
import org.junit.Test


class MultinomialTest {

    @Test
    fun test_empty_multinomial() {
        val empty = Multinomial(1.0, 1.0)
        val logProb = empty.logPosteriorPredictive(setOf("p", "n"), Counter("a", "b", "c"))
        assertEquals(0.0, logProb["p"])
        assertEquals(0.0, logProb["n"])
    }

    @Test
    fun test_unseen_words_are_ignored() {
        val f = Multinomial(1.0, 1.0).batchUpdate(listOf(
                "p" to Counter("foo", "foo", "bar"),
                "n" to Counter("foo", "bar", "baz")
        ))
        val lp1 = f.logPosteriorPredictive(setOf("p", "n"), Counter("foo", "bar", "baz"))
        val lp2 = f.logPosteriorPredictive(setOf("p", "n"), Counter("foo", "bar", "baz", "zap"))
        assertEquals(lp1["p"]!!, lp2["p"]!!, 0.0)
        assertEquals(lp1["n"]!!, lp2["n"]!!, 0.0)
    }

    @Test
    fun test_log_posterior_predictive_approaches_mle_multinomial_as_samples_approaches_infinity() {
        val n = 1000
        val updates = (0..n).flatMap {
            listOf(
                    "p" to Counter("a", "b"),
                    "n" to Counter("b", "c")
            )
        }
        val pseudoCount = 1.0
        val multinomial = Multinomial(1.0, pseudoCount).batchUpdate(updates)
        val actual = multinomial.logPosteriorPredictive(setOf("p", "n"), Counter("a", "b"))

        assertEquals(2, actual.size)

        val e1 = Math.log(n + pseudoCount) - Math.log(2 * n + 3 * pseudoCount) + Math.log(n + pseudoCount) - Math.log(2 * n + 3 * pseudoCount)
        val e2 = Math.log(pseudoCount) - Math.log(2 * n + 3 * pseudoCount) + Math.log(n + pseudoCount) - Math.log(2 * n + 3 * pseudoCount)

        assertEquals(e1 - e2, actual["p"]!! - actual["n"]!!, 0.001)
    }

    @Test
    fun test_log_posterior_predictive_is_dirichlet_multinomial() {
        /*
            import numpy as np
            from tensorflow.distributions import DirichletMultinomial
            import tensorflow as tf
            pseudoCount = 1.0

            counts = np.array([[1., 1., 0.], [0., 1., 1.], [0., 0., .0]], dtype=np.float32)
            alpha = counts+pseudoCount
            n = np.float32([2.0, 2.0, 2.0]) # number of draws, all 2 since np.sum(x) == 2
            dm = DirichletMultinomial(n, alpha)

            x = np.array([1, 1, 0])

            with tf.Session() as sess:
                print(sess.run(dm.log_prob(x)))

            > [-1.3217559 -2.014903  -1.7917595]
        */
        val multinomial = Multinomial(1.0, 1.0).batchUpdate(listOf(
                "p" to Counter("a", "b"),
                "n" to Counter("b", "c")
        ))
        val logProb = multinomial.logPosteriorPredictive(setOf("p", "n", "x"), Counter("a", "b"))

        assertEquals(-1.3217559, logProb["p"]!!, 1e-6)
        assertEquals(-2.014903, logProb["n"]!!, 1e-6)
        assertEquals(-1.7917595, logProb["x"]!!, 1e-6)
    }

}