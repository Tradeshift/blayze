package com.tradeshift.blayze.features

import org.junit.Assert.assertEquals
import org.junit.Test
import kotlin.math.*


class GaussianTest {

    @Test
    fun given_values_computes_gaussian_probabilities_with_maximum_likelihood_estimator() {
        val numbers = (1..100).map { it.toDouble() }

        var se = StreamingEstimator(1.0)
        for (number in numbers.drop(1)) {
            se = se.add(number)
        }
        val gaussian = Gaussian(mapOf("p" to se))

        val mean = numbers.sum() / numbers.size
        val variance = numbers.map { (it - mean).pow(2) }.sum() / (numbers.size - 1)

        val x = 45.12
        val expected = ln(1.0 / sqrt(2 * PI * variance) * exp(-(x - mean).pow(2) / (2 * variance)))
        assertEquals(expected, gaussian.logProbability(setOf("p"), x)["p"]!!, 0.000001)
    }

    @Test
    fun given_unknown_outcome_return_zero() {
        val gaussian = Gaussian(mapOf("p" to StreamingEstimator(3.56)))
        assertEquals(mapOf("foo" to 0.0), gaussian.logProbability(setOf("foo"), 123.21))
    }

    @Test
    fun given_single_value_estimator_return_zero() {
        val gaussian = Gaussian(mapOf("p" to StreamingEstimator(3.56)))
        assertEquals(mapOf("p" to 0.0), gaussian.logProbability(setOf("p"), 123.21))
    }

    @Test
    fun given_batches_updates_correctly() {
        val mutable = MutableGaussian()
        mutable
                .batchUpdate(listOf(
                        "p" to 1.0,
                        "p" to 2.0,
                        "n" to 3.0
                ))
        mutable.batchUpdate(listOf(
                "p" to 3.0,
                "n" to 2.0
        ))
        val actual = mutable.toFeature()

        val expected = Gaussian(mapOf(
                "p" to StreamingEstimator(1.0).add(2.0).add(3.0),
                "n" to StreamingEstimator(3.0).add(2.0)
        ))

        val x = 2.4212
        assertEquals(expected.logProbability(setOf("p"), x), actual.logProbability(setOf("p"), x))
        assertEquals(expected.logProbability(setOf("n"), x), actual.logProbability(setOf("n"), x))
    }
}
