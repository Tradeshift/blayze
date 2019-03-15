package com.tradeshift.blayze.features

import org.junit.Assert.*
import org.junit.Test
import kotlin.math.*

class GaussianTest {

    @Test
    fun outcome_with_less_than_two_samples_returns_log_prob_of_least_likely_outcome() {
        val gauss = Gaussian().batchUpdate(
                listOf(
                        "a" to 25.0,
                        "a" to 30.0,
                        "a" to 35.0,
                        "b" to 10.0,
                        "b" to 15.0,
                        "b" to 20.0,
                        "c" to 90.0
                )
        )

        val predictive = gauss.logPosteriorPredictive(setOf("a", "b", "c", "d"), 20.0, null)

        assertTrue(predictive["b"]!! > predictive["a"]!!)
        assertEquals(predictive["a"]!!, predictive["c"]!!, 1e-6)
        assertEquals(predictive["a"]!!, predictive["d"]!!, 1e-6)
    }

    @Test
    fun gaussian_features_are_invariant_to_scaling_and_shifting() {
        val expected = 2.4425467832421157
        for (scale in listOf(1.0, 0.1, 10.0, -1.0)) {
            for (shift in listOf(0.0, -10.0, 20.0)) {
                val model = Gaussian().batchUpdate(listOf(
                        "t-shirt" to scale * 10.0 + shift,
                        "t-shirt" to scale * 19.0 + shift,
                        "sweater" to scale * -10.0 + shift,
                        "sweater" to scale * -20.0 + shift
                ))
                val actual = model.logPosteriorPredictive(setOf("t-shirt", "sweater"), scale * 8.0 + shift, null)

                assertEquals(expected, actual["t-shirt"]!! - actual["sweater"]!!, 1e-6)
            }
        }
    }

    @Test
    fun if_no_outcomes_have_meaningful_log_probabilities_returns_zero_for_all_outcomes() {
        val gauss = Gaussian().batchUpdate(
                listOf(
                        "a" to 25.0,
                        "b" to 20.0,
                        "b" to 20.0
                )
        )
        val predictive = gauss.logPosteriorPredictive(setOf("a", "b", "c"), 20.0, null)

        assertEquals(0.0, predictive["a"]!!, 1e-6)
        assertEquals(0.0, predictive["b"]!!, 1e-6)
        assertEquals(0.0, predictive["c"]!!, 1e-6)
    }

    @Test
    fun zero_variance_outcomes_gives_log_prob_of_least_likely_outcome() {
        val gauss = Gaussian().batchUpdate(
                listOf(
                        "a" to 25.0,
                        "a" to 30.0,
                        "a" to 35.0,
                        "b" to 10.0,
                        "b" to 15.0,
                        "b" to 20.0,
                        "c" to 90.0,
                        "c" to 90.0,
                        "c" to 90.0
                )
        )
        val predictive = gauss.logPosteriorPredictive(setOf("a", "b", "c"), 20.0, null)

        assertTrue(predictive["b"]!! > predictive["a"]!!)
        assertEquals(predictive["a"]!!, predictive["c"]!!, 1e-6)
    }

    @Test
    fun posterior_predictive_is_t_distribution() {
        val gauss = Gaussian().batchUpdate(
                listOf(
                        "p" to 20.0,
                        "p" to 30.0,
                        "p" to 40.0
                )
        )

        /*
            import numpy as np
            import scipy
            d = np.array([20, 30, 40])
            n = 3.0
            scipy.stats.t.pdf(23.0, n, loc=d.mean(), scale=np.sqrt(d.var()*(n+1)/n))
        */
        val expected = 0.02782119452355812

        val x = 23.0
        val actual = Math.exp(gauss.logPosteriorPredictive(setOf("p"), x, null)["p"]!!)

        assertEquals(expected, actual, 1e-6)
    }

    @Test
    fun approaches_mle_gaussian_with_enough_samples() {
        val numbers = (1..1000).flatMap { listOf(2.0, 6.0) }

        val gaussian = Gaussian().batchUpdate(
                numbers.map { "p" to it }
        )

        val mean = numbers.sum() / numbers.size
        val variance = numbers.map { (it - mean).pow(2) }.sum() / (numbers.size - 1)

        val std = sqrt(variance)
        for (i in (-200..200)) {
            val x = mean + i / 100.0 * std // mean +- 2stddev
            val expected = ln(1.0 / sqrt(2 * PI * variance) * exp(-(x - mean).pow(2) / (2 * variance)))
            assertEquals(expected, gaussian.logPosteriorPredictive(setOf("p"), x, null)["p"]!!, 0.001)
        }
    }

    @Test
    fun posterior_predictive_with_prior_parameters_is_t_distribution() {
        val gauss = Gaussian().batchUpdate(
                listOf(
                        "p" to 20.0,
                        "p" to 30.0,
                        "p" to 40.0
                )
        )

        //From https://en.wikipedia.org/wiki/Conjugate_prior#When_likelihood_function_is_a_continuous_distribution
        /*
            import numpy as np
            import scipy
            d = np.array([20, 30, 40])
            n = 3.0
            mu0 = 28.0
            nu = 2.0
            alpha = 1.0
            beta = 120.0

            mup = (nu*mu0 +n*d.mean())/(nu+n)
            nup = nu+n
            alphap = alpha + n/2.0
            betap = beta + 0.5*np.sum((d-d.mean())**2) + (n*nu)/(nu+n)*(d.mean()-mu0)**2/2.0

            scipy.stats.t.pdf(23.0, 2*alphap, loc=mup, scale=np.sqrt(betap*(nup+1)/(nup*alphap)))

        */
        val expected = 0.029822246851240995

        val x = 23.0
        val actual = Math.exp(gauss.logPosteriorPredictive(setOf("p"), x, Gaussian.Parameters(mu0 = 28.0, nu = 2, beta = 120.0, alpha = 1))["p"]!!)

        assertEquals(expected, actual, 1e-6)
    }

}
