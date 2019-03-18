package com.tradeshift.blayze.features

import com.tradeshift.blayze.Protos
import com.tradeshift.blayze.dto.Outcome
import com.tradeshift.blayze.logStudentT
import kotlin.math.*

/**
 * A feature for numbers that approximately follow a normal distribution, e.g. age, amounts, etc.
 *
 * Uses a Normal-inverse gamma prior on the parameters of the gaussian. See https://en.wikipedia.org/wiki/Conjugate_prior#Continuous_distributions
 * and https://en.wikipedia.org/wiki/Student%27s_t-distribution#Generalized_Student's_t-distribution
 *
 * Note, after much deliberation we've set the four default prior hyper-parameters to zero. For outcomes with zero variance or less than two samples we return
 * the minimum log probability of the other outcomes. If all the outcomes are undefined (e.g. first time you add a gaussian feature), the feature returns 0 for
 * all outcomes, effectively ignoring the feature. Yes this is a giant hack, but it works, and after 2 days of fiddling with the hyper-parameters it became
 * apparent that there'd always be a counter argument against a certain set of hyper priors.
 *
 * The current solution satisfies:
 *  - The gaussian feature is scale and shift invariant, which is very nice (adding priors break this)
 *  - Adding new outcomes doesn't break the classifier (with priors, new outcomes might be more or less likely than previously observed outcomes with properly estimated distributions, which can quickly overrule other features)
 *  - Adding new gaussian features doesn't break the classifier (same as above)
 */
class Gaussian private constructor(
        val defaultParameters: Gaussian.Parameters,
        private val estimators: Map<Outcome, StreamingEstimator>
) : Feature<Gaussian, Double, Gaussian.Parameters> {

    /**
     * @param mu0 Prior mean
     * @param nu Prior mean was estimated from nu observations
     * @param beta 1/2 prior sum of squared deviations
     * @param alpha Prior variance was estimated from 2·alpha observations with sample mean mu0 and sum of squared deviations 2·beta
     *
     * See https://en.wikipedia.org/wiki/Conjugate_prior#Continuous_distributions and the docs of [Gaussian]
     *
     * These parameters are only used in [logPosteriorPredictive], not [batchUpdate].
     *
     * Unless you really know what you're doing, you probably shouldn't change these from the default values.
     */
    data class Parameters(
            val mu0: Double = 0.0,
            val nu: Int = 0,
            val beta: Double = 0.0,
            val alpha: Int = 0
    )

    /**
     * Create a new gaussian feature.
     * Unless you really know what you're doing, you should probably use the default parameters.
     *
     * See [Parameters]
     */
    constructor(parameters: Parameters = Parameters()) : this(parameters, mapOf())


    override fun withParameters(parameters: Parameters): Gaussian {
        return Gaussian(parameters, estimators)
    }

    /**
     * See [Feature.batchUpdate]
     *
     * Parameters are not used.
     */
    override fun batchUpdate(updates: List<Pair<Outcome, Double>>, parameters: Parameters?): Gaussian {
        val map = estimators.toMutableMap()
        for ((outcome, x) in updates) {
            map[outcome] = map[outcome]?.add(x) ?: StreamingEstimator(x)
        }
        return Gaussian(defaultParameters, map)
    }

    override fun logPosteriorPredictive(outcomes: Set<Outcome>, value: Double, parameters: Parameters?): Map<Outcome, Double> {
        val results = mutableMapOf<Outcome, Double?>()
        for (outcome in outcomes) {
            results[outcome] = logPropabilityOutcome(outcome, value, parameters)
        }

        // Giant hack
        val min = results.values.filterNotNull().min() ?: 0.0
        return results.map { (k, v) -> k to (v ?: min) }.toMap()
    }

    /**
     * See https://en.wikipedia.org/wiki/Conjugate_prior#Continuous_distributions
     */
    private fun logPropabilityOutcome(outcome: Outcome, value: Double, parameters: Parameters?): Double? {
        val p = parameters ?: defaultParameters

        val est = estimators[outcome]
        val n = (est?.count ?: 0).toDouble()
        val mu = est?.mean ?: 0.0
        val sigma = est?.stdev ?: 0.0

        if (n < 2 || sigma == 0.0) { // Hack
            return null
        }

        val pmu = (p.nu * p.mu0 + n * mu) / (p.nu + n)
        val pnu = (p.nu + n)
        val palpha = (p.alpha + n / 2.0)
        val ss = sigma.pow(2.0) * (n - 1) // n-1 here, since the way std is calculated
        val pbeta = (p.beta + (1.0 / 2.0) * ss + n * p.nu / (n + p.nu) * (mu - p.mu0).pow(2.0) / 2.0)

        return logStudentT(value, 2 * palpha, pmu, sqrt(pbeta * (pnu + 1) / (palpha * pnu)))
    }

    /**
     * B. P. Welford (1962). "Note on a method for calculating corrected sums of squares and products".
     */
    class StreamingEstimator private constructor(
            val count: Int,
            val mean: Double,
            private val m2: Double
    ) {
        constructor(x: Double) : this(1, x, 0.0)

        fun add(x: Double): StreamingEstimator {
            var (count, mean, m2) = Triple(count, mean, m2)
            count += 1
            val delta = x - mean
            mean += delta / count
            val delta2 = x - mean
            m2 += delta * delta2

            return StreamingEstimator(count, mean, m2)
        }

        val stdev: Double by lazy {
            if (count < 2) {
                0.0
            } else {
                sqrt(m2 / (count - 1))
            }
        }

        operator fun component1(): Int {
            return count
        }

        operator fun component2(): Double {
            return mean
        }

        operator fun component3(): Double {
            return stdev
        }

        fun toProto(): Protos.StreamingEstimator {
            return Protos.StreamingEstimator.newBuilder()
                    .setCount(count)
                    .setMean(mean)
                    .setM2(m2)
                    .build()
        }

        companion object {
            fun fromProto(proto: Protos.StreamingEstimator): StreamingEstimator {
                return StreamingEstimator(proto.count, proto.mean, proto.m2)
            }
        }
    }

    fun toProto(): Protos.Gaussian {
        return Protos.Gaussian.newBuilder()
                .setMu0(defaultParameters.mu0)
                .setNu(defaultParameters.nu)
                .setBeta(defaultParameters.beta)
                .setAlpha(defaultParameters.alpha)
                .putAllEstimators(estimators.mapValues { it.value.toProto() })
                .build()
    }

    companion object {
        fun fromProto(proto: Protos.Gaussian): Gaussian {
            return Gaussian(
                    Parameters(proto.mu0, proto.nu, proto.beta, proto.alpha),
                    proto.estimatorsMap.mapValues { StreamingEstimator.fromProto(it.value) }
            )
        }
    }

}
