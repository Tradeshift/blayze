package com.tradeshift.blayze.features

import com.tradeshift.blayze.Protos
import com.tradeshift.blayze.dto.Outcome
import kotlin.math.*

/**
 * A feature for numbers that approximately follow a normal distribution, e.g. age, amounts, etc.
 */
class Gaussian(
        private val estimators: Map<Outcome, StreamingEstimator> = mapOf()
) : Feature<Double> {

    override fun toMutableFeature(): MutableGaussian {
        return MutableGaussian(estimators.toMutableMap())
    }

    override fun logProbability(outcomes: Set<Outcome>, value: Double): Map<Outcome, Double> {
        require(value.isFinite(), { "Value must be finite. It was $value" })
        val results = mutableMapOf<Outcome, Double>()
        for (outcome in outcomes) {
            results[outcome] = logPropabilityOutcome(outcome, value)
        }
        return results
    }

    private fun logPropabilityOutcome(outcome: Outcome, value: Double): Double {
        // p(x|mu,sigma)        = 1/sqrt(2*pi*sigma^2)              * exp(-(x-mu)^2/(2*sigma^2))
        // log(p(x|mu, sigma)   = log(1) - log(sqrt(2*pi*sigma^2))  - (x-mu)^2/(2*sigma^2)
        //                      = -log(sqrt(2*pi*sigma^2))          - (x-mu)^2/(2*sigma^2)
        //                      = -log(sigma*sqrt(2*pi))            - (x-mu)^2/(2*sigma^2)
        //                      = -log(sigma) - log(sqrt(2*pi))     - (x-mu)^2/(2*sigma^2)
        val (mu, sigma) = estimators[outcome] ?: return 0.0
        if (sigma == 0.0) {
            return 0.0
        }
        return -ln(sigma) - ln(sqrt(2 * PI)) - (value - mu).pow(2).div(2 * sigma.pow(2))
    }

    fun toProto(): Protos.Gaussian {
        return Protos.Gaussian.newBuilder().putAllEstimators(estimators.mapValues { it.value.toProto() }).build()
    }

    companion object {
        fun fromProto(proto: Protos.Gaussian): Gaussian {
            return Gaussian(proto.estimatorsMap.mapValues { StreamingEstimator.fromProto(it.value) })
        }
    }

}

class MutableGaussian(
        private val estimators: MutableMap<Outcome, StreamingEstimator> = mutableMapOf()
) : MutableFeature<Double> {

    override fun batchUpdate(updates: List<Pair<Outcome, Double>>) {
        val map = estimators
        for ((outcome, x) in updates) {
            map[outcome] = map[outcome]?.add(x) ?: StreamingEstimator(x)
        }
    }

    override fun toFeature(): Gaussian {
        return Gaussian(estimators)
    }

}

class StreamingEstimator private constructor(
        private val count: Int,
        val mean: Double,
        private val m2: Double
) {
    constructor(x: Double) : this(1, x, 0.0)

    /**
     * B. P. Welford (1962). "Note on a method for calculating corrected sums of squares and products".
     */
    fun add(x: Double): StreamingEstimator {
        require(x.isFinite(), { "x must be finite. It was $x" })
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

    operator fun component1(): Double {
        return mean
    }

    operator fun component2(): Double {
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