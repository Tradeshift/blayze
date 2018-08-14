package com.tradeshift.blayze.features

import com.tradeshift.blayze.Protos
import com.tradeshift.blayze.collection.Counter
import com.tradeshift.blayze.collection.SparseIntVector
import com.tradeshift.blayze.dto.Outcome
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.pow

class Multinomial private constructor(
        private val includeFeatureProbability: Double = 1.0,
        private val pseudoCount: Double = 1.0,
        private val outcomeIndices: Map<Outcome, Int>,
        private val features: Map<String, SparseIntVector>
) : Feature<Multinomial, Counter<String>> {

    /**
     * A feature for multinomial data.
     *
     * @property includeFeatureProbability Include new features with this probability. See Ad Click Prediction: a View from the Trenches, Table 2
     * @property pseudoCount Add this number to all counts, even zero counts. Prevents 0 probability. See http://en.wikipedia.org/wiki/Naive_Bayes_classifier#Multinomial_naive_Bayes
     */
    constructor(includeFeatureProbability: Double = 1.0, pseudoCount: Double = 1.0) :
            this(includeFeatureProbability, pseudoCount, HashMap(), HashMap())

    /**
     * The number of features seen for each outcome
     */
    private val nFeaturesPerOutcome: IntArray by lazy {
        val res = IntArray(outcomeIndices.size)
        for (vec in features.values) {
            for ((idx, count) in vec) {
                res[idx] += count
            }
        }
        res
    }

    override fun batchUpdate(updates: List<Pair<Outcome, Counter<String>>>): Multinomial {
        val featuresCopy = features.toMutableMap()
        val outcomesCopy = outcomeIndices.toMutableMap()

        fun getOrCreateIndex(key: Outcome) = outcomesCopy.getOrPut(key, { outcomesCopy.size })

        val formattedUpdates = invertUpdates(sampleUpdates(updates.asSequence()))
        for ((feature, counter) in formattedUpdates) {
            val indexToUpdate = counter.mapKeys { getOrCreateIndex(it.key)  }
            val vec = SparseIntVector.fromMap(indexToUpdate)
            featuresCopy[feature] = featuresCopy[feature]?.add(vec) ?: vec
        }
        return Multinomial(includeFeatureProbability, pseudoCount, outcomesCopy, featuresCopy)
    }

    /**
     * The unnormalized log probability for each outcome, which is defined as:
     *
     *      logProb_o = sum_f(log(S_of + pseudoCount) - log(sum_i(S_oi + pseudoCount))
     *
     * where sum_f is a sum over all input features, S_of is the number of times feature f has been seen with outcome o,
     * and sum_i is a sum over all seen features.
     *
     * In the case where the count matrix S_cf is sparse this computation will be dominated by zero values. To retain
     * performance in this case this implementation reduces the number of computations required by separating
     * the calculation of zero-valued entries of S and non-zero entries:
     *
     *      logProb_o = sum_fnonzero((log(S_of + pseudoCount) - log(sum_i(S_oi + pseudoCount))
     *                  + sum_fzero((log(0 + pseudoCount) - log(sum_i(S_oi + pseudoCount))
     *
     *                = sum_fnonzero((log(S_of + pseudoCount) - log(sum_i(S_oi + pseudoCount))
     *                  + sum_fzero(1) * (log(S_of + pseudoCount) - log(sum_i(S_oi + pseudoCount))
     *
     * The second term only needs to be computed once instead of summing over all input features.
     */
    override fun logProbability(outcomes: Set<Outcome>, value: Counter<String>): Map<Outcome, Double> {
        // sum all log probabilities for non-zero feature-outcome combinations
        val nonZeroLogProbs = DoubleArray(outcomeIndices.size)
        val nonZeroCounts = IntArray(outcomeIndices.size)
        var nFeatures = 0
        for ((feature, count) in value) {
            val vector = features[feature]
            if (vector != null) {
                nFeatures += count
                for ((idx, v) in vector) {
                    nonZeroLogProbs[idx] += count * logProbability(v, nFeaturesPerOutcome[idx])
                    nonZeroCounts[idx] += count
                }
            }
        }

        // then compute log probabilities for zero valued feature-outcome combinations
        val zeroLogProbs = nonZeroCounts.mapIndexed { idx, nonZeroCount ->
            val zeroCount = nFeatures - nonZeroCount
            zeroCount * logProbability(0, nFeaturesPerOutcome[idx])
        }

        return outcomes.map { outcome ->
            val idx = outcomeIndices[outcome]
            val logProb = if (idx != null)
                nonZeroLogProbs[idx] + zeroLogProbs[idx]
            else {
                // unseen outcomeIndices have zero counts
                nFeatures * logProbability(0, 0)
            }
            outcome to logProb
        }.toMap()
    }

    private fun logProbability(numerator: Int, denominator: Int): Double {
        return ln(numerator + pseudoCount) - ln(denominator + max(features.size, 1) * pseudoCount)
    }

    private fun sampleUpdates(updates: Sequence<Pair<Outcome, Counter<String>>>): Sequence<Pair<Outcome, Counter<String>>> {
        fun sampleFeature(count: Int) = Math.random() < (1.0 - (1.0 - includeFeatureProbability).pow(count))
        val knownFeatures = features.keys.toMutableSet()
        return updates.map { (outcome, counter) ->
            val filteredFeatures = counter.filter { it.key in knownFeatures || sampleFeature(it.value) }
            knownFeatures.addAll(filteredFeatures.keys)
            outcome to Counter(filteredFeatures)
        }
    }

    private fun invertUpdates(updates: Sequence<Pair<Outcome, Counter<String>>>): Map<String, Counter<Outcome>> {
        val flipped = HashMap<String, HashMap<Outcome, Int>>()
        for ((outcome, counter) in updates) {
            for ((feature, count) in counter) {
                val outcomeCounter = flipped.getOrPut(feature, { HashMap() })
                outcomeCounter[outcome] = count + (outcomeCounter[outcome] ?: 0)
            }
        }
        return flipped.mapValues { Counter(it.value) }
    }

    fun toProto(): Protos.Multinomial = Protos.Multinomial.newBuilder()
            .setIncludeFeatureProbability(includeFeatureProbability)
            .setPseudoCount(pseudoCount)
            .putAllOutcomes(outcomeIndices)
            .putAllFeatures(features.mapValues { it.value.toProto() })
            .build()

    companion object {
        fun fromProto(proto: Protos.Multinomial) = Multinomial(
                proto.includeFeatureProbability,
                proto.pseudoCount,
                proto.outcomesMap,
                proto.featuresMap.mapValues { SparseIntVector.fromProto(it.value) })
    }
}