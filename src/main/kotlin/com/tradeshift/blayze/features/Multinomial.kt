package com.tradeshift.blayze.features

import com.tradeshift.blayze.Protos
import com.tradeshift.blayze.collection.Counter
import com.tradeshift.blayze.collection.SparseIntVector
import com.tradeshift.blayze.dto.Outcome
import com.tradeshift.blayze.logBeta
import java.lang.Math.log
import kotlin.math.pow

class Multinomial private constructor(
        private val includeFeatureProbability: Double = 1.0,
        private val pseudoCount: Double = 0.1,
        private val outcomeToIdx: Map<Outcome, Int>,
        private val features: Map<String, SparseIntVector>
) : Feature<Multinomial, Counter<String>> {

    /**
     * A feature for multinomial data.
     *
     * @property includeFeatureProbability Include new features with this probability. See Ad Click Prediction: a View from the Trenches, Table 2
     * @property pseudoCount Effectively adds this number to all counts, even zero counts.
     */
    constructor(includeFeatureProbability: Double = 1.0, pseudoCount: Double = 0.1) :
            this(includeFeatureProbability, pseudoCount, HashMap(), HashMap())

    /**
     * The number of features seen for each outcome
     */
    private val nFeaturesPerOutcome: IntArray by lazy {
        val res = IntArray(outcomeToIdx.size)
        for (vec in features.values) {
            for ((idx, count) in vec) {
                res[idx] += count
            }
        }
        res
    }

    private val idxToOutcome: Map<Int, Outcome> by lazy {
        outcomeToIdx.map { it.value to it.key }.toMap()
    }

    override fun batchUpdate(updates: List<Pair<Outcome, Counter<String>>>): Multinomial {
        val featuresCopy = features.toMutableMap()
        val outcomesCopy = outcomeToIdx.toMutableMap()

        fun getOrCreateIndex(key: Outcome) = outcomesCopy.getOrPut(key) { outcomesCopy.size }

        val formattedUpdates = invertUpdates(sampleUpdates(updates.asSequence()))
        for ((feature, counter) in formattedUpdates) {
            val indexToUpdate = counter.mapKeys { getOrCreateIndex(it.key) }
            val vec = SparseIntVector.fromMap(indexToUpdate)
            featuresCopy[feature] = featuresCopy[feature]?.add(vec) ?: vec
        }
        return Multinomial(includeFeatureProbability, pseudoCount, outcomesCopy, featuresCopy)
    }


    // See https://en.wikipedia.org/wiki/Dirichlet-multinomial_distribution#Dirichlet-multinomial_as_a_compound_distribution
    override fun logPosteriorPredictive(outcomes: Set<Outcome>, value: Counter<String>): Map<Outcome, Double> {
        val n = value.values.sum()
        if (n == 0 || features.isEmpty()) { // empty input or empty model
            return outcomes.map { it to 0.0 }.toMap()
        }

        val result = mutableMapOf<String, Double>()
        val logBetaZeroCounts = value.mapValues { log(it.value.toDouble()) + logBeta(0 + pseudoCount, it.value.toDouble()) }
        val logBetaZeroCountsSum = logBetaZeroCounts.map { it.value }.sum()
        for (outcome in outcomes) {
            val outcomeIdx = outcomeToIdx[outcome]
            val alpha_0 = if (outcomeIdx != null) {
                nFeaturesPerOutcome[outcomeIdx]
            } else {
                0
            }
            val numerator = log(n.toDouble()) + logBeta(alpha_0 + features.size * pseudoCount, n.toDouble())
            result[outcome] = numerator - logBetaZeroCountsSum //assuming every class has seen every word 0 times
        }

        for ((word, count) in value) {
            val alpha_kc = features[word]
            if (alpha_kc != null) {
                val wrong = logBetaZeroCounts[word]!!
                for ((outcomeIdx, i) in alpha_kc) {
                    val outcome = idxToOutcome[outcomeIdx]!!
                    val prev = result[outcome]
                    if (prev != null) { //outcomes may be a subset of all the outcomes we've observed
                        val correct = log(count.toDouble()) + logBeta(i + pseudoCount, count.toDouble())
                        result[outcome] = prev - (-wrong + correct) //subtract error, add correct value
                    }
                }
            }
        }

        return result
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
            .putAllOutcomes(outcomeToIdx)
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