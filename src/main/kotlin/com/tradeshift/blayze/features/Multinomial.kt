package com.tradeshift.blayze.features

import com.tradeshift.blayze.Protos
import com.tradeshift.blayze.collection.SparseVector
import com.tradeshift.blayze.collection.Counter
import com.tradeshift.blayze.dto.Outcome
import kotlin.math.ln
import kotlin.math.pow


/**
 * A feature for multinomial data.
 *
 * @property includeFeatureProbability Include new features with this probability. See Ad Click Prediction: a View from the Trenches, Table 2
 * @property pseudoCount Add this number to all counts, even zero counts. Prevents 0 probability. See http://en.wikipedia.org/wiki/Naive_Bayes_classifier#Multinomial_naive_Bayes
 */
class Multinomial private constructor(
        private val includeFeatureProbability: Double,
        private val pseudoCount: Double,
        private val featureMap: Map<String, SparseVector>,
        private val outcomeIndices: Map<Outcome, Int>
) : Feature<Multinomial, Counter<String>> {

    constructor(includeFeatureProbability: Double = 1.0, pseudoCount: Double = 1.0) :
            this(includeFeatureProbability, pseudoCount, mapOf(), mapOf())

    private val outcomeCounts: IntArray by lazy {
        val res = IntArray(outcomeIndices.size)
        for (vector in featureMap.values) {
            vector.indexed().forEach { res[it.index] += it.value }
        }
        res
    }

    override fun batchUpdate(updates: List<Pair<Outcome, Counter<String>>>): Multinomial {
        val outcomesCopy = outcomeIndices.toMutableMap()
        val featuresCopy = featureMap.toMutableMap()
        for ((feature, counter) in invertUpdates(sampleUpdates(updates.asSequence()))) {
            val update = counter.mapKeys { outcomesCopy.getOrPut(it.key, { outcomesCopy.size }) }
            val vec: SparseVector = SparseVector.fromMap(update)
            featuresCopy.compute(feature, { _, previous -> previous?.add(vec) ?: vec })
        }
        return Multinomial(includeFeatureProbability, pseudoCount, featuresCopy, outcomesCopy)
    }

    override fun logProbability(outcomes: Set<Outcome>, value: Counter<String>): Map<Outcome, Double> {
        val logCounts = DoubleArray(outcomeIndices.size)
        val nonZeroCounts = IntArray(outcomeIndices.size)
        var nFeatures = 0
        for ((feature, featureCount) in value) {
            val vector = featureMap[feature]
            if (vector != null) {
                nFeatures += featureCount
                vector.indexed().forEach { (idx, value) ->
                    nonZeroCounts[idx] += featureCount
                    logCounts[idx] += ln(value + pseudoCount) * featureCount
                }
            }
        }

        fun logProbZeros(nonZeroCount: Int): Double = (nFeatures - nonZeroCount) * ln(pseudoCount)
        fun logProbDenominator(outcomeCount: Int) =  nFeatures * ln(outcomeCount + featureMap.size * pseudoCount)

        return outcomes.map { outcome ->
            val idx = outcomeIndices[outcome]
            val logProb = if (idx != null) {
                logCounts[idx] + logProbZeros(nonZeroCounts[idx]) - logProbDenominator(outcomeCounts[idx])
            } else {
                // Unseen outcomes have only zero values
                logProbZeros(nonZeroCount = 0) - logProbDenominator(0)
            }
            outcome to logProb
        }.toMap()
    }

    private fun sampleUpdates(updates: Sequence<Pair<Outcome, Counter<String>>>): Sequence<Pair<Outcome, Counter<String>>> {
        fun shouldSampleFeature(count: Int) = Math.random() < (1.0 - (1.0 - includeFeatureProbability).pow(count))
        val knownFeatures = featureMap.keys.toMutableSet()
        return updates.map { (outcome, counts) ->
            val filteredCounts = counts.filter { it.key in knownFeatures || shouldSampleFeature(it.value) }
            knownFeatures.addAll(filteredCounts.keys)
            outcome to Counter(filteredCounts)
        }
    }

    private fun invertUpdates(updates: Sequence<Pair<Outcome, Counter<String>>>): Map<String, Counter<Outcome>> =
            group(updates.asSequence().flatMap { (o, cnt) -> cnt.map { Triple(it.key, o, it.value) }.asSequence() })

    fun toProto(): Protos.Multinomial {
        return Protos.Multinomial.newBuilder()
                .setIncludeFeatureProbability(includeFeatureProbability)
                .setPseudoCount(pseudoCount)
                .setSparseTable(
                        Protos.SparseTable.newBuilder()
                                .putAllOutcomes(outcomeIndices)
                                .putAllFeatureMap(featureMap.mapValues { it.value.toProto() })
                                .build()
                )
                .build()
    }

    companion object {
        fun fromProto(proto: Protos.Multinomial): Multinomial {
            return if (proto.hasSparseTable()) {
                val featureMap = proto.sparseTable.featureMapMap.mapValues { SparseVector.fromProto(it.value) }
                Multinomial(proto.includeFeatureProbability, proto.pseudoCount, featureMap, proto.sparseTable.outcomesMap)
            } else {
                val updates = group(proto.table.entriesList.asSequence().map { Triple(it.rowKey, it.columnKey, it.count) })
                Multinomial(proto.includeFeatureProbability, proto.pseudoCount).batchUpdate(updates.toList())
            }
        }
    }
}

private fun group(updates: Sequence<Triple<String, String, Int>>): Map<String, Counter<String>> = updates
        .groupingBy { it.first }
        .fold({ _, _ -> mutableMapOf<String, Int>() }) { _, counter, triple ->
            counter[triple.second] = (counter[triple.second] ?: 0) + triple.third
            counter
        }
        .mapValues { Counter(it.value) }
