package com.tradeshift.blayze.features

import com.tradeshift.blayze.Protos
import com.tradeshift.blayze.collection.Counter
import com.tradeshift.blayze.collection.SparseTable
import com.tradeshift.blayze.dto.Outcome
import kotlin.math.ln
import kotlin.math.pow

/**
 * A feature for multinomial data.
 *
 * @property includeFeatureProbability Include new features with this probability. See Ad Click Prediction: a View from the Trenches, Table 2
 * @property pseudoCount Add this number to all counts, even zero counts. Prevents 0 probability. See http://en.wikipedia.org/wiki/Naive_Bayes_classifier#Multinomial_naive_Bayes
 */
class Multinomial(
        private val includeFeatureProbability: Double = 1.0,
        private val pseudoCount: Double = 1.0,
        private val sparseTable: SparseTable = SparseTable()
) : Feature<Multinomial, Counter<String>> {

    override fun batchUpdate(updates: List<Pair<Outcome, Counter<String>>>): Multinomial {
        val newSparseTable: SparseTable = sparseTable
                .add(invertUpdates(sampleUpdates(updates.asSequence())).map { it.key to it.value }.asSequence())
        return Multinomial(includeFeatureProbability, pseudoCount, newSparseTable)
    }

    override fun logProbability(outcomes: Set<Outcome>, value: Counter<String>): Map<Outcome, Double> {
        fun logNumerator(v: Int) = ln(v + pseudoCount)
        fun logDenominator(v: Int) = ln(v + sparseTable.features.size * pseudoCount)

        val nFeatures = value.filterKeys(sparseTable.features::contains).values.sum()
        val logProbs = sparseTable
                .sumRows(value, ::logNumerator)
                .mapValues { it.value - nFeatures * logDenominator(sparseTable.sumRows[it.key]!!) }
        return outcomes.map { it to (logProbs[it] ?: nFeatures * (logNumerator(0)- logDenominator(0))) }.toMap()
    }

    private fun sampleUpdates(updates: Sequence<Pair<Outcome, Counter<String>>>): Sequence<Pair<Outcome, Counter<String>>> {
        fun shouldSampleFeature(count: Int) = Math.random() < (1.0 - (1.0 - includeFeatureProbability).pow(count))
        val knownFeatures = sparseTable.features.toMutableSet()
        return updates.map { (outcome, counts) ->
            val filteredCounts = counts.filter { it.key in knownFeatures || shouldSampleFeature(it.value) }
            knownFeatures.addAll(filteredCounts.keys)
            outcome to Counter(filteredCounts)
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
            .setSparseTable(sparseTable.toProto())
            .build()

    companion object {
        fun fromProto(proto: Protos.Multinomial): Multinomial  {
            if (!proto.hasSparseTable()) throw RuntimeException("Multinomial proto has no SparseTable set")
            val sparseTable = SparseTable.fromProto(proto.sparseTable)
            return Multinomial(proto.includeFeatureProbability, proto.pseudoCount, sparseTable)
        }
    }
}