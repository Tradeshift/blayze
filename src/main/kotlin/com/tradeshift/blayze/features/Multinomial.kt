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

    private fun invertUpdates(updates: Sequence<Pair<Outcome, Counter<String>>>): Map<String, Counter<Outcome>> =
            group(updates.asSequence().flatMap { (o, cnt) -> cnt.map { Triple(it.key, o, it.value) }.asSequence() })

    fun toProto(): Protos.Multinomial = Protos.Multinomial.newBuilder()
            .setIncludeFeatureProbability(includeFeatureProbability)
            .setPseudoCount(pseudoCount)
            .setSparseTable(sparseTable.toProto())
            .build()

    companion object {
        fun fromProto(proto: Protos.Multinomial): Multinomial {
            return if (proto.hasSparseTable()) {
                Multinomial(proto.includeFeatureProbability, proto.pseudoCount, SparseTable.fromProto(proto.sparseTable))
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
