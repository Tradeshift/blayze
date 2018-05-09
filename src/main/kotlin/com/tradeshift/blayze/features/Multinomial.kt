package com.tradeshift.blayze.features

import com.tradeshift.blayze.Protos
import com.tradeshift.blayze.collection.Counter
import com.tradeshift.blayze.collection.Table
import com.tradeshift.blayze.collection.tableOf
import com.tradeshift.blayze.dto.Outcome
import kotlin.math.ln
import kotlin.math.pow

/**
 * A feature for multinomial data.
 *
 * @property includeFeatureProbability Include new features with this probability. See Ad Click Prediction: a View from the Trenches, Table 2
 * @property pseudoCount Add this number to all counts, even zero counts. Prevents 0 probability. See http://en.wikipedia.org/wiki/Naive_Bayes_classifier#Multinomial_naive_Bayes
 * @property countTable Table with row of outcomes and column of features and the count for each occurrence e.g. {{"positive": {"awesome": 67, "terrible": 14}, "negative": {"awesome": 11, "terrible": 114}}
 */
class Multinomial(
        private val includeFeatureProbability: Double = 1.0,
        private val pseudoCount: Double = 1.0,
        private val countTable: Table<String, String, Int> = tableOf()
) : Feature<Multinomial, Counter<String>> {

    private val outcomeCounter: Counter<String> by lazy { sumColumns(countTable) }

    override fun batchUpdate(updates: List<Pair<Outcome, Counter<String>>>): Multinomial {
        val updatedTable = countTable.toMutableTable()

        val knownFeatures = updatedTable.columnKeySet.toMutableSet()
        for ((outcome, counts) in updates) {
            for ((feature, value) in counts) {
                if (knownFeatures.contains(feature) || Math.random() < (1.0 - (1.0 - includeFeatureProbability).pow(value))) {
                    knownFeatures.add(feature)
                    val count = updatedTable[outcome, feature] ?: 0
                    updatedTable.put(outcome, feature, count + value)
                }
            }
        }
        return Multinomial(includeFeatureProbability, pseudoCount, updatedTable.toTable())
    }

    override fun logProbability(outcomes: Set<Outcome>, value: Counter<String>): Map<Outcome, Double> {
        val knownFeatures: Set<String> = value.keys.intersect(countTable.columnKeySet)
        val numOfFeatures = countTable.columnKeySet.size

        val results = mutableMapOf<Outcome, Double>()
        for (outcome in outcomes) {
            var logProb = 0.0
            for (feature in knownFeatures) {
                val count = ((countTable[outcome, feature] ?: 0) + pseudoCount)
                logProb += value[feature] * (ln(count) - ln(outcomeCounter[outcome] + (numOfFeatures * pseudoCount)))
            }
            results[outcome] = logProb
        }
        return results
    }

    private fun sumColumns(table: Table<String, String, Int>): Counter<String> {
        val counts = mutableMapOf<String, Int>()
        for (cell in table.entries) {
            val outcome = cell.key.first
            val current = counts.getOrDefault(outcome, 0)
            counts[outcome] = current + cell.value
        }
        return Counter(counts)
    }

    fun toProto(): Protos.Multinomial {
        return Protos.Multinomial.newBuilder()
                .setIncludeFeatureProbability(includeFeatureProbability)
                .setPseudoCount(pseudoCount)
                .setTable(Protos.Table.newBuilder()
                        .addAllEntries(countTable.entries.map {
                            Protos.Entry.newBuilder()
                                    .setRowKey(it.key.first)
                                    .setColumnKey(it.key.second)
                                    .setCount(it.value)
                                    .build()
                        }).build()
                ).build()
    }

    companion object {
        fun fromProto(proto: Protos.Multinomial): Multinomial {
            return Multinomial(proto.includeFeatureProbability, proto.pseudoCount, tableOf(proto.table.entriesList.map { Pair(Pair(it.rowKey, it.columnKey), it.count) }))
        }
    }
}
