package com.tradeshift.blayze.collection

import com.tradeshift.blayze.Protos
import com.tradeshift.blayze.dto.FeatureValue
import com.tradeshift.blayze.dto.Outcome

class SparseTable(private val outcomeIndices: Map<Outcome, Int>, private val rows: Map<FeatureValue, SparseVector>) {

    constructor(): this(HashMap(), HashMap())

    val sumRows: Map<Outcome, Int> by lazy {
        val res = IntArray(outcomeIndices.size)
        rows.values.forEach { it.indexed().forEach { (idx, count) -> res[idx] += count }}
        outcomeIndices.mapValues { res[it.value] }
    }

    val features: Set<FeatureValue>
        get() = rows.keys

    fun add(updates: Sequence<Pair<FeatureValue, Counter<Outcome>>>): SparseTable {
        val outcomeIndicesCopy = outcomeIndices.toMutableMap()
        val rowsCopy = rows.toMutableMap()
        for ((feature, counter) in updates) {
            val indexedUpdate = counter.mapKeys { outcomeIndicesCopy.getOrPut(it.key, { outcomeIndicesCopy.size }) }
            val vec: SparseVector = SparseVector.fromMap(indexedUpdate)
            rowsCopy.compute(feature, { _, previous -> previous?.add(vec) ?: vec })
        }
        return SparseTable(outcomeIndicesCopy, rowsCopy)
    }

    fun sumRows(counter: Counter<FeatureValue>, transform: (Int) -> Double): Map<Outcome, Double> {
        val sum = DoubleArray(outcomeIndices.size)
        val nonZeroCounts = IntArray(outcomeIndices.size)
        var nFeatures = 0
        for ((feature, count) in counter) {
            val vector = rows[feature]
            if (vector != null) {
                nFeatures += count
                for((idx, value) in vector.indexed()) {
                    sum[idx] += transform(value) * count
                    nonZeroCounts[idx] += count
                }

            }
        }
        val zeroValue = transform(0)
        nonZeroCounts.map { (nFeatures - it) * zeroValue }.forEachIndexed { idx, zeroTotal -> sum[idx] += zeroTotal }
        return outcomeIndices.mapValues { sum[it.value] }
    }

    fun toProto(): Protos.SparseTable = Protos.SparseTable.newBuilder()
            .putAllOutcomes(outcomeIndices)
            .putAllFeatureMap(rows.mapValues { it.value.toProto() })
            .build()

    companion object {
        fun fromProto(proto: Protos.SparseTable) = SparseTable(
                proto.outcomesMap,
                proto.featureMapMap.mapValuesTo(HashMap(), { SparseVector.fromProto(it.value) })
        )
    }
}