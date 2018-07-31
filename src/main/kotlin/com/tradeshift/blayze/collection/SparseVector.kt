package com.tradeshift.blayze.collection

import java.util.TreeMap

class SparseVector private constructor(private val indices: IntArray, private val values: IntArray) {

    fun add(o: SparseVector): SparseVector {
        val m = TreeMap<Int, Int>()
        (0 until indices.size).forEach { m[indices[it]] = values[it] }
        (0 until o.indices.size).forEach { m[o.indices[it]] = (m[o.indices[it]] ?: 0) + o.values[it] }
        return SparseVector(m.keys.toIntArray(), m.values.toIntArray())
    }

    fun indexed(): Iterator<IndexedValue<Int>> {
        return indices.asSequence().withIndex().map { (cursor, idx) -> IndexedValue(idx, values[cursor]) }.iterator()
    }

    companion object {
        fun fromMap(map: Map<Int, Int>): SparseVector {
            val sortedNonZeros = map.filter { it.value != 0 }.toSortedMap()
            return SparseVector(sortedNonZeros.keys.toIntArray(), sortedNonZeros.values.toIntArray())
        }
    }
}
