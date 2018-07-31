package com.tradeshift.blayze.collection

import org.junit.Assert.*
import org.junit.Test

class SparseVectorTest {
    private val vector = SparseVector.fromMap(mapOf(
            3 to 30,
            2 to 20,
            1 to 10,
            7 to 70
    ))

    @Test
    fun iterates_in_order() {
        val expected = listOf(1 to 10, 2 to 20, 3 to 30, 7 to 70)
                .map { IndexedValue(it.first, it.second) }
        assertEquals(expected, vector.indexed().asSequence().toList())
    }

    @Test
    fun add() {
        val other = SparseVector.fromMap(mapOf(1 to 10, 4 to 40, 7 to 70))

        val expected = listOf(1 to 20, 2 to 20, 3 to 30, 4 to 40, 7 to 140)
                .map { IndexedValue(it.first, it.second) }
        assertEquals(expected, vector.add(other).indexed().asSequence().toList())
    }
}