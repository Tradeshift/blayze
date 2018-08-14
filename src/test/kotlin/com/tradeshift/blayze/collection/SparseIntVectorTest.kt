package com.tradeshift.blayze.collection

import org.junit.Assert.*
import org.junit.Test

class SparseIntVectorTest {
    private val vector = SparseIntVector.fromMap(mapOf(
            3 to 30,
            2 to 20,
            1 to 10,
            7 to 70
    ))

    @Test
    fun iterates_in_order() {
        val expected = listOf(1 to 10, 2 to 20, 3 to 30, 7 to 70)
        assertEquals(expected, vector.toList())
    }

    @Test
    fun add() {
        val other = SparseIntVector.fromMap(mapOf(1 to 10, 4 to 40, 7 to 70))

        val expected = listOf(1 to 20, 2 to 20, 3 to 30, 4 to 40, 7 to 140)
        assertEquals(expected, vector.add(other).toList())
    }
}