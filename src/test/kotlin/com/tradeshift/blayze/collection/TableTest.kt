package com.tradeshift.blayze.collection

import org.junit.Assert.*

import org.junit.Test

class TableTest {
    private val table = tableOf(
            "1" to "a" to 1,
            "1" to "b" to 2,
            "2" to "a" to 3,
            "2" to "b" to 4
    )

    @Test
    fun get() {
        assertEquals(1, table["1", "a"])
        assertEquals(2, table["1", "b"])
        assertEquals(3, table["2", "a"])
        assertEquals(4, table["2", "b"])
    }

    @Test
    fun row_and_keyset() {
        assertEquals(setOf("a", "b"), table.columnKeySet)
        assertEquals(setOf("1", "2"), table.rowKeySet)
    }

    @Test
    fun entries() {
        val localMap = mapOf(
                "1" to "a" to 1,
                "1" to "b" to 2,
                "2" to "a" to 3,
                "2" to "b" to 4
        )
        assertEquals(localMap.entries, table.entries)
    }

    @Test
    fun row_and_keyset_mutable_table() {
        val mutableTable = table.toMutableTable()

        assertEquals(setOf("a", "b"), mutableTable.columnKeySet)
        assertEquals(setOf("1", "2"), mutableTable.rowKeySet)

        mutableTable.put("3", "c", 5)

        assertEquals(setOf("a", "b", "c"), mutableTable.columnKeySet)
        assertEquals(setOf("1", "2", "3"), mutableTable.rowKeySet)
    }

}
