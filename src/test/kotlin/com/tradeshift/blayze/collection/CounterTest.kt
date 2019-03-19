package com.tradeshift.blayze.collection

import org.junit.Assert.assertEquals
import org.junit.Test

class CounterTest {

    @Test
    fun when_no_counts_return_zero() {
        val c = Counter("foo", "foo", "bar")
        assertEquals(2, c["foo"])
        assertEquals(1, c["bar"])
        assertEquals(0, c["baz"])
    }

    @Test
    fun test_constructors_count_unique() {
        var c = Counter("foo")
        assertEquals(1, c["foo"])

        c = Counter("foo", "foo", "bar")
        assertEquals(2, c["foo"])
        assertEquals(1, c["bar"])

        c = Counter(listOf("foo", "foo", "baz"))
        assertEquals(2, c["foo"])
        assertEquals(1, c["baz"])
    }


    @Test
    fun test_counters_are_equal() {
        assertEquals(Counter("baz"), Counter("baz"))
    }

}
