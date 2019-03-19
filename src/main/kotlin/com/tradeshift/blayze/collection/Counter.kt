package com.tradeshift.blayze.collection

/**
 * A immutable map that counts each unique values of an iterable, and returns zero instead of Int? for keys that are not present.
 */
class Counter<T>(private val counts: Map<T, Int> = mapOf()) : Map<T, Int> by counts {

    constructor(entries: Iterable<T>) : this(entries.groupingBy { it }.eachCount())
    constructor(vararg entries: T) : this(entries.asIterable())

    override operator fun get(key: T): Int {
        return counts[key] ?: 0
    }

    override fun equals(other: Any?): Boolean {
        return counts == other
    }

    override fun hashCode(): Int {
        return counts.hashCode()
    }

    override fun toString(): String {
        return counts.toString()
    }

}
