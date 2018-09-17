package com.tradeshift.blayze.collection

/**
 * A immutable map that counts each unique values of an iterable, and returns zero instead of Int? for keys that are not present.
 */
class Counter<T>(private val counts: Map<T, Int> = mapOf()) : Map<T, Int> by counts {

    constructor(entries: Iterable<T>) : this(entries.groupingBy { it }.eachCount())
    constructor(vararg entries: T) : this(entries.asIterable())
    constructor(entry: T) : this(mapOf(entry to 1))

    override operator fun get(key: T): Int {
        return counts[key] ?: 0
    }

}
