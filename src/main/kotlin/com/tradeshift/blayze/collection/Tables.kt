package com.tradeshift.blayze.collection

import com.google.common.collect.Interners
import com.google.common.collect.Interner
import java.util.HashMap


/**
 * A table class which can be used to store and access value via a RowKey and a ColumnKey
 */

interface Table<RowKey, ColumnKey, V>{
    val entries: Set<Map.Entry<Pair<RowKey, ColumnKey>, V>>
    val rowKeySet: Set<RowKey>
    val columnKeySet: Set<ColumnKey>

    operator fun get(k1: RowKey, k2: ColumnKey): V?
    fun toMutableTable(): MutableTable<RowKey, ColumnKey, V>
    fun toTable(): Table<RowKey, ColumnKey, V>
}

interface MutableTable<RowKey, ColumnKey, V> : Table<RowKey, ColumnKey, V> {
    fun put(rowKey: RowKey, columnKey: ColumnKey, value: V)

}

/**
 * A table implementation based on a Map<Pair<RowKey, ColumnKey>, V>
 */
class MapTable<RowKey, ColumnKey, V>(val map: Map<Pair<RowKey, ColumnKey>, V> = mapOf()) : Table<RowKey, ColumnKey, V> {

    val iR = Interners.newStrongInterner<RowKey>()
    val iC = Interners.newStrongInterner<ColumnKey>()

    override val entries: Set<Map.Entry<Pair<RowKey, ColumnKey>, V>> = map.entries

    override val rowKeySet: Set<RowKey> by lazy { // as table is immutable, we use lazy delegates to only compute the property once on first access
        map.keys.map { it.first }.toSet()
    }

    override val columnKeySet: Set<ColumnKey> by lazy {
        map.keys.map { it.second }.toSet()
    }

    override operator fun get(k1: RowKey, k2: ColumnKey) = map[k1 to k2]

    override fun toMutableTable(): MutableTable<RowKey, ColumnKey, V> {
        return MutableMapTable(map.toMutableMap())
    }

    override fun toTable(): Table<RowKey, ColumnKey, V> {
        return this
    }
}

/**
 * A mutable table implementation based on a Map<Pair<RowKey, ColumnKey>, V>
 */
class MutableMapTable<RowKey, ColumnKey, V>(
        val map: MutableMap<Pair<RowKey, ColumnKey>, V> = mutableMapOf()
) : MutableTable<RowKey, ColumnKey, V>{

    val iR = Interners.newStrongInterner<RowKey>()
    val iC = Interners.newStrongInterner<ColumnKey>()

    override val entries: Set<Map.Entry<Pair<RowKey, ColumnKey>, V>>
        get() = map.entries

    override val rowKeySet: Set<RowKey>
        get() = map.keys.map { it.first }.toSet()

    override val columnKeySet: Set<ColumnKey>
        get() = map.keys.map { it.second }.toSet()

    override operator fun get(k1: RowKey, k2: ColumnKey) = map[k1 to k2]

    override fun put(rowKey: RowKey, columnKey: ColumnKey, value: V) {
        map[iR.intern(rowKey) to iC.intern(columnKey)] = value
    }

    override fun toTable(): Table<RowKey, ColumnKey, V> {
        return MapTable(map.toMap())
    }

    override fun toMutableTable(): MutableTable<RowKey, ColumnKey, V> {
        return this
    }
}

fun <RowKey, ColumnKey, V> tableOf(pairs: Iterable<Pair<Pair<RowKey, ColumnKey>, V>>): Table<RowKey, ColumnKey, V> = MapTable(pairs.toMap())
fun <RowKey, ColumnKey, V> tableOf(vararg pairs: Pair<Pair<RowKey, ColumnKey>, V>): Table<RowKey, ColumnKey, V> = MapTable(pairs.toMap())
fun <RowKey, ColumnKey, V> mutableTableOf(vararg pairs: Pair<Pair<RowKey, ColumnKey>, V>): MutableTable<RowKey, ColumnKey, V> = MutableMapTable(pairs.toMap().toMutableMap())
