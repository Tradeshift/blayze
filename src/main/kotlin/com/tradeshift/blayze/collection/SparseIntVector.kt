package com.tradeshift.blayze.collection

import com.google.protobuf.ByteString
import com.tradeshift.blayze.Protos
import java.nio.ByteBuffer
import java.util.*


/**
 * A sparse vector that stores non-zero indices and values in primitive arrays.
 */
class SparseIntVector private constructor(private val indices: IntArray, private val values: IntArray) : Iterable<Pair<Int, Int>> {

    fun add(other: SparseIntVector): SparseIntVector {
        val m = TreeMap<Int, Int>()
        m.putAll(this)
        other.forEach { (idx, value) -> m[idx] = (m[idx] ?: 0) + value }
        return SparseIntVector(m.keys.toIntArray(), m.values.toIntArray())
    }

    /**
     * Iterator over non-zero elements of the vector.
     * @return An iterator of <index, value> pairs of non-zero elements
     */
    override fun iterator(): Iterator<Pair<Int, Int>> = indices.zip(values).iterator()

    fun toProto(): Protos.SparseIntVector {
        val iBuffer = ByteBuffer.allocate(Integer.BYTES * indices.size)
        iBuffer.asIntBuffer().put(indices)
        val vBuffer = ByteBuffer.allocate(Integer.BYTES * values.size)
        vBuffer.asIntBuffer().put(values)

        return Protos.SparseIntVector.newBuilder()
                .setIndices(ByteString.copyFrom(iBuffer))
                .setValues(ByteString.copyFrom(vBuffer))
                .build()
    }

    companion object {
        fun fromMap(map: Map<Int, Int>): SparseIntVector {
            val sortedNonZeros = map.filter { it.value != 0 }.toSortedMap()
            return SparseIntVector(sortedNonZeros.keys.toIntArray(), sortedNonZeros.values.toIntArray())
        }

        fun fromProto(proto: Protos.SparseIntVector): SparseIntVector {
            val indices = ByteBuffer.wrap(proto.indices.toByteArray()).asIntBuffer().array()
            val values = ByteBuffer.wrap(proto.values.toByteArray()).asIntBuffer().array()
            return SparseIntVector(indices, values)
        }

    }
}
