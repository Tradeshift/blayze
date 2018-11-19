package com.tradeshift.blayze.collection

import com.google.protobuf.ByteString
import com.tradeshift.blayze.Protos
import java.nio.ByteBuffer
import java.util.TreeMap

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

    fun toProto(): Protos.SparseIntVector = Protos.SparseIntVector.newBuilder()
            .setIndices(toByteString(indices)).setValues(toByteString(values)).build()


    companion object {
        private fun toIntArray(bytes: ByteString): IntArray {
            val buffer = bytes.asReadOnlyByteBuffer().asIntBuffer()
            val arr = IntArray(buffer.remaining())
            buffer.get(arr)
            return arr
        }

        private fun toByteString(ints: IntArray): ByteString {
            val bb = ByteBuffer.allocate(ints.size * Integer.BYTES)
            bb.asIntBuffer().put(ints)
            return ByteString.copyFrom(bb)
        }

        fun fromMap(map: Map<Int, Int>): SparseIntVector {
            val sortedNonZeros = map.filter { it.value != 0 }.toSortedMap()
            return SparseIntVector(sortedNonZeros.keys.toIntArray(), sortedNonZeros.values.toIntArray())
        }

        fun fromProto(proto: Protos.SparseIntVector): SparseIntVector =
                SparseIntVector(toIntArray(proto.indices), toIntArray(proto.values))
    }
}
