package com.tradeshift.blayze.features

import com.tradeshift.blayze.Protos
import com.tradeshift.blayze.collection.Counter
import com.tradeshift.blayze.dto.FeatureValue
import com.tradeshift.blayze.dto.Outcome

/**
 * A feature for categorical data, i.e. 1 of K data, e.g. user-ids, countries, etc.
 */
class Categorical(private val delegate: Multinomial = Multinomial()) : Feature<FeatureValue> {
    override fun toMutableFeature(): MutableCategorical {
        return MutableCategorical(delegate.toMutableFeature())
    }

    override fun logProbability(outcomes: Set<Outcome>, value: FeatureValue): Map<Outcome, Double> {
        val counts = Counter(value)
        return delegate.logProbability(outcomes, counts)
    }

    fun toProto(): Protos.Categorical {
        return Protos.Categorical.newBuilder().setDelegate(delegate.toProto()).build()
    }

    companion object {
        fun fromProto(proto: Protos.Categorical): Categorical {
            return Categorical(Multinomial.fromProto(proto.delegate))
        }
    }

}

class MutableCategorical(private val delegate: MutableMultinomial = MutableMultinomial()) : MutableFeature<FeatureValue> {
    override fun batchUpdate(updates: List<Pair<Outcome, FeatureValue>>) {
        val categories = updates.map { Pair(it.first, Counter(it.second)) }
        delegate.batchUpdate(categories)
    }

    override fun toFeature(): Categorical {
        return Categorical(delegate.toFeature())
    }

}
