package com.tradeshift.blayze.features

import com.tradeshift.blayze.Protos
import com.tradeshift.blayze.collection.Counter
import com.tradeshift.blayze.dto.FeatureValue
import com.tradeshift.blayze.dto.Outcome

/**
 * A feature for categorical data, i.e. 1 of K data, e.g. user-ids, countries, etc.
 */
class Categorical(val delegate: Multinomial = Multinomial()) : Feature<Categorical, FeatureValue, Multinomial.Parameters> {

    constructor(includeFeatureProbability: Double = 1.0, pseudoCount: Double = 0.1) : this(Multinomial(includeFeatureProbability, pseudoCount))

    override fun withParameters(parameters: Multinomial.Parameters): Categorical {
        return Categorical(delegate.withParameters(parameters))
    }

    fun toProto(): Protos.Categorical {
        return Protos.Categorical.newBuilder().setDelegate(delegate.toProto()).build()
    }

    companion object {
        fun fromProto(proto: Protos.Categorical): Categorical {
            return Categorical(Multinomial.fromProto(proto.delegate))
        }
    }

    override fun batchUpdate(updates: List<Pair<Outcome, FeatureValue>>, parameters: Multinomial.Parameters?): Categorical {
        val categories = updates.map { Pair(it.first, Counter(it.second)) }
        return Categorical(delegate.batchUpdate(categories, parameters))
    }

    override fun logPosteriorPredictive(outcomes: Set<Outcome>, value: FeatureValue, parameters: Multinomial.Parameters?): Map<Outcome, Double> {
        val counts = Counter(value)
        return delegate.logPosteriorPredictive(outcomes, counts, parameters)
    }
}
