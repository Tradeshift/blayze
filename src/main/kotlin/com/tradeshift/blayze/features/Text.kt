package com.tradeshift.blayze.features

import com.tradeshift.blayze.Protos
import com.tradeshift.blayze.collection.Counter
import com.tradeshift.blayze.dto.FeatureValue
import com.tradeshift.blayze.dto.Outcome


/**
 * A feature for free text, e.g. reviews, comments, email messages, etc. Lowercases and splits the text into words on spaces.
 * Text processing is intentionally minimal. To get good results on your data you might have to do additional preprocessing.
 */
class Text(val delegate: Multinomial = Multinomial()) : Feature<Text, FeatureValue, Multinomial.Parameters> {

    constructor(includeFeatureProbability: Double = 1.0, pseudoCount: Double = 0.1) : this(Multinomial(includeFeatureProbability, pseudoCount))

    override fun withParameters(parameters: Multinomial.Parameters): Text {
        return Text(delegate.withParameters(parameters))
    }

    fun toProto(): Protos.Text {
        return Protos.Text.newBuilder().setDelegate(delegate.toProto()).build()
    }

    companion object {
        fun fromProto(proto: Protos.Text): Text {
            return Text(Multinomial.fromProto(proto.delegate))
        }
    }

    override fun batchUpdate(updates: List<Pair<Outcome, FeatureValue>>, parameters: Multinomial.Parameters?): Text {
        val words = updates.map { Pair(it.first, WordCounter.countWords(it.second)) }
        return Text(delegate.batchUpdate(words, parameters))
    }

    override fun logPosteriorPredictive(outcomes: Set<Outcome>, value: FeatureValue, parameters: Multinomial.Parameters?): Map<Outcome, Double> {
        val inputWordCounts = WordCounter.countWords(value)
        return delegate.logPosteriorPredictive(outcomes, inputWordCounts, parameters)
    }

    object WordCounter {
        fun countWords(q: String): Counter<String> {
            val words = q.toLowerCase().split(" ").filter { it.isNotEmpty() }
            return Counter(words)
        }
    }
}
