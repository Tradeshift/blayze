package com.tradeshift.blayze.features

import com.tradeshift.blayze.Protos
import com.tradeshift.blayze.collection.Counter
import com.tradeshift.blayze.dto.FeatureValue
import com.tradeshift.blayze.dto.Outcome


/**
 * A feature for free text, e.g. reviews, comments, email messages, etc. Does simple pre-processing and splits the text into words.
 *
 * The pre-processing replaces all non-letters and non-numbers with spaces, lowercases, splits on spaces and finally removes english stopwords.
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
        private val stopWords = WordCounter::class.java.getResource("/english-stop-words-large.txt")
                .readText().split("\n").toSet()

        fun countWords(q: String): Counter<String> {
            // https://www.regular-expressions.info/unicode.html
            val words = q
                    .replace("[^\\p{L}\\p{N}]+".toRegex(), " ") // only keep any kind of letter and numbers from any language, others become space
                    .trim()
                    .lowercase()
                    .split(" ")
                    .filter { it !in stopWords }
            // with this setup, the 20newsgroup score is above 0.64

            return Counter(words)
        }
    }
}
