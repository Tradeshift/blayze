package com.tradeshift.blayze

import com.tradeshift.blayze.dto.FeatureName
import com.tradeshift.blayze.dto.Inputs
import com.tradeshift.blayze.dto.Outcome
import com.tradeshift.blayze.dto.Update
import com.tradeshift.blayze.features.*

/**
 * A flexible and robust naive bayes classifier that supports multiple features of multiple types, and iterative updating.
 *
 * The naive bayes classifier computes the probability of outcomes given inputs, p(outcome|inputs)
 *
 * The bayes aspect is that it uses bayes rule, such that,
 *
 *      p(outcome|inputs) = p(inputs|outcome)p(outcome)/p(inputs)
 *
 * Since p(inputs) is the same for all outcomes we can disregard it, as long as we remember to normalize later,
 *
 *      p(outcome|inputs) ~ p(inputs|outcome)p(outcome)
 *
 * The naive aspect is that is assumes inputs are independent given the outcome, such that,
 *
 *      p(outcome|input_1, input_2) ~ p(input_1|outcome)p(input_2|outcome)p(outcome)
 *
 * This way the problem breaks down into estimating the prior, p(outcome), and a set of likelihoods, p(input_n|outcome).
 *  * The prior, p(outcome), is estimated as #outcome_c/sum_c(#outcome_c), i.e. number of times outcome_c has been seen over how many outcomes we've seen in total.
 *  * The likelihoods are estimated by named [Feature]s, with names matching the names given in the [Inputs].
 *
 * The main usecase is initializing an empty Model, and then building the model by repeatedly calling [batchAdd] with new data. See [Update].
 *
 * The model is designed to be flexible and robust.
 *  * New features can be added on the fly by including them in the list of [Update] when calling [batchAdd]
 *  * When calling [predict], only the features in the [Inputs] are considered.
 *  * When calling [predict], if [Inputs] contain a feature not present in the model, it is ignored.
 *
 * @param priorCounts           Number of times outcome has been seen, e.g {"positive": 2, "negative": 3}
 * @param textFeatures          The text features used by this model.
 * @param categoricalFeatures   The categorical features used by this model.
 * @param gaussianFeatures      The gaussian features used by this model.
 */
class Model(
        private val priorCounts: Map<Outcome, Int> = mapOf(),
        private val textFeatures: Map<FeatureName, Text> = mapOf(),
        private val categoricalFeatures: Map<FeatureName, Categorical> = mapOf(),
        private val gaussianFeatures: Map<FeatureName, Gaussian> = mapOf()
) {

    private val logPrior: Map<String, Double> by lazy {
        val logTotal = Math.log(priorCounts.values.sum().toDouble())
        priorCounts.mapValues { Math.log(it.value.toDouble()) - logTotal }
    }

    fun toProto(): Protos.Model {
        return Protos.Model.newBuilder()
                .setModelVersion(Model.version)
                .putAllPriorCounts(priorCounts)
                .putAllTextFeatures(textFeatures.mapValues { it.value.toProto() })
                .putAllCategoricalFeatures(categoricalFeatures.mapValues { it.value.toProto() })
                .putAllGaussianFeatures(gaussianFeatures.mapValues { it.value.toProto() })
                .build()
    }

    fun toMutableModel(): MutableModel {
        return MutableModel(
                priorCounts.toMutableMap(),
                textFeatures.mapValues { it.value.toMutableFeature() }.toMutableMap(),
                categoricalFeatures.mapValues { it.value.toMutableFeature() }.toMutableMap(),
                gaussianFeatures.mapValues { it.value.toMutableFeature() }.toMutableMap()
        )
    }

    companion object {

        private val version = 2

        fun fromProto(proto: Protos.Model): Model {
            if (proto.modelVersion != version) {
                throw IllegalArgumentException("This version of blayze requires protobuf model version $version " +
                        "Attempted to load protobuf with version ${proto.modelVersion}")
            }
            return Model(
                    proto.priorCountsMap,
                    proto.textFeaturesMap.mapValues { Text.fromProto(it.value) },
                    proto.categoricalFeaturesMap.mapValues { Categorical.fromProto(it.value) },
                    proto.gaussianFeaturesMap.mapValues { Gaussian.fromProto(it.value) }
            )
        }

    }

    /**
     * Predicts the outcome of [Inputs] using naive bayes, e.g. p(outcome|inputs) = p(inputs|outcome)p(outcome)/p(inputs)
     *
     * @return predicted outcomes and their probability, e.g. {"positive": 0.3124, "negative": 0.6876}
     */
    fun predict(inputs: Inputs): Map<Outcome, Double> {
        if (priorCounts.isEmpty()) {
            return mapOf()
        }

        val maps = mutableListOf<Map<Outcome, Double>>()
        maps.add(logPrior)
        maps.addAll(logProbabilities(textFeatures, inputs.text))
        maps.addAll(logProbabilities(categoricalFeatures, inputs.categorical))
        maps.addAll(logProbabilities(gaussianFeatures, inputs.gaussian))

        return normalize(sumMaps(maps))
    }

    private fun <V> logProbabilities(features: Map<FeatureName, Feature<V>>, values: Map<FeatureName, V>): List<Map<Outcome, Double>> {
        return values.map { (key, value) -> features[key]?.logProbability(logPrior.keys, value) }.filterNotNull()
    }

    private fun sumMaps(maps: List<Map<Outcome, Double>>): Map<Outcome, Double> {
        val sum = mutableMapOf<Outcome, Double>()
        for (map in maps) {
            for ((key, value) in map) {
                val current = sum.getOrDefault(key, 0.0)
                sum[key] = current + value
            }
        }
        return sum
    }

    private fun normalize(suggestions: Map<Outcome, Double>): Map<Outcome, Double> {
        val max: Double = suggestions.maxBy({ it.value })?.value ?: 0.0
        val vals = suggestions.mapValues { Math.exp(it.value - max) }
        val norm = vals.values.sum()
        return vals.mapValues { it.value / norm }
    }


}


class MutableModel(
        private val priorCounts: MutableMap<Outcome, Int> = hashMapOf(),
        private val textFeatures: MutableMap<FeatureName, MutableText> = hashMapOf(),
        private val categoricalFeatures: MutableMap<FeatureName, MutableCategorical> = hashMapOf(),
        private val gaussianFeatures: MutableMap<FeatureName, MutableGaussian> = hashMapOf()
) {

    fun toModel(): Model {
        return Model(
                priorCounts,
                textFeatures.mapValues { it.value.toFeature() },
                categoricalFeatures.mapValues { it.value.toFeature() },
                gaussianFeatures.mapValues { it.value.toFeature() }
        )
    }

    /**
     * Creates a new model with the [Update]s added.
     */
    fun add(update: Update) {
        batchAdd(listOf(update))
    }

    /**
     * Creates a new model with the updates added.
     *
     * @param updates List of observed updates
     * @return new updated Model
     */
    fun batchAdd(updates: List<Update>) {
        updates.map { it.outcome }.groupingBy { it }.eachCountTo(priorCounts)

        updateFeatures(categoricalFeatures, { MutableCategorical() }, updates, { it.categorical })
        updateFeatures(textFeatures, { MutableText() }, updates, { it.text })
        updateFeatures(gaussianFeatures, { MutableGaussian() }, updates, { it.gaussian })

    }

    private fun <V, F : MutableFeature<V>> updateFeatures(
            features: MutableMap<FeatureName, F>,
            creator: () -> F,
            updates: List<Update>,
            extractor: (Inputs) -> Map<FeatureName, V>) {

        val outcomes = updates.map { it.outcome }
        val values = updates.map { extractor(it.inputs) }

        val data: Map<FeatureName, List<Pair<Outcome, V>>> = zipOutcomesAndValues(outcomes, values)
        data.toList().parallelStream().forEach { (name, data) ->
            synchronized(features) {
                //This should be lightning fast
                features.getOrPut(name, creator)
            }.batchUpdate(data) // This is the heavy part
        }
    }

    private fun <V> zipOutcomesAndValues(outcomes: List<Outcome>, values: List<Map<FeatureName, V>>): Map<FeatureName, List<Pair<Outcome, V>>> {
        val result = mutableMapOf<FeatureName, MutableList<Pair<Outcome, V>>>()
        for ((outcome, feature) in outcomes zip values) {
            for ((featureName, value) in feature) {
                result.getOrPut(featureName, { mutableListOf() }).add(outcome to value)
            }
        }
        return result
    }


}