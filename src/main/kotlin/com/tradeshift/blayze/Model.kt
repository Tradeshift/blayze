package com.tradeshift.blayze

import com.tradeshift.blayze.dto.FeatureName
import com.tradeshift.blayze.dto.Inputs
import com.tradeshift.blayze.dto.Outcome
import com.tradeshift.blayze.dto.Update
import com.tradeshift.blayze.features.*

/**
 * A flexible and robust Bayesian Naive Bayes classifier that supports multiple features of multiple types, and online learning.
 *
 * The bayesian naive bayes classifier computes the probability of outcomes conditioned on the inputs and the observed data D: p(outcome | inputs, D)
 *
 * The main usecase is initializing an empty Model, and then building the model by repeatedly calling [batchAdd] with new data. See [Update].
 *
 * The model is designed to be flexible and robust.
 *  * New features can be added on the fly by including them in the list of [Update] when calling [batchAdd]
 *  * When calling [predict], only the features in the [Inputs] are considered.
 *  * When calling [predict], if [Inputs] contain a feature not present in the model, it is ignored.
 *
 * The bayes aspect is that it uses bayes rule, such that,
 *
 *      p(outcome | inputs, D) = p(inputs | outcome, D) p(outcome | D) / p(inputs | D)
 *
 * Since p(inputs | D) is the same for all outcomes we can disregard it, as long as we remember to normalize later,
 *
 *      p(outcome | inputs, D) ~ p(inputs | outcome, D) p(outcome | D)
 *
 * The naive aspect is that is assumes inputs are conditionally independent given the outcome, such that,
 *
 *      p(outcome | input_1, input_2, D) ~ p(input_1 | outcome, D)p(input_2 | outcome, D)p(outcome | D)
 *
 * The bayesian aspect is that it integrates out the parameters of the estimated distributions, T, instead of e.g. using maximum likelihood estimates. This makes
 * the model more robust, especially with relatively little data.
 *
 *      p(outcome | inputs, D) ~ ∫ p(inputs | outcome, T) p(outcome | T) p(T | D) dT
 *
 * The classifier breaks down into estimating the prior, p(outcome | D), and a set of posterior predictive likelihoods, p(input_n | outcome, D).
 *  * The prior, p(outcome | D), is (#outcome_c + q)/sum_c(#outcome_c + q), i.e. number of times outcome_c has been seen over how many outcomes we've seen in total, where q is a pseudo count.
 *  * The posterior predictive likelihoods are estimated by named [Feature]s, with names matching the names given in the [Inputs]. See each Feature description.
 *
 * @param priorCounts           Number of times outcomes have been seen, e.g {"positive": 2, "negative": 3}
 * @param textFeatures          The text features used by this model.
 * @param categoricalFeatures   The categorical features used by this model.
 * @param gaussianFeatures      The gaussian features used by this model.
 * @param pseudoCount           Pseudo count of observed outcomes. Is added to all outcome counts in the [priorCounts].
 */
class Model(
        private val priorCounts: Map<Outcome, Int> = mapOf(),
        private val textFeatures: Map<FeatureName, Text> = mapOf(),
        private val categoricalFeatures: Map<FeatureName, Categorical> = mapOf(),
        private val gaussianFeatures: Map<FeatureName, Gaussian> = mapOf(),
        private val pseudoCount: Int = 0
) {

    data class Parameters(
            val text: Map<FeatureName, Multinomial.Parameters> = mapOf(),
            val categorial: Map<FeatureName, Multinomial.Parameters> = mapOf(),
            val gaussian: Map<FeatureName, Gaussian.Parameters> = mapOf()
    )

    // Categorical distribution with dirichlet prior, see https://en.wikipedia.org/wiki/Conjugate_prior#Discrete_distributions
    private val logPrior: Map<String, Double> by lazy {
        priorCounts.mapValues { Math.log(pseudoCount + it.value.toDouble()) } // We don't need to subtract the log(total count), since that is constant w.r.t outcomes, so is normalized away later.
    }

    fun toProto(): Protos.Model {
        return Protos.Model.newBuilder()
                .setModelVersion(Model.serializationVersion)
                .putAllPriorCounts(priorCounts)
                .putAllTextFeatures(textFeatures.mapValues { it.value.toProto() })
                .putAllCategoricalFeatures(categoricalFeatures.mapValues { it.value.toProto() })
                .putAllGaussianFeatures(gaussianFeatures.mapValues { it.value.toProto() })
                .setPseudoCount(pseudoCount)
                .build()
    }

    companion object {
        private const val serializationVersion = 3

        fun fromProto(proto: Protos.Model): Model {
            if (proto.modelVersion != serializationVersion) {
                throw IllegalArgumentException("This version of blayze requires protobuf model version $serializationVersion " +
                        "Attempted to load protobuf with version ${proto.modelVersion}")
            }
            return Model(
                    proto.priorCountsMap,
                    proto.textFeaturesMap.mapValues { Text.fromProto(it.value) },
                    proto.categoricalFeaturesMap.mapValues { Categorical.fromProto(it.value) },
                    proto.gaussianFeaturesMap.mapValues { Gaussian.fromProto(it.value) },
                    proto.pseudoCount
            )
        }
    }

    /**
     * Predicts the outcome of [Inputs] using bayesian naive bayes, e.g. p(outcome | inputs, D) = p(inputs | outcome, D)p(outcome | D)/p(inputs | D), where D is the previously observed data.
     *
     * @return predicted outcomes and their probability, e.g. {"positive": 0.3124, "negative": 0.6876}
     */
    fun predict(inputs: Inputs, parameters: Parameters = Parameters()): Map<Outcome, Double> {
        if (priorCounts.isEmpty()) {
            return mapOf()
        }

        val maps = mutableListOf<Map<Outcome, Double>>()
        maps.add(logPrior)
        maps.addAll(logProbabilities(textFeatures, inputs.text, parameters.text))
        maps.addAll(logProbabilities(categoricalFeatures, inputs.categorical, parameters.categorial))
        maps.addAll(logProbabilities(gaussianFeatures, inputs.gaussian, parameters.gaussian))

        return normalize(sumMaps(maps))
    }

    /**
     * Creates a new model with the [Update] added.
     */
    fun add(update: Update): Model {
        return batchAdd(listOf(update))
    }

    /**
     * Creates a new model with the updates added.
     *
     * @param updates List of observed updates
     * @return new updated Model
     */
    fun batchAdd(updates: List<Update>, parameters: Parameters = Parameters()): Model {
        val newPriorCounts: Map<String, Int> = updates.map { it.outcome }.groupingBy { it }.eachCountTo(priorCounts.toMutableMap())

        val newCategoricalFeatures = updateFeatures(categoricalFeatures, { Categorical() }, updates, { it.categorical }, parameters.categorial)
        val newTextFeatures = updateFeatures(textFeatures, { Text() }, updates, { it.text }, parameters.text)
        val newGaussianFeatures = updateFeatures(gaussianFeatures, { Gaussian() }, updates, { it.gaussian }, parameters.gaussian)

        return Model(newPriorCounts, newTextFeatures, newCategoricalFeatures, newGaussianFeatures)
    }

    private fun <F, V, P> logProbabilities(features: Map<FeatureName, Feature<F, V, P>>, values: Map<FeatureName, V>, parameters: Map<FeatureName, P>): List<Map<Outcome, Double>> {
        val maps = mutableListOf<Map<Outcome, Double>>()
        for ((featureName, featureValue) in values) {
            val feature = features[featureName]
            if (feature != null) {
                val logPosteriorPredictive = feature.logPosteriorPredictive(logPrior.keys, featureValue, parameters[featureName])
                assert(logPosteriorPredictive.keys == logPrior.keys) {
                    "$featureName outcomes did not match logPrior outcomes. Expected ${logPrior.keys}, Actual: ${logPosteriorPredictive.keys}"
                }
                maps.add(logPosteriorPredictive)
            }
        }
        return maps
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


    private fun <V, P, F : Feature<F, V, P>> updateFeatures(
            old: Map<FeatureName, F>,
            creator: () -> F,
            updates: List<Update>,
            extractor: (Inputs) -> Map<FeatureName, V>,
            parameters: Map<FeatureName, P>
    ): Map<FeatureName, F> {

        val outcomes = updates.map { it.outcome }
        val values = updates.map { extractor(it.inputs) }

        val data: Map<FeatureName, List<Pair<Outcome, V>>> = zipOutcomesAndValues(outcomes, values)
        val features = old.toMutableMap()
        for ((name, pairs) in data) {
            val f = features.getOrDefault(name, creator())
            features[name] = f.batchUpdate(pairs, parameters[name])
        }
        return features
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
