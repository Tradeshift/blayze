package com.tradeshift.blayze.features

import com.tradeshift.blayze.dto.Outcome

/**
 * A feature for use in a bayesian naive bayes classifier with input of type [V] and parameters of type [P].
 *
 * Implementations must use their own class for the type [F]
 */
interface Feature<F, V, P> {

    /**
     * The log posterior predictive probability of the value, conditioned on the observed data, D, and each outcome.
     * $p(value | outcome, D) = int_{t} p(value | outcome, t) p(t | D) dt$, where t are the parameters of the distribution, which are integrated out.
     *
     * See https://en.wikipedia.org/wiki/Posterior_predictive_distribution, and https://en.wikipedia.org/wiki/Conjugate_prior
     * Need only be correct up to an additive constant.
     *
     * If [parameters] is null, the default parameters of the feature will be used. See [withParameters].
     */
    fun logPosteriorPredictive(outcomes: Set<Outcome>, value: V, parameters: P? = null): Map<Outcome, Double>

    /**
     * Returns a new feature updated with the updates.
     * If [parameters] is null, the default parameters of the feature will be used. See [withParameters].
     */
    fun batchUpdate(updates: List<Pair<Outcome, V>>, parameters: P? = null): F

    /**
     * Returns a new feature with [parameters] as the default parameters.
     *
     * Subsequent calls to [logPosteriorPredictive] and [batchUpdate] on this new feature without explicit parameters, will use these default parameters.
     */
    fun withParameters(parameters: P): F
}
