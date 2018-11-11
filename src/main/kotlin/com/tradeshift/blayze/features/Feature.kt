package com.tradeshift.blayze.features

import com.tradeshift.blayze.dto.Outcome

/**
 * A feature for use in a bayesian naive bayes classifier with input of type [V].
 *
 * Implementations must use their own class for the type [F]
 */
interface Feature<F,V> {

    /**
     *  The log posterior predictive probability of the value, conditioned on the observed data, D, and each outcome.
     *  $p(value | outcome, D) = int_{t} p(value | outcome, t) p(t | D) dt$, where t are the parameters of the distribution, which are integrated out.
     *
     *  See https://en.wikipedia.org/wiki/Posterior_predictive_distribution, and https://en.wikipedia.org/wiki/Conjugate_prior
     *  Need only be correct up to an additive constant.
     */
    fun logPosteriorPredictive(outcomes: Set<Outcome>, value: V): Map<Outcome, Double>

    /**
     * Create a new feature of type [F] from this feature updated with the updates. Must be the same type as implementing class.
     */
    fun batchUpdate(updates: List<Pair<Outcome, V>>): F
}