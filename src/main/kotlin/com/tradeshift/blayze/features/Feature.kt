package com.tradeshift.blayze.features

import com.tradeshift.blayze.dto.Outcome

/**
 * A feature for use in a naive bayes classifier which computes the likelihood of p(input|outcome), where input is of type [V]
 *
 * Implementations must use their own class for the type [F]
 */
interface Feature<F,V> {

    /**
     *  The log probability for each of the outcomes, given the value. The log probability need only be correct up to an additive constant, i.e. constant terms can be dropped.
     */
    fun logProbability(outcomes: Set<Outcome>, value: V): Map<Outcome, Double>

    /**
     * Create a new feature of type [F] from this feature updated with the updates. Must be the same type as implementing class.
     */
    fun batchUpdate(updates: List<Pair<Outcome, V>>): F
}