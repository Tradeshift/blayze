package com.tradeshift.blayze.features

import com.tradeshift.blayze.features.Gaussian.StreamingEstimator
import org.junit.Assert.*
import org.junit.Test
import kotlin.math.sqrt

class StreamingEstimatorTest {

    @Test
    fun when_given_a_single_value_then_gives_variance_zero() {
        val estimator = StreamingEstimator(5.3)
        assertEquals(5.3, estimator.mean, 0.0)
        assertEquals(0.0, estimator.stdev, 0.0)
    }

    @Test
    fun when_given_multiple_values_then_compute_correct_mean_and_variance() {
        val estimator = StreamingEstimator(5.3).add(2.6).add(1.1)
        assertEquals(3.0, estimator.mean, 0.0)
        assertEquals(sqrt(4.53), estimator.stdev, 0.0)
    }
}