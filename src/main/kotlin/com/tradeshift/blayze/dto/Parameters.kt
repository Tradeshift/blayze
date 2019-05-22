package com.tradeshift.blayze.dto

import com.tradeshift.blayze.Protos
import com.tradeshift.blayze.features.Gaussian
import com.tradeshift.blayze.features.Multinomial

/**
 * Parameters of a model.
 *
 * @param priorPseudoCount Pseudo count of observed outcomes. Is added to all outcome counts in the [priorCounts].
 * @param text parameters of the text features. See [Multinomial.Parameters].
 * @param categorical parameters of the categorical features. See [Multinomial.Parameters].
 * @param gaussian parameters of the gaussian features. See [Gaussian.Parameters].
 */
data class Parameters(
        val priorPseudoCount: Int = 0,
        val text: Map<FeatureName, Multinomial.Parameters> = mapOf(),
        val categorical: Map<FeatureName, Multinomial.Parameters> = mapOf(),
        val gaussian: Map<FeatureName, Gaussian.Parameters> = mapOf()
){
    fun toProto(): Protos.Parameters {
        return Protos.Parameters.newBuilder()
                .setPriorPseudoCount(priorPseudoCount)
                .putAllTextParameters(text.mapValues { it.value.toProto() })
                .putAllCategoricalParameters(categorical.mapValues { it.value.toProto() })
                .putAllGaussianParameters(gaussian.mapValues { it.value.toProto() })
                .build()
    }

    companion object {
        fun fromProto(proto: Protos.Parameters): Parameters {
            return Parameters(proto.priorPseudoCount,
                    proto.textParametersMap.mapValues { Multinomial.Parameters.fromProto(it.value) },
                    proto.categoricalParametersMap.mapValues { Multinomial.Parameters.fromProto(it.value) },
                    proto.gaussianParametersMap.mapValues { Gaussian.Parameters.fromProto(it.value) }
            )
        }
    }
}
