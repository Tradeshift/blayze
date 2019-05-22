package com.tradeshift.blayze.dto

import com.tradeshift.blayze.Protos
import com.tradeshift.blayze.features.Gaussian
import com.tradeshift.blayze.features.Multinomial

typealias FeatureName = String
typealias FeatureValue = String
typealias Outcome = String

/**
 * Named inputs, e.g. (in json)
 *
 *      {
 *          "text": {
 *              "subject": "Attention, is it true?",
 *              "body": "Good day dear beneficiary. This is Secretary to president of Benin republic is writing this email ..."
 *          },
 *          "categorical": {
 *              "sender": "WWW.@galaxy.ocn.ne.jp",
 *              "mailed-by": "galaxy.ocn.ne.jp",
 *              "reply-to": "njokua35@gmail.com"
 *          },
 *          "gaussian": {
 *              "n_words": 482
 *          },
 *          "parameters" : {
 *              "textParameters": {
 *                  "subject": {
 *                      "includeFeatureProbability": 0.5,
 *                      "pseudoCount": 1
 *                  }
 *              }
 *          }
 *      }
 */

data class Inputs(
        val text: Map<FeatureName, FeatureValue> = mapOf(),
        val categorical: Map<FeatureName, FeatureValue> = mapOf(),
        val gaussian: Map<FeatureName, Double> = mapOf(),
        val parameters: Parameters = Parameters()

) {
    fun toProto(): Protos.Inputs {
        return Protos.Inputs.newBuilder()
                .putAllText(text)
                .putAllCategorical(categorical)
                .putAllGaussian(gaussian)
                .setParameters(parameters.toProto())
                .build()
    }

    companion object {
        fun fromProto(proto: Protos.Inputs): Inputs {
            return Inputs(
                    proto.textMap,
                    proto.categoricalMap,
                    proto.gaussianMap,
                    Parameters.fromProto(proto.parameters)
            )
        }
    }
}
