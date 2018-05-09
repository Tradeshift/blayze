package com.tradeshift.blayze.dto

import com.tradeshift.blayze.Protos

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
 *          }
 *      }
 */
data class Inputs(
        val text: Map<FeatureName, FeatureValue> = mapOf(),
        val categorical: Map<FeatureName, FeatureValue> = mapOf(),
        val gaussian: Map<FeatureName, Double> = mapOf()
) {
    fun toProto(): Protos.Inputs {
        return Protos.Inputs.newBuilder()
                .putAllText(text)
                .putAllCategorical(categorical)
                .putAllGaussian(gaussian)
                .build()
    }

    companion object {
        fun fromProto(proto: Protos.Inputs): Inputs {
            return Inputs(proto.textMap, proto.categoricalMap, proto.gaussianMap)
        }
    }
}
