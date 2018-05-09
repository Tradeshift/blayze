package com.tradeshift.blayze.dto

import com.tradeshift.blayze.Protos

/**
 * Outcome and [Inputs], e.g. (in json)
 *
 *      {
 *          "outcome": "spam",
 *          "inputs": {
 *              "text": {
 *                  "subject": "Attention, is it true?",
 *                  "body": "Good day dear beneficiary. This is Secretary to president of Benin republic is writing this email ..."
 *              },
 *              "categorical": {
 *                  "sender": "WWW.@galaxy.ocn.ne.jp",
 *                  "mailed-by": "galaxy.ocn.ne.jp",
 *                  "reply-to": "njokua35@gmail.com"
 *              },
 *              "gaussian": {
 *                  "n_words": 482
 *              }
 *          }
 *      }
 */
data class Update(
        val inputs: Inputs,
        val outcome: Outcome
) {
    fun toProto(): Protos.Update {
        return Protos.Update.newBuilder()
                .setInputs(inputs.toProto())
                .setOutcome(outcome)
                .build()

    }

    companion object {
        fun fromProto(proto: Protos.Update): Update {
            return Update(Inputs.fromProto(proto.inputs), proto.outcome)
        }
    }
}