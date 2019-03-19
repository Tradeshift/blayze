package com.tradeshift.blayze

import com.tradeshift.blayze.dto.Inputs
import com.tradeshift.blayze.dto.Update
import com.tradeshift.blayze.features.Multinomial
import com.tradeshift.blayze.features.Text
import org.junit.Assert
import org.junit.Test
import kotlin.streams.toList

class ModelIntegrationTest {

    @Test
    fun can_fit_20newsgroup() {
        val train = newsgroup("/20newsgroup_train.txt")
        val model = Model(textFeatures = mapOf("q" to Text(Multinomial(pseudoCount = 0.1)))).batchAdd(train)


        val test = newsgroup("/20newsgroup_test.txt")
        val acc = test
                .parallelStream()
                .map {
                    if (it.outcome == model.predict(it.inputs).maxBy { it.value }?.key) {
                        1.0
                    } else {
                        0.0
                    }
                }
                .toList()
                .average()

        Assert.assertTrue("expected $acc > 0.65", acc > 0.65) // sklearn MultinomialNB with a CountVectorizer gets ~0.646
    }

    @Test
    fun can_fit_iris_dataset() {
        val iris = iris()
        val model = Model().batchAdd(iris)
        val correct = iris.map { if (model.predict(it.inputs).maxBy { it.value }!!.key == it.outcome) 1 else 0 }.sum()
        Assert.assertEquals(143, correct) // sklearn.naive_bayes.GaussianNB gets 144/150 correct.
    }

    private fun newsgroup(fname: String): List<Update> {
        val lines = this::class.java.getResource(fname).readText(Charsets.UTF_8).split("\n")
        val updates = mutableListOf<Update>()

        for (line in lines) {
            val split = line.split(" ".toRegex(), 2).toTypedArray()
            val outcome = split[0]
            var f = Inputs()
            if (split.size == 2) { //some are legit empty
                f = Inputs(mapOf("q" to split[1]))
            }
            updates.add(Update(f, outcome))
        }
        return updates
    }

    private fun iris(): List<Update> {
        val lines = this::class.java.getResource("/iris.csv").readText(Charsets.UTF_8).split("\n")
        val updates = mutableListOf<Update>()
        for (s in lines.drop(1)) {
            val split = s.split(",")
            updates.add(Update(Inputs(
                    gaussian = mapOf(
                            "sepal_length" to split[0].toDouble(),
                            "sepal_width" to split[1].toDouble(),
                            "petal_length" to split[2].toDouble(),
                            "petal_width" to split[3].toDouble()

                    )),
                    split.last())
            )
        }
        return updates
    }


}
