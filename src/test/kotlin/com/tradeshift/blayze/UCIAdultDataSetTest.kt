import com.tradeshift.blayze.Model
import com.tradeshift.blayze.dto.Inputs
import com.tradeshift.blayze.dto.Update
import org.junit.Assert
import org.junit.Test
import kotlin.streams.toList

class UCIAdultDataSetTest {

    enum class FeatureType {
        Category, Gaussian
    }

    val FEATURE_COLUMN_NAMES = arrayListOf(
            "age" to FeatureType.Gaussian,            // 0
            "workclass" to FeatureType.Category,      // 1
            "fnlwgt" to FeatureType.Gaussian,         // 2, final weight
            "education" to FeatureType.Category,      // 3
            "education-num" to FeatureType.Gaussian,  // 4
            "marital-status" to FeatureType.Category, // 5
            "occupation" to FeatureType.Category,     // 6
            "relationship" to FeatureType.Category,   // 7
            "race" to FeatureType.Category,           // 8
            "sex" to FeatureType.Category,            // 9
            "capital-gain" to FeatureType.Gaussian,   // 10
            "capital-loss" to FeatureType.Gaussian,   // 11
            "hours-per-week" to FeatureType.Gaussian, // 12
            "native-country" to FeatureType.Category // 13
    )

    @Test
    fun can_fit_uci_adult_dataset() {
        val train = uciAdult("adult.train.txt")
        val model = Model().batchAdd(train)

        val test = uciAdult("adult.test.txt")
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

        println(acc)
        Assert.assertTrue("expected $acc > 0.83", acc > 0.83)
    }

    fun uciAdult(fname: String): List<Update> {
        val lines = this::class.java.getResource(fname).readText(Charsets.UTF_8).split("\n")
        val updates = mutableListOf<Update>()

        for (line in lines) {
            val split = line.split(",".toRegex()).toTypedArray()
            if (split.size < 14) continue

            var outcome = "Unknown"
            val categoryFeature = mutableMapOf<String, String>()
            val gaussianFeature = mutableMapOf<String, Double>()

            for ((i, item_raw) in split.iterator().withIndex()) {
                val item = item_raw.trim()

                if (i < 14) {
                    if (item.isEmpty() || item == "?") {
                        continue
                    }
                    val (featureName, featureType) = FEATURE_COLUMN_NAMES[i]

                    if (featureType == FeatureType.Category) {
                        categoryFeature.put(featureName, item)
                    } else if (featureType == FeatureType.Gaussian) {
                        val itemValue = item.toDoubleOrNull()
                        if (itemValue != null) {
                            gaussianFeature.put(featureName, itemValue)
                        }
                    }

                } else {
                    outcome = item.replace(".", "")
                }
            }

            val f = Inputs(
                    categorical = categoryFeature,
                    gaussian = gaussianFeature
            )

            updates.add(Update(f, outcome))
        }
        return updates
    }
}
