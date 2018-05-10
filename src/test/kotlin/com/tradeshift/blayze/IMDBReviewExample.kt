package com.tradeshift.blayze

import com.tradeshift.blayze.dto.Inputs
import com.tradeshift.blayze.dto.Update
import com.tradeshift.blayze.features.Multinomial
import com.tradeshift.blayze.features.Text
import okhttp3.OkHttpClient
import okhttp3.Request
import org.apache.commons.compress.archivers.tar.TarArchiveEntry
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream
import org.apache.commons.compress.utils.IOUtils
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException

val dataFolder = "./data"
val dataFile = dataFolder + "/aclImdb_v1.tar.gz"
val dataFileExtractDir = dataFolder + "/aclImdb"

fun main(args: Array<String>) {
    // loading data
    downloadData()
    val trainTest = getTrainTest()
    val train = trainTest.first.toList()
    val test = trainTest.second.toList()
    println("Train size: ${train.size}; test size: ${test.size}")
    // train
    val model = Model(textFeatures = mapOf("review" to Text(Multinomial(pseudoCount = 1.0))))  // Text(Multinomial(pseudoCount = 1.0) , useBigram = true)
            .batchAdd(train)

    // test
    val acc = test
            .map {
                if (it.outcome == model.predict(it.inputs).maxBy { it.value }?.key)  1.0 else 0.0
            }
            .average()

    println("Finished, test acc=$acc")
}

fun getTrainTest(): Pair<Sequence<Update>, Sequence<Update>> {
    val trainPos = File(dataFileExtractDir + "/train/pos").walk().filter { it.isFile }.map {
        Update(Inputs(text = mapOf("review" to it.readText())), outcome = "pos")
    }
    val trainNeg = File(dataFileExtractDir + "/train/neg").walk().filter { it.isFile }.map {
        Update(Inputs(text = mapOf("review" to it.readText())), outcome = "neg")
    }
    val train = trainPos + trainNeg

    val testPos = File(dataFileExtractDir + "/test/pos").walk().filter { it.isFile }.map {
        Update(Inputs(text = mapOf("review" to it.readText())), outcome = "pos")
    }
    val testNeg = File(dataFileExtractDir + "/test/neg").walk().filter { it.isFile }.map {
        Update(Inputs(text = mapOf("review" to it.readText())), outcome = "neg")
    }
    val test = testPos + testNeg

    return train to test
}

fun downloadData() {
    val client = OkHttpClient()
    val request = Request.Builder().url("http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz").build()
    val response = client.newCall(request).execute()
    if (!response.isSuccessful) {
        throw IOException("Failed to download file: " + response)
    }

    File("./data").mkdirs()

    if (!File(dataFile).isFile) {
        println("Downloading IMDB review Data to $dataFile ...")
        val fos = FileOutputStream(dataFile)
        fos.write(response.body()?.bytes())
        fos.close()
        println("IMDB review Data downloaded to $dataFile")
    }
    if (!File(dataFileExtractDir).isDirectory) extractData()
}

/**
 * https://memorynotfound.com/java-tar-example-compress-decompress-tar-tar-gz-files/
 */
fun extractData() {
    TarArchiveInputStream(GzipCompressorInputStream(FileInputStream(dataFile))).use { fin ->
        println("Extracting IMDB review data to $dataFolder")
        var entry: TarArchiveEntry? = null
        while (fin.nextTarEntry.let { entry = it; it != null }) {
            if (entry!!.isDirectory) {
                continue
            }
            val curfile = File(dataFolder, entry!!.name)
            val parent = curfile.parentFile
            if (!parent.exists()) {
                parent.mkdirs()
            }
            IOUtils.copy(fin, FileOutputStream(curfile))
        }
        println("IMDB review data extracted to $dataFolder")
    }
}