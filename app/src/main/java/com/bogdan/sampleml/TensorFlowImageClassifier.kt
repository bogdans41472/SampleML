package com.bogdan.sampleml

import android.annotation.SuppressLint
import android.content.res.AssetManager
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.io.BufferedReader
import java.io.FileInputStream
import java.io.IOException
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteBuffer.allocateDirect
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.*
import kotlin.collections.ArrayList
import kotlin.experimental.and
import kotlin.math.min


class TensorFlowImageClassifier : Classifier {

    private val maxResults = 3
    private val batchSize = 1
    private val pixelSize = 3
    private val threshold = 0.1f

    private val imageMean = 128
    private val imageStd = 128.0f

    private var interpreter: Interpreter? = null
    private var inputSize: Int = 0
    private var labelList: List<String>? = null
    private var quant: Boolean = false

    fun create (assetManager: AssetManager, modelPath: String,
                labelPath: String, inputSize: Int, quant: Boolean): Classifier {
            val classifier = TensorFlowImageClassifier()
            classifier.interpreter = Interpreter(classifier.loadModelFile(assetManager, modelPath),
                    Interpreter.Options())
            classifier.labelList = classifier.loadLabelList(assetManager, labelPath)
            classifier.inputSize = inputSize
            classifier.quant = quant

            return classifier
    }

    override fun recognizeImage(bitmap: Bitmap): List<Classifier.Recognition> {
        val byteBuffer = convertBitmapToByteBuffer(bitmap)
        return if (quant) {
            val result = Array(1) { ByteArray(labelList!!.size) }
            interpreter!!.run(byteBuffer, result)
            getSortedResultByte(result)
        } else {
            val result = Array(1) { FloatArray(labelList!!.size) }
            interpreter!!.run(byteBuffer, result)
            getSortedResultFloat(result)
        }
    }

    override fun close() {
        interpreter!!.close()
        interpreter = null
    }

    private fun loadModelFile(assetManager: AssetManager, modelPath: String): MappedByteBuffer {
        val fileDescriptor = assetManager.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun loadLabelList(assetManager: AssetManager, labelPath: String): List<String> {
        val labelList = ArrayList<String>()
        val reader = BufferedReader(InputStreamReader(assetManager.open(labelPath)))
        val line = reader.readLine()

        while(line != null) {
            labelList.add(line)
        }

        reader.close()
        return labelList
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer: ByteBuffer = if (quant) {
            allocateDirect(batchSize * inputSize * inputSize * pixelSize)
        } else {
            allocateDirect(4 * batchSize * inputSize * inputSize * pixelSize)
        }

        byteBuffer.order(ByteOrder.nativeOrder())
        val intValues = IntArray(inputSize * inputSize)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        var pixel = 0
        for (i in 0 until inputSize) {
            for (j in 0 until inputSize) {
                val `val` = intValues[pixel++]
                if (quant) {
                    byteBuffer.put((`val` shr 16 and 0xFF).toByte())
                    byteBuffer.put((`val` shr 8 and 0xFF).toByte())
                    byteBuffer.put((`val` and 0xFF).toByte())
                } else {
                    byteBuffer.putFloat(((`val` shr 16 and 0xFF) - imageMean) / imageStd)
                    byteBuffer.putFloat(((`val` shr 8 and 0xFF) - imageMean) / imageStd)
                    byteBuffer.putFloat(((`val` and 0xFF) - imageMean) / imageStd)
                }

            }
        }
        return byteBuffer
    }

    private fun getSortedResultByte(labelProbArray: Array<ByteArray>): List<Classifier.Recognition> {

        val pq = PriorityQueue(
            maxResults,
            Comparator<Classifier.Recognition> { lhs, rhs ->
                (rhs.confidence!!).compareTo(lhs.confidence!!)
            })

        for (i in labelList!!.indices) {
            val confidence = (labelProbArray[0][i] and 0xff.toByte()) / 255.0f
            if (confidence > threshold) {
                pq.add(
                    Classifier.Recognition(
                        "" + i,
                        if (labelList!!.size > i) labelList!![i] else "unknown",
                        confidence, quant
                    )
                )
            }
        }

        val recognitions = ArrayList<Classifier.Recognition>()
        val recognitionsSize = min(pq.size, maxResults)
        for (i in 0 until recognitionsSize) {
            recognitions.add(pq.poll())
        }

        return recognitions
    }

    @SuppressLint("DefaultLocale")
    private fun getSortedResultFloat(labelProbArray: Array<FloatArray>): List<Classifier.Recognition> {

        val pq = PriorityQueue(
            maxResults,
            Comparator<Classifier.Recognition> { lhs, rhs ->
                (rhs.confidence!!).compareTo(lhs.confidence!!)
            })

        for (i in labelList!!.indices) {
            val confidence = labelProbArray[0][i]
            if (confidence > threshold) {
                pq.add(
                    Classifier.Recognition(
                        "" + i,
                        if (labelList!!.size > i) labelList!![i] else "unknown",
                        confidence, quant
                    )
                )
            }
        }

        val recognitions = ArrayList<Classifier.Recognition>()
        val recognitionsSize = min(pq.size, maxResults)
        for (i in 0 until recognitionsSize) {
            if (pq.poll() != null) {
                recognitions.add(pq.poll())
            }
        }

        return recognitions
    }


}
