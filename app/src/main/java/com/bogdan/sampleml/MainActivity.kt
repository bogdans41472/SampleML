package com.bogdan.sampleml

import android.Manifest.permission.CAMERA
import android.Manifest.permission.READ_EXTERNAL_STORAGE
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.wonderkiln.camerakit.*
import kotlinx.android.synthetic.main.activity_main.*
import java.util.concurrent.Executor
import java.util.concurrent.Executors


class MainActivity : AppCompatActivity() {

    private val PERMISSION_REQUEST_CODE: Int = 1
    private val modelPath: String = "mobilenet_quant_v1_224.tflite"
    private val labelPath = "labels.txt"
    private var classifier: Classifier? = null
    private val inputSize = 224
    private var cameraView: CameraView? = null
    private var executor: Executor = Executors.newSingleThreadExecutor()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        cameraView = camera
        cameraView?.addCameraKitListener(object : CameraKitEventListener {
            override fun onVideo(p0: CameraKitVideo?) {
                cameraView = camera
            }

            override fun onEvent(p0: CameraKitEvent?) {
            }

            override fun onError(p0: CameraKitError?) {
            }

            override fun onImage(cameraKitImage: CameraKitImage) {
                var bitmap = cameraKitImage.bitmap
                bitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, false)
                iamgeView_results.setImageBitmap(bitmap)

                val results = classifier?.recognizeImage(bitmap)

                textView_results.text = results.toString()
            }
        })

        button_toggleCamera.setOnClickListener {
            cameraView?.toggleFacing()
        }
        button_toggleObjectDetection.setOnClickListener{
            cameraView?.captureImage()
        }

        initTensorFlowAndLoadModel()

    }

    private fun initTensorFlowAndLoadModel() {
        executor.execute {
            try {
                classifier = TensorFlowImageClassifier().create(
                    assets,
                    modelPath,
                    labelPath,
                    inputSize,
                    true
                )
                makeButtonVisible()
            } catch (e: Exception) {
                throw RuntimeException("Error initializing TensorFlow!", e)
            }
        }
    }

    private fun makeButtonVisible() {
        button_toggleObjectDetection.visibility = View.VISIBLE
    }

    private fun checkPermission(): Boolean {
        return (ContextCompat.checkSelfPermission(this, CAMERA) ==
                PackageManager.PERMISSION_GRANTED && ContextCompat.checkSelfPermission(this,
            READ_EXTERNAL_STORAGE
        ) == PackageManager.PERMISSION_GRANTED)
    }

    private fun requestPermission() {
        ActivityCompat.requestPermissions(this,
            arrayOf(READ_EXTERNAL_STORAGE, CAMERA), PERMISSION_REQUEST_CODE)
    }
}
