package com.bogdan.sampleml

import android.Manifest.permission.CAMERA
import android.Manifest.permission.READ_EXTERNAL_STORAGE
import android.annotation.SuppressLint
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import com.wonderkiln.camerakit.*
import kotlinx.android.synthetic.main.activity_main.*
import java.io.File
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.Executor
import java.util.concurrent.Executors


class MainActivity : AppCompatActivity() {

    private val REQUEST_IMAGE_CAPTURE: Int = 1
    private val permissionRequestCode: Int = 101
    private val modelPath: String = "mobilenet_quant_v1_224.tflite"
    private val labelPath = "labels.txt"
    private val inputSize = 224

    private var classifier: Classifier? = null
    private var currentPhotoPath: String? = null
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
            if(checkPermission()) {
                cameraView?.toggleFacing()
            } else requestPermission()
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
            arrayOf(READ_EXTERNAL_STORAGE, CAMERA), permissionRequestCode)
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray) {
            when (requestCode) {
            permissionRequestCode -> {
                if (grantResults.isNotEmpty() && grantResults[0] ==
                    PackageManager.PERMISSION_GRANTED &&
                    grantResults[1] == PackageManager.PERMISSION_GRANTED) {
                    takePicture()
                } else {
                    Toast.makeText(this, "Permission Denied", Toast.LENGTH_SHORT).show()
                }
                return
            }
            else -> {
                //Do nothing
            }
        }
    }

    private fun takePicture() {
        val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        val file = createFile()
        val uri = FileProvider.getUriForFile(
            this,
            "com.bogdan.sampleml",
            file
        )
        intent.putExtra(MediaStore.EXTRA_OUTPUT, uri)
        startActivityForResult(intent, REQUEST_IMAGE_CAPTURE)
    }

    @SuppressLint("SimpleDateFormat")
    private fun createFile(): File {
        val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss").format(Date())
        val storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES)
        return File.createTempFile(
            "JPEG_${timeStamp}",
            ".jpg",
            storageDir
        ).apply {
            currentPhotoPath = absolutePath
        }
    }
}
