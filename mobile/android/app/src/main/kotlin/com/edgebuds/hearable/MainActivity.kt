package com.edgebuds.hearable

import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import java.io.File
import java.util.concurrent.Executors

class MainActivity : FlutterActivity() {
    private val channelName = "com.edgebuds.hearable/bluetooth_sco"
    private lateinit var scoAudioManager: BluetoothScoAudioManager
    private val ioExecutor = Executors.newSingleThreadExecutor()

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        scoAudioManager = BluetoothScoAudioManager(applicationContext)

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, channelName)
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "startBluetoothSco" -> {
                        val status = scoAudioManager.startBluetoothScoRouting()
                        result.success(status)
                    }
                    "recordCoughCheckIn" -> {
                        val durationSec = call.argument<Int>("durationSeconds") ?: 6
                        val outFile = File(cacheDir, "active_checkin_16khz_mono.wav")
                        ioExecutor.execute {
                            try {
                                val recorded = scoAudioManager.recordActiveCoughCheckIn(
                                    outFile,
                                    durationSec
                                )
                                runOnUiThread {
                                    result.success(recorded.absolutePath)
                                }
                            } catch (e: Exception) {
                                runOnUiThread {
                                    result.error("RECORD_ERROR", e.message, null)
                                }
                            }
                        }
                    }
                    "stopBluetoothSco" -> {
                        scoAudioManager.stopBluetoothScoRouting()
                        result.success(true)
                    }
                    else -> result.notImplemented()
                }
            }
    }
}
