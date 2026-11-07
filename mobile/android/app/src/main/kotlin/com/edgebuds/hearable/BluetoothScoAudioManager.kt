package com.edgebuds.hearable

import android.annotation.SuppressLint
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.media.AudioDeviceInfo
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import java.io.File
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Gerenciador Nativo de Áudio para Samsung Galaxy Buds (Buds Core / Buds 2).
 *
 * Implementa a Seção 2 e Seção 3.C do PRD/Spec:
 * - Força o roteamento do microfone via protocolo Bluetooth SCO (`TYPE_BLUETOOTH_SCO`)
 *   usando `startBluetoothSco()` e `setBluetoothScoOn(true)`.
 * - Captura áudio em 16.000 Hz (16 kHz), 1 canal (Mono), PCM 16-bit (codec mSBC).
 */
class BluetoothScoAudioManager(private val context: Context) {

    companion object {
        const val SAMPLE_RATE_HZ = 16000
        const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
        const val AUDIO_ENCODING = AudioFormat.ENCODING_PCM_16BIT
    }

    private val audioManager: AudioManager =
        context.getSystemService(Context.AUDIO_SERVICE) as AudioManager

    @Volatile
    private var isRecording: Boolean = false

    /**
     * Inicia o canal Bluetooth SCO e seleciona explicitamente a entrada `TYPE_BLUETOOTH_SCO`.
     */
    fun startBluetoothScoRouting(): Map<String, Any> {
        audioManager.mode = AudioManager.MODE_IN_COMMUNICATION
        @Suppress("DEPRECATION")
        audioManager.startBluetoothSco()
        @Suppress("DEPRECATION")
        audioManager.isBluetoothScoOn = true

        val scoDevice = findBluetoothScoInputDevice()
        return mapOf(
            "sco_active" to audioManager.isBluetoothScoOn,
            "device_found" to (scoDevice != null),
            "device_product_name" to (scoDevice?.productName?.toString() ?: "Samsung Galaxy Buds SCO"),
            "sample_rate_hz" to SAMPLE_RATE_HZ,
            "channels" to 1
        )
    }

    /**
     * Busca o dispositivo de entrada `AudioDeviceInfo.TYPE_BLUETOOTH_SCO` conectado.
     */
    fun findBluetoothScoInputDevice(): AudioDeviceInfo? {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            val inputs = audioManager.getDevices(AudioManager.GET_DEVICES_INPUTS)
            for (device in inputs) {
                if (device.type == AudioDeviceInfo.TYPE_BLUETOOTH_SCO) {
                    return device
                }
            }
        }
        return null
    }

    /**
     * Executa a gravação do Check-in Ativo (5 a 10 segundos) diretamente do microfone
     * Bluetooth SCO em um arquivo WAV PCM 16-bit, 16 kHz Mono.
     */
    @SuppressLint("MissingPermission")
    fun recordActiveCoughCheckIn(outputWavFile: File, durationSeconds: Int = 6): File {
        val minBufSize = AudioRecord.getMinBufferSize(
            SAMPLE_RATE_HZ,
            CHANNEL_CONFIG,
            AUDIO_ENCODING
        ).coerceAtLeast(SAMPLE_RATE_HZ * 2)

        val audioRecord = AudioRecord(
            MediaRecorder.AudioSource.VOICE_COMMUNICATION,
            SAMPLE_RATE_HZ,
            CHANNEL_CONFIG,
            AUDIO_ENCODING,
            minBufSize
        )

        // Força a entrada via TYPE_BLUETOOTH_SCO (Galaxy Buds) quando disponível
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            findBluetoothScoInputDevice()?.let { scoDevice ->
                audioRecord.setPreferredDevice(scoDevice)
            }
        }

        val totalSamplesToRead = SAMPLE_RATE_HZ * durationSeconds
        val pcmBuffer = ShortArray(1024)
        var samplesReadTotal = 0

        FileOutputStream(outputWavFile).use { fos ->
            // Reserva 44 bytes para o cabeçalho RIFF/WAVE
            fos.write(ByteArray(44))
            audioRecord.startRecording()
            isRecording = true

            while (isRecording && samplesReadTotal < totalSamplesToRead) {
                val readCount = audioRecord.read(pcmBuffer, 0, pcmBuffer.size)
                if (readCount > 0) {
                    val byteBuf = ByteBuffer.allocate(readCount * 2).order(ByteOrder.LITTLE_ENDIAN)
                    for (i in 0 until readCount) {
                        byteBuf.putShort(pcmBuffer[i])
                    }
                    fos.write(byteBuf.array())
                    samplesReadTotal += readCount
                }
            }

            audioRecord.stop()
            audioRecord.release()
            isRecording = false
        }

        writeWaveHeader(outputWavFile, samplesReadTotal * 2, SAMPLE_RATE_HZ, 1, 16)
        return outputWavFile
    }

    /**
     * Encerra o roteamento Bluetooth SCO e restaura o modo de áudio padrão.
     */
    fun stopBluetoothScoRouting() {
        isRecording = false
        @Suppress("DEPRECATION")
        audioManager.isBluetoothScoOn = false
        @Suppress("DEPRECATION")
        audioManager.stopBluetoothSco()
        audioManager.mode = AudioManager.MODE_NORMAL
    }

    private fun writeWaveHeader(
        file: File,
        pcmDataByteCount: Int,
        sampleRate: Int,
        channels: Int,
        bitsPerSample: Int
    ) {
        val byteRate = sampleRate * channels * (bitsPerSample / 8)
        val blockAlign = (channels * (bitsPerSample / 8)).toShort()
        val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)

        header.put("RIFF".toByteArray(Charsets.US_ASCII))
        header.putInt(36 + pcmDataByteCount)
        header.put("WAVE".toByteArray(Charsets.US_ASCII))
        header.put("fmt ".toByteArray(Charsets.US_ASCII))
        header.putInt(16) // Subchunk1Size (PCM)
        header.putShort(1) // AudioFormat (1 = PCM)
        header.putShort(channels.toShort())
        header.putInt(sampleRate)
        header.putInt(byteRate)
        header.putShort(blockAlign)
        header.putShort(bitsPerSample.toShort())
        header.put("data".toByteArray(Charsets.US_ASCII))
        header.putInt(pcmDataByteCount)

        RandomAccessFile(file, "rw").use { raf ->
            raf.seek(0)
            raf.write(header.array())
        }
    }
}
