package com.arm.aichat.audio

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import java.nio.ByteBuffer
import java.nio.ByteOrder

private const val SAMPLE_RATE = 16000
private const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
private const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
private const val BYTES_PER_SAMPLE = 2

class AudioRecorder {
    private var audioRecord: AudioRecord? = null
    private var isRecording = false

    @SuppressLint("MissingPermission")
    fun start(): Flow<ShortArray> = flow {
        val minBuffer = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT)
        val bufferSize = minBuffer.coerceAtLeast(SAMPLE_RATE * BYTES_PER_SAMPLE)
        val buffer = ShortArray(bufferSize / BYTES_PER_SAMPLE)

        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            SAMPLE_RATE,
            CHANNEL_CONFIG,
            AUDIO_FORMAT,
            bufferSize
        ).apply {
            startRecording()
        }
        isRecording = true

        while (isRecording) {
            val read = audioRecord?.read(buffer, 0, buffer.size) ?: 0
            if (read > 0) {
                emit(buffer.copyOf(read))
            }
        }
    }

    fun stop() {
        isRecording = false
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null
    }
}

fun ShortArray.toFloatArray(): FloatArray {
    return FloatArray(size) { i ->
        (this[i] / 32768.0f).coerceIn(-1f, 1f)
    }
}

fun decodePcm16ToFloatArray(bytes: ByteArray): FloatArray {
    val shortBuffer = ByteBuffer.wrap(bytes)
        .order(ByteOrder.LITTLE_ENDIAN)
        .asShortBuffer()
    val shorts = ShortArray(shortBuffer.remaining())
    shortBuffer.get(shorts)
    return shorts.toFloatArray()
}
