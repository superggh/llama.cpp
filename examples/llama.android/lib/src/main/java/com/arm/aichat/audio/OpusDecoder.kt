package com.arm.aichat.audio

import android.util.Log

class OpusDecoder(sampleRate: Int = 16000, channels: Int = 1) {
    private var handle: Long = 0

    init {
        System.loadLibrary("ai-chat")
        handle = nativeCreate(sampleRate, channels)
        if (handle == 0L) throw RuntimeException("Failed to create OpusDecoder")
    }

    fun decode(opusData: ByteArray, frameSize: Int = 320): ShortArray {
        val pcmOut = ShortArray(frameSize)
        val decoded = nativeDecode(handle, opusData, opusData.size, pcmOut, frameSize)
        if (decoded < 0) {
            Log.w("OpusDecoder", "Decode failed: $decoded")
            return ShortArray(0)
        }
        return if (decoded == frameSize) pcmOut else pcmOut.copyOf(decoded)
    }

    fun release() {
        if (handle != 0L) {
            nativeDestroy(handle)
            handle = 0
        }
    }

    protected fun finalize() {
        release()
    }

    private external fun nativeCreate(sampleRate: Int, channels: Int): Long
    private external fun nativeDestroy(handle: Long)
    private external fun nativeDecode(handle: Long, opusData: ByteArray, opusLen: Int, pcmOut: ShortArray, frameSize: Int): Int
}
