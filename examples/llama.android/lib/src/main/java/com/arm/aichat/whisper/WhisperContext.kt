package com.arm.aichat.whisper

import android.os.Build
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors

private const val LOG_TAG = "WhisperContext"

class WhisperContext private constructor(private var ptr: Long) {
    private val scope: CoroutineScope = CoroutineScope(
        Executors.newSingleThreadExecutor().asCoroutineDispatcher()
    )

    suspend fun transcribe(audioData: FloatArray): String = withContext(scope.coroutineContext) {
        require(ptr != 0L) { "Whisper context not initialized" }
        val numThreads = WhisperCpuConfig.preferredThreadCount
        WhisperLib.fullTranscribe(ptr, numThreads, audioData)
        val textCount = WhisperLib.getTextSegmentCount(ptr)
        buildString {
            for (i in 0 until textCount) {
                append(WhisperLib.getTextSegment(ptr, i))
            }
        }
    }

    suspend fun transcribeWithTimestamps(audioData: FloatArray): List<WhisperSegment> =
        withContext(scope.coroutineContext) {
            require(ptr != 0L)
            val numThreads = WhisperCpuConfig.preferredThreadCount
            WhisperLib.fullTranscribe(ptr, numThreads, audioData)
            val count = WhisperLib.getTextSegmentCount(ptr)
            (0 until count).map { i ->
                WhisperSegment(
                    text = WhisperLib.getTextSegment(ptr, i),
                    t0 = WhisperLib.getTextSegmentT0(ptr, i),
                    t1 = WhisperLib.getTextSegmentT1(ptr, i)
                )
            }
        }

    suspend fun release() = withContext(scope.coroutineContext) {
        if (ptr != 0L) {
            WhisperLib.freeContext(ptr)
            ptr = 0
        }
    }

    protected fun finalize() {
        kotlinx.coroutines.runBlocking { release() }
    }

    companion object {
        fun createContextFromFile(filePath: String): WhisperContext {
            val ptr = WhisperLib.initContext(filePath)
            if (ptr == 0L) {
                throw RuntimeException("Failed to create whisper context from $filePath")
            }
            return WhisperContext(ptr)
        }

        fun getSystemInfo(): String = WhisperLib.getSystemInfo()
    }
}

data class WhisperSegment(
    val text: String,
    val t0: Long,
    val t1: Long
)

private object WhisperCpuConfig {
    val preferredThreadCount: Int
        get() = Runtime.getRuntime().availableProcessors().coerceIn(1, 4)
}

private class WhisperLib {
    companion object {
        init {
            Log.d(LOG_TAG, "Primary ABI: ${Build.SUPPORTED_ABIS[0]}")
            Log.d(LOG_TAG, "Loading libai-chat.so (whisper bundled)")
            System.loadLibrary("ai-chat")
        }

        external fun initContext(modelPath: String): Long
        external fun freeContext(contextPtr: Long)
        external fun fullTranscribe(contextPtr: Long, numThreads: Int, audioData: FloatArray)
        external fun getTextSegmentCount(contextPtr: Long): Int
        external fun getTextSegment(contextPtr: Long, index: Int): String
        external fun getTextSegmentT0(contextPtr: Long, index: Int): Long
        external fun getTextSegmentT1(contextPtr: Long, index: Int): Long
        external fun getSystemInfo(): String
    }
}
