package com.arm.aichat.funasr

import android.os.Build
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors

private const val LOG_TAG = "FunasrContext"

class FunasrContext private constructor(private var ptr: Long) {
    private val scope: CoroutineScope = CoroutineScope(
        Executors.newSingleThreadExecutor().asCoroutineDispatcher()
    )

    suspend fun transcribe(audioData: FloatArray): String = withContext(scope.coroutineContext) {
        require(ptr != 0L) { "FunASR context not initialized" }
        FunasrLib.transcribe(ptr, audioData)
    }

    suspend fun release() = withContext(scope.coroutineContext) {
        if (ptr != 0L) {
            FunasrLib.free(ptr)
            ptr = 0
        }
    }

    protected fun finalize() {
        kotlinx.coroutines.runBlocking { release() }
    }

    companion object {
        fun create(encoderPath: String, llmPath: String, useSharedModel: Boolean = false): FunasrContext {
            val ptr = FunasrLib.init(encoderPath, llmPath, useSharedModel)
            if (ptr == 0L) {
                throw RuntimeException("Failed to create FunASR context")
            }
            Log.i(LOG_TAG, "FunASR context created, encoder=$encoderPath llm=$llmPath shared=$useSharedModel")
            return FunasrContext(ptr)
        }

        fun getSystemInfo(): String = FunasrLib.getSystemInfo()
    }
}

private class FunasrLib {
    companion object {
        init {
            Log.d(LOG_TAG, "Primary ABI: ${Build.SUPPORTED_ABIS[0]}")
            Log.d(LOG_TAG, "Loading libai-chat.so (funasr bundled)")
            System.loadLibrary("ai-chat")
        }

        external fun init(encoderPath: String, llmPath: String, useSharedModel: Boolean = false): Long
        external fun transcribe(contextPtr: Long, audioData: FloatArray): String
        external fun free(contextPtr: Long)
        external fun getSystemInfo(): String
    }
}
