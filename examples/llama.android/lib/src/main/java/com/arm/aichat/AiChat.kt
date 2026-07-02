package com.arm.aichat

import android.content.Context
import com.arm.aichat.internal.InferenceEngineImpl

/**
 * Main entry point for Arm's AI Chat library.
 */
object AiChat {
    /**
     * Get the inference engine single instance.
     */
    fun getInferenceEngine(context: Context) = InferenceEngineImpl.getInstance(context)

    /**
     * Load a shared LLM model that can be reused by the inference engine and FunASR.
     * Must be called before loading either engine if model sharing is desired.
     * Returns an opaque native pointer, or 0 on failure.
     */
    fun loadSharedModel(context: Context, modelPath: String): Long {
        // Ensure native library is loaded before calling JNI
        InferenceEngineImpl.getInstance(context)
        return InferenceEngineImpl.loadSharedModel(modelPath)
    }
}
