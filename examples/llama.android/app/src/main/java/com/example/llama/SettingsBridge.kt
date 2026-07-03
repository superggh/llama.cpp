package com.example.llama

import android.webkit.JavascriptInterface
import com.arm.aichat.translator.TranslatorOrchestrator

class SettingsBridge {
    @Volatile
    private var translator: TranslatorOrchestrator? = null

    fun setTranslator(t: TranslatorOrchestrator) {
        translator = t
    }

    @JavascriptInterface
    fun updateConfig(key: String, value: String) {
        val t = translator ?: return
        val v = value.toFloatOrNull()
        when (key) {
            "silenceThresholdDb" -> v?.let { t.updateVadParams(it, null, null) }
            "minSpeechDurationMs" -> v?.let { t.updateVadParams(null, null, v.toInt()) }
            "minSilenceDurationMs" -> v?.let { t.updateVadParams(null, v.toInt(), null) }
            "forceSplitSec" -> v?.let { t.setForceSplitSec(v.toInt()) }
            "sourceLanguage" -> t.setSourceLanguage(value)
            "targetLanguage" -> t.setTargetLanguage(value)
            "encoderThreads" -> v?.let { t.setEncoderThreads(v.toInt()) }
            "partialIntervalMs" -> v?.let { t.setPartialIntervalMs(v.toInt()) }
        }
    }

    @JavascriptInterface
    fun startVoice() { translator?.startVoiceChat() }

    @JavascriptInterface
    fun stopVoice() { translator?.stopVoiceChat() }

    @JavascriptInterface
    fun getAudioLevel(): Float = translator?.getAudioLevel() ?: -50f

    @JavascriptInterface
    fun setTtsEnabled(enabled: Boolean) { translator?.setTtsEnabled(enabled) }

    @JavascriptInterface
    fun getTtsEnabled(): Boolean = translator?.isTtsEnabled() ?: false

    @JavascriptInterface
    fun getConfig(): String = """
        {
            "silenceThresholdDb": 28,
            "minSpeechDurationMs": 400,
            "minSilenceDurationMs": 400,
            "forceSplitSec": 4,
            "sourceLanguage": "Auto",
            "targetLanguage": "English",
            "encoderThreads": 4,
            "partialIntervalMs": 4000,
            "ttsEnabled": false
        }
    """.trimIndent()
}
