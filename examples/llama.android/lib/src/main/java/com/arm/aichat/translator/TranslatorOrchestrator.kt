package com.arm.aichat.translator

import android.content.Context
import android.os.PowerManager
import android.util.Log
import com.arm.aichat.InferenceEngine
import com.arm.aichat.audio.AudioRecorder
import com.arm.aichat.audio.EnergyVad
import com.arm.aichat.audio.OpusDecoder
import com.arm.aichat.audio.append
import com.arm.aichat.audio.toFloatArray
import com.arm.aichat.ble.BleTranslatorService
import com.arm.aichat.funasr.FunasrContext
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

private const val TAG = "Translator"

class TranslatorOrchestrator(
    context: Context,
    private val engine: InferenceEngine,
    val ble: BleTranslatorService = BleTranslatorService(context)
) {
    private val appContext = context.applicationContext
    private val powerManager = appContext.getSystemService(Context.POWER_SERVICE) as PowerManager
    private var wakeLock: PowerManager.WakeLock? = null

    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private val audioRecorder = AudioRecorder()
    private var opusDecoder: OpusDecoder? = null

    private var funasrContext: FunasrContext? = null
    private var isChatting = false
    private var isLoadingFunasr = AtomicBoolean(false)
    private val isWhisperBusy = AtomicBoolean(false)

    private val _state = MutableStateFlow<State>(State.Idle)
    val state: StateFlow<State> = _state.asStateFlow()

    private val _events = MutableSharedFlow<Event>()
    val events: Flow<Event> = _events

    val bleConnectionState = ble.connectionState

    private val segmentChannel = Channel<FloatArray>(Channel.UNLIMITED)
    private val pendingSegmentCount = AtomicInteger(0)
    private var segmentProcessorJob: kotlinx.coroutines.Job? = null

    init {
        scope.launch {
            try {
                opusDecoder = OpusDecoder(SAMPLE_RATE, 1)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to create opus decoder", e)
            }
        }
        startSegmentProcessor()
    }

    private fun startSegmentProcessor() {
        segmentProcessorJob = scope.launch {
            for (samples in segmentChannel) {
                if (!isActive) break
                processSegment(samples)
                pendingSegmentCount.decrementAndGet()
            }
        }
    }

    private fun listenFromBle(onPcm: suspend (ShortArray) -> Unit) {
        scope.launch {
            try {
                ble.receivedAudio.collect { opusFrame ->
                    val decoder = opusDecoder ?: return@collect
                    val pcm = decoder.decode(opusFrame, SAMPLES_PER_FRAME)
                    if (pcm.isNotEmpty()) {
                        onPcm(pcm)
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "BLE audio error", e)
            }
        }
    }

    private fun feedVad(vad: EnergyVad, pcm: ShortArray, buffer: ShortArray): ShortArray {
        var audioBuffer = buffer
        val segments = vad.process(pcm)
        for (segment in segments) {
            if (segment.isSpeech) {
                audioBuffer = audioBuffer.append(segment.samples)
                enqueueSegment(audioBuffer.toFloatArray())
                audioBuffer = ShortArray(0)
            }
        }
        if (vad.pendingSize() > SAMPLE_RATE * 4) {
            val flushed = vad.flush()
            if (flushed != null && flushed.samples.isNotEmpty()) {
                audioBuffer = audioBuffer.append(flushed.samples)
                enqueueSegment(audioBuffer.toFloatArray())
                audioBuffer = ShortArray(0)
            }
        }
        return audioBuffer
    }

    private fun enqueueSegment(samples: FloatArray) {
        if (samples.isEmpty()) return
        if (pendingSegmentCount.get() >= MAX_PENDING_SEGMENTS) {
            Log.w(TAG, "Dropping segment, queue full (${pendingSegmentCount.get()})")
            return
        }
        pendingSegmentCount.incrementAndGet()
        val offered = segmentChannel.trySend(samples)
        if (!offered.isSuccess) {
            pendingSegmentCount.decrementAndGet()
            Log.w(TAG, "Failed to enqueue segment")
        } else {
            scope.launch { _events.emit(Event.QueueUpdate(pendingSegmentCount.get())) }
        }
    }

    fun loadFunasrModel(encoderPath: String, llmPath: String, useSharedModel: Boolean = false) {
        if (isLoadingFunasr.compareAndSet(false, true)) {
            scope.launch {
                try {
                    _state.value = State.LoadingWhisper
                    funasrContext?.release()
                    funasrContext = FunasrContext.create(encoderPath, llmPath, useSharedModel)
                    _state.value = State.Ready
                    _events.emit(Event.FunasrLoaded)
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to load FunASR", e)
                    _state.value = State.Error("FunASR load failed: ${e.message}")
                } finally {
                    isLoadingFunasr.set(false)
                }
            }
        }
    }

    fun startScanningForDevice(address: String? = null) {
        ble.startScan(address)
    }

    fun stopScanning() {
        ble.stopScan()
    }

    private var translationPrefix = "You are a translator. Translate the user's input to English. Output only the translation, no explanation.\n\n"

    private var currentVad: EnergyVad? = null
    private var currentAudioBuffer: ShortArray = ShortArray(0)

    private var partialTranscribeJob: kotlinx.coroutines.Job? = null
    private var accumulatedPartialText = ""

    fun startVoiceChat(sourceLanguage: String = "Auto", targetLanguage: String = "English", systemPrompt: String? = null) {
        if (isChatting) return
        isChatting = true
        _state.value = State.Listening
        acquireWakeLock()
        pendingSegmentCount.set(0)
        accumulatedPartialText = ""

        translationPrefix = systemPrompt ?: buildTranslationPrompt(sourceLanguage, targetLanguage)

        val vad = EnergyVad()
        currentVad = vad
        currentAudioBuffer = ShortArray(0)
        val useBleAudio = bleConnectionState.value is BleTranslatorService.ConnectionState.Ready

        // Periodic partial transcription during active speech
        partialTranscribeJob = scope.launch {
            while (isChatting) {
                delay(PARTIAL_INTERVAL_MS)
                if (!isChatting) break
                if (isWhisperBusy.get()) continue
                val v = currentVad ?: continue
                val pending = v.getPendingBuffer()
                if (pending.size < PARTIAL_MIN_SAMPLES) continue
                isWhisperBusy.set(true)
                try {
                    val text = funasrContext?.transcribe(pending.toFloatArray()) ?: ""
                    if (text.isNotBlank() && text != accumulatedPartialText && !text.equals("/sil", ignoreCase = true)) {
                        accumulatedPartialText = text
                        _events.emit(Event.PartialTranscription(text))
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "Partial transcribe failed", e)
                } finally {
                    isWhisperBusy.set(false)
                }
            }
        }

        if (useBleAudio) {
            setStatus("Listening from BLE device...")
            listenFromBle { pcm ->
                currentAudioBuffer = feedVad(vad, pcm, currentAudioBuffer)
            }
        } else {
            setStatus("Listening from phone microphone...")
            scope.launch {
                try {
                    audioRecorder.start().collect { chunk ->
                        currentAudioBuffer = feedVad(vad, chunk, currentAudioBuffer)
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Recording error", e)
                    _state.value = State.Error("Recording error: ${e.message}")
                } finally {
                    finalizeVoiceChat()
                }
            }
        }
    }

    private fun buildTranslationPrompt(sourceLanguage: String, targetLanguage: String): String {
        return if (targetLanguage.equals("Chinese", ignoreCase = true)) {
            "将以下文本翻译为中文。注意只输出翻译结果，不要额外解释："
        } else {
            "Translate the following text into $targetLanguage. Note that you should only output the translated result without any additional explanation:"
        }
    }

    private fun setStatus(message: String) {
        Log.i(TAG, message)
    }

    fun stopVoiceChat() {
        isChatting = false
        audioRecorder.stop()
        partialTranscribeJob?.cancel()
        partialTranscribeJob = null
        finalizeVoiceChat()
        releaseWakeLock()
    }

    private fun acquireWakeLock() {
        try {
            wakeLock?.release()
            wakeLock = powerManager.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "aichat:VoiceTranslation").apply {
                setReferenceCounted(false)
                acquire(10 * 60 * 1000L)
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to acquire wake lock", e)
        }
    }

    private fun releaseWakeLock() {
        try {
            wakeLock?.let {
                if (it.isHeld) it.release()
            }
            wakeLock = null
        } catch (e: Exception) {
            Log.w(TAG, "Failed to release wake lock", e)
        }
    }

    private fun finalizeVoiceChat() {
        val vad = currentVad ?: return
        currentVad = null
        var buffer = currentAudioBuffer
        currentAudioBuffer = ShortArray(0)
        val flushed = vad.flush()
        if (flushed != null && flushed.samples.isNotEmpty()) {
            buffer = buffer.append(flushed.samples)
        }
        if (buffer.isNotEmpty()) {
            enqueueSegment(buffer.toFloatArray())
        }
    }

    fun releaseOpusDecoder() {
        opusDecoder?.release()
        opusDecoder = null
    }

    fun translateText(text: String, sourceLanguage: String = "Auto", targetLanguage: String = "English") {
        translationPrefix = buildTranslationPrompt(sourceLanguage, targetLanguage)
        scope.launch {
            _state.value = State.Translating
            try {
                val translated = translateWithLlm(text)
                _events.emit(Event.TranslationResult(text, translated))
                ble.sendText(translated)
                _state.value = State.Ready
            } catch (e: Exception) {
                Log.e(TAG, "Translation failed", e)
                _state.value = State.Error("Translation failed: ${e.message}")
            }
        }
    }

    private suspend fun translateWithLlm(text: String): String =
        withContext(Dispatchers.IO) {
            engine.resetContext()
            engine.setSystemPrompt(translationPrefix)
            val sb = StringBuilder()
            engine.sendUserPrompt(text).collect { token ->
                sb.append(token)
            }
            sb.toString().trim()
        }

    private suspend fun processSegment(samples: FloatArray) {
        _state.value = State.Transcribing
        val funasr = funasrContext ?: return
        isWhisperBusy.set(true)
        try {
            val text = funasr.transcribe(samples)
            if (text.isBlank() || text.equals("/sil", ignoreCase = true)) {
                _state.value = if (isChatting) State.Listening else State.Ready
                return
            }
            _events.emit(Event.TranscriptionResult(text))
            accumulatedPartialText = ""
            _state.value = State.Translating
            val translated = translateWithLlm(text)
            if (translated.isNotBlank()) {
                _events.emit(Event.TranslationResult(text, translated))
                ble.sendText(translated)
            }
            _state.value = if (isChatting) State.Listening else State.Ready
        } catch (e: Exception) {
            Log.e(TAG, "Process segment failed", e)
            _state.value = if (isChatting) State.Listening else State.Ready
        } finally {
            isWhisperBusy.set(false)
        }
    }

    fun destroy() {
        stopVoiceChat()
        segmentProcessorJob?.cancel()
        segmentChannel.close()
        releaseWakeLock()
        releaseOpusDecoder()
        scope.launch {
            funasrContext?.release()
            funasrContext = null
            ble.disconnect()
        }
        scope.cancel()
    }

    sealed class State {
        object Idle : State()
        object LoadingWhisper : State()
        object Ready : State()
        object Listening : State()
        object Transcribing : State()
        object Translating : State()
        data class Error(val message: String) : State()
    }

    sealed class Event {
        object FunasrLoaded : Event()
        object WhisperLoaded : Event()
        data class PartialTranscription(val text: String) : Event()
        data class TranscriptionResult(val text: String) : Event()
        data class TranslationResult(val source: String, val translated: String) : Event()
        data class QueueUpdate(val pending: Int) : Event()
    }

    companion object {
        private const val SAMPLE_RATE = 16000
        private const val SAMPLES_PER_FRAME = 320 // 20ms @ 16kHz
        private const val MAX_PENDING_SEGMENTS = 2
        private const val PARTIAL_INTERVAL_MS = 4000L
        private const val PARTIAL_MIN_SAMPLES = SAMPLE_RATE * 4 // 4 seconds
    }
}

