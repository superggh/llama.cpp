package com.arm.aichat.audio

import kotlin.concurrent.withLock
import kotlin.math.sqrt
import java.util.concurrent.locks.ReentrantLock

class EnergyVad(
    private val sampleRate: Int = 16000,
    private val frameMs: Int = 30,
    private val silenceThresholdDb: Float = -40f,
    private val minSpeechDurationMs: Int = 300,
    private val minSilenceDurationMs: Int = 700
) {
    private val frameSize = sampleRate * frameMs / 1000
    private val minSpeechFrames = minSpeechDurationMs / frameMs
    private val minSilenceFrames = minSilenceDurationMs / frameMs

    private val lock = ReentrantLock()
    private val ringBuffer = ArrayDeque<Short>()
    private var speechFrames = 0
    private var silenceFrames = 0
    private var isSpeaking = false
    private var pendingSpeech = mutableListOf<Short>()

    data class Segment(val samples: ShortArray, val isSpeech: Boolean) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is Segment) return false
            return isSpeech == other.isSpeech && samples.contentEquals(other.samples)
        }
        override fun hashCode(): Int = samples.contentHashCode() * 31 + isSpeech.hashCode()
    }

    fun pendingSize(): Int = lock.withLock { pendingSpeech.size }
    fun getPendingBuffer(): ShortArray = lock.withLock { pendingSpeech.toShortArray() }

    fun process(samples: ShortArray): List<Segment> = lock.withLock {
        val results = mutableListOf<Segment>()
        ringBuffer.addAll(samples.toList())

        while (ringBuffer.size >= frameSize) {
            val frame = ShortArray(frameSize)
            repeat(frameSize) { frame[it] = ringBuffer.removeFirst() }
            val segment = processFrame(frame)
            if (segment != null) results.add(segment)
        }
        results
    }

    fun flush(): Segment? = lock.withLock {
        if (pendingSpeech.isNotEmpty()) {
            val samples = pendingSpeech.toShortArray()
            pendingSpeech.clear()
            Segment(samples, true)
        } else {
            null
        }
    }

    private fun processFrame(frame: ShortArray): Segment? {
        val energyDb = 10f * kotlin.math.log10(rms(frame).toDouble().coerceAtLeast(1e-10)).toFloat()
        val isSpeechFrame = energyDb > silenceThresholdDb

        return when {
            isSpeechFrame -> {
                speechFrames++
                silenceFrames = 0
                pendingSpeech.addAll(frame.toList())
                if (!isSpeaking && speechFrames >= minSpeechFrames) {
                    isSpeaking = true
                }
                null
            }
            isSpeaking -> {
                silenceFrames++
                pendingSpeech.addAll(frame.toList())
                if (silenceFrames >= minSilenceFrames) {
                    isSpeaking = false
                    speechFrames = 0
                    silenceFrames = 0
                    val samples = pendingSpeech.toShortArray()
                    pendingSpeech.clear()
                    Segment(samples, true)
                } else {
                    null
                }
            }
            else -> {
                speechFrames = (speechFrames - 1).coerceAtLeast(0)
                null
            }
        }
    }

    private fun rms(frame: ShortArray): Float {
        var sum = 0L
        for (s in frame) {
            sum += s.toInt() * s.toInt()
        }
        return sqrt(sum.toFloat() / frame.size)
    }

    internal fun List<Short>.toShortArray(): ShortArray {
        return ShortArray(size) { this[it] }
    }
}

fun ShortArray.append(other: ShortArray): ShortArray {
    return ShortArray(size + other.size).apply {
        this@append.copyInto(this)
        other.copyInto(this, this@append.size)
    }
}
