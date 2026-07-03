package com.example.llama

import android.Manifest
import android.bluetooth.BluetoothManager
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.Toast
import androidx.activity.addCallback
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.arm.aichat.AiChat
import com.arm.aichat.InferenceEngine
import com.arm.aichat.ble.BleTranslatorService
import com.arm.aichat.translator.TranslatorOrchestrator
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.UUID

class MainActivity : AppCompatActivity() {

    private lateinit var webView: WebView
    private lateinit var engine: InferenceEngine
    private lateinit var translator: TranslatorOrchestrator
    private lateinit var bridge: SettingsBridge

    private var generationJob: Job? = null
    private var isModelReady = false
    private var currentModelFile: File? = null

    private val messages = mutableListOf<Message>()
    private val lastAssistantMsg = StringBuilder()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        onBackPressedDispatcher.addCallback { Log.w(TAG, "Ignore back press") }

        webView = findViewById(R.id.webview)
        webView.settings.javaScriptEnabled = true
        webView.settings.domStorageEnabled = true
        webView.settings.allowFileAccess = true
        webView.settings.allowContentAccess = true
        bridge = SettingsBridge()
        webView.addJavascriptInterface(bridge, "Android")
        webView.webViewClient = object : WebViewClient() {
            override fun onPageFinished(view: WebView, url: String) {
                pushEvent("bridge", "ready")
            }
        }
        webView.loadUrl("file:///android_asset/react-ui/index.html")

        requestPermissionsIfNeeded()

        lifecycleScope.launch(Dispatchers.Default) {
            engine = AiChat.getInferenceEngine(applicationContext)
            translator = TranslatorOrchestrator(applicationContext, engine)
            bridge.setTranslator(translator)
            observeTranslator()
            withContext(Dispatchers.Main) {
                loadModelsOnStartup()
            }
        }
    }

    private fun pushEvent(type: String, data: Any) {
        val json = "\"$data\""
        try {
            webView.evaluateJavascript(
                "window.dispatchEvent(new CustomEvent('native-event', {detail:{type:'$type',data:$json}}))",
                null
            )
        } catch (e: Exception) {
            Log.w(TAG, "pushEvent failed: $e")
        }
    }

    private fun findMonoAsrModel(): File? {
        val asrDir = File(filesDir, "funasr")
        return asrDir.listFiles { _, name ->
            name.endsWith(".gguf", ignoreCase = true) && name.startsWith("funasr-nano", ignoreCase = true)
        }?.maxByOrNull { it.lastModified() }
    }

    private fun findFunasrEncoder(): File? {
        val enc = File(filesDir, "funasr/funasr-encoder-f16.gguf")
        return if (enc.exists() && enc.isFile && enc.canRead()) enc else null
    }

    private fun findFunasrLlmModel(): File? {
        val funasrDir = File(filesDir, "funasr")
        return funasrDir.listFiles { _, name ->
            name.endsWith(".gguf", ignoreCase = true) && name.startsWith("qwen3", ignoreCase = true)
        }?.maxByOrNull { it.lastModified() }
    }

    private fun findTranslationModel(): File? {
        val modelsDir = File(filesDir, "models")
        return modelsDir.listFiles { _, name -> name.endsWith(".gguf", ignoreCase = true) }
            ?.maxByOrNull { it.lastModified() }
    }

    private fun loadModelsOnStartup() {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val encoderFile = findFunasrEncoder()
                val funasrLlmFile = findFunasrLlmModel()
                val translationModelFile = findTranslationModel()

                if (encoderFile != null && funasrLlmFile != null) {
                    loadFunasrModel(encoderFile.path, funasrLlmFile.path, useSharedModel = false)
                } else {
                    withContext(Dispatchers.Main) { pushEvent("status", "ASR model not found") }
                }

                if (translationModelFile != null) {
                    loadLlmModel(translationModelFile)
                } else {
                    withContext(Dispatchers.Main) { pushEvent("status", "Translation model not found") }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Load models failed", e)
                withContext(Dispatchers.Main) { pushEvent("status", "Load failed: ${e.message}") }
            }
        }
    }

    private fun loadFunasrModel(encoderPath: String, llmPath: String, useSharedModel: Boolean) {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                translator.loadFunasrModel(encoderPath, llmPath, useSharedModel)
                withContext(Dispatchers.Main) { pushEvent("status", "ASR loaded") }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    pushEvent("status", "ASR failed: ${e.message}")
                    Toast.makeText(this@MainActivity, "ASR load failed", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    private fun loadLlmModel(modelFile: File) {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                currentModelFile = modelFile
                engine.loadModel(modelFile.path)
                withContext(Dispatchers.Main) {
                    isModelReady = true
                    currentModelFile = modelFile
                    pushEvent("status", "LLM ready: ${modelFile.name}")
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    pushEvent("status", "LLM failed: ${e.message}")
                    Toast.makeText(this@MainActivity, "LLM load failed: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    private fun observeTranslator() {
        lifecycleScope.launch {
            translator.state.collect { state ->
                pushEvent("state", state.javaClass.simpleName)
            }
        }
        lifecycleScope.launch {
            translator.events.collect { event ->
                when (event) {
                    is TranslatorOrchestrator.Event.FunasrLoaded -> pushEvent("status", "FunASR ready")
                    is TranslatorOrchestrator.Event.TranscriptionResult -> pushEvent("transcribed", event.text)
                    is TranslatorOrchestrator.Event.TranslationResult -> pushEvent("translated", event.translated)
                    is TranslatorOrchestrator.Event.PartialTranscription -> {}
                    is TranslatorOrchestrator.Event.QueueUpdate -> {}
                    is TranslatorOrchestrator.Event.WhisperLoaded -> {}
                }
            }
        }
        lifecycleScope.launch {
            translator.bleConnectionState.collect { state ->
                pushEvent("ble", state.javaClass.simpleName)
            }
        }
    }

    private fun requestPermissionsIfNeeded() {
        val permissions = mutableListOf<String>()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.BLUETOOTH_SCAN) != PackageManager.PERMISSION_GRANTED)
                permissions.add(Manifest.permission.BLUETOOTH_SCAN)
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.BLUETOOTH_CONNECT) != PackageManager.PERMISSION_GRANTED)
                permissions.add(Manifest.permission.BLUETOOTH_CONNECT)
        } else {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.BLUETOOTH) != PackageManager.PERMISSION_GRANTED)
                permissions.add(Manifest.permission.BLUETOOTH)
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.BLUETOOTH_ADMIN) != PackageManager.PERMISSION_GRANTED)
                permissions.add(Manifest.permission.BLUETOOTH_ADMIN)
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED)
            permissions.add(Manifest.permission.RECORD_AUDIO)
        if (permissions.isNotEmpty()) permissionLauncher.launch(permissions.toTypedArray())
    }

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { results ->
        val denied = results.filter { !it.value }.keys
        if (denied.isNotEmpty()) Toast.makeText(this, "Permissions required: $denied", Toast.LENGTH_LONG).show()
    }

    override fun onStop() { generationJob?.cancel(); super.onStop() }
    override fun onDestroy() { translator.destroy(); engine.destroy(); super.onDestroy() }

    companion object {
        private val TAG = MainActivity::class.java.simpleName
    }
}
