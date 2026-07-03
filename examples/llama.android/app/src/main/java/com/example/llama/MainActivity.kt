package com.example.llama

import android.Manifest
import android.bluetooth.BluetoothManager
import android.content.Context
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.widget.EditText
import android.widget.TextView
import android.widget.ArrayAdapter
import android.widget.AutoCompleteTextView
import com.google.android.material.textfield.TextInputLayout
import android.widget.Toast
import androidx.activity.addCallback
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.arm.aichat.AiChat
import com.arm.aichat.InferenceEngine
import com.arm.aichat.ble.BleTranslatorService
import com.arm.aichat.gguf.GgufMetadata
import com.arm.aichat.gguf.GgufMetadataReader
import com.arm.aichat.translator.TranslatorOrchestrator
import com.google.android.material.button.MaterialButton
import com.google.android.material.floatingactionbutton.FloatingActionButton
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.onCompletion
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.util.UUID

class MainActivity : AppCompatActivity() {

    private lateinit var ggufTv: TextView
    private lateinit var messagesRv: RecyclerView
    private lateinit var userInputEt: EditText
    private lateinit var userActionFab: FloatingActionButton
    private lateinit var btnLoadWhisper: MaterialButton
    private lateinit var btnBle: MaterialButton
    private lateinit var btnVoice: MaterialButton
    private lateinit var statusTv: TextView
    private lateinit var sourceLangInput: AutoCompleteTextView
    private lateinit var targetLangInput: AutoCompleteTextView
    private lateinit var sourceLangLayout: TextInputLayout
    private lateinit var targetLangLayout: TextInputLayout

    private lateinit var engine: InferenceEngine
    private lateinit var translator: TranslatorOrchestrator

    private var generationJob: Job? = null
    private var isModelReady = false
    private var isVoiceActive = false
    @Deprecated("whisper replaced by FunASR") private var whisperModelPath: String? = null
    private var currentModelFile: File? = null

    private val languages = listOf("Auto", "Chinese", "English", "Japanese", "Korean", "Spanish", "French", "German", "Russian", "Arabic")
    private var sourceLanguage = "Auto"
    private var targetLanguage = "English"

    private val messages = mutableListOf<Message>()
    private val lastAssistantMsg = StringBuilder()
    private val messageAdapter = MessageAdapter(messages)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        onBackPressedDispatcher.addCallback { Log.w(TAG, "Ignore back press for simplicity") }

        ggufTv = findViewById(R.id.gguf)
        messagesRv = findViewById(R.id.messages)
        messagesRv.layoutManager = LinearLayoutManager(this).apply { stackFromEnd = true }
        messagesRv.adapter = messageAdapter
        userInputEt = findViewById(R.id.user_input)
        userActionFab = findViewById(R.id.fab)
        btnLoadWhisper = findViewById(R.id.btn_load_whisper)
        btnBle = findViewById(R.id.btn_ble)
        btnVoice = findViewById(R.id.btn_voice)
        statusTv = findViewById(R.id.status)
        sourceLangInput = findViewById(R.id.source_lang)
        targetLangInput = findViewById(R.id.target_lang)
        sourceLangLayout = findViewById(R.id.source_lang_layout)
        targetLangLayout = findViewById(R.id.target_lang_layout)

        setupLanguagePickers()

        requestPermissionsIfNeeded()

        lifecycleScope.launch(Dispatchers.Default) {
            engine = AiChat.getInferenceEngine(applicationContext)
            translator = TranslatorOrchestrator(applicationContext, engine)
            observeTranslator()
            withContext(Dispatchers.Main) {
                loadModelsOnStartup()
            }
        }

        userActionFab.setOnClickListener {
            if (isModelReady) handleUserInput() else getContent.launch(arrayOf("*/*"))
        }

        btnLoadWhisper.setOnClickListener { handleLoadFunasrClick() }
        btnBle.setOnClickListener { handleBleAction() }
        btnVoice.setOnClickListener { toggleVoiceChat() }
        findViewById<com.google.android.material.button.MaterialButton>(R.id.btn_swap_lang).setOnClickListener { swapLanguages() }
    }

    private fun findFunasrLlmModel(): File? {
        val funasrDir = File(filesDir, "funasr")
        val candidates = funasrDir.listFiles { _, name ->
            name.endsWith(FILE_EXTENSION_GGUF, ignoreCase = true) && name.startsWith("qwen3", ignoreCase = true)
        }
        return candidates?.maxByOrNull { it.lastModified() }
    }

    private fun findTranslationModel(): File? {
        val modelsDir = ensureModelsDirectory()
        val candidates = modelsDir.listFiles { _, name ->
            name.endsWith(FILE_EXTENSION_GGUF, ignoreCase = true)
        }
        return candidates?.maxByOrNull { it.lastModified() }
    }

    private fun findFunasrEncoder(): File? {
        val enc = File(filesDir, "funasr/funasr-encoder-f16.gguf")
        return if (enc.exists() && enc.isFile && enc.canRead()) enc else null
    }

    private fun loadModelsOnStartup() {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val encoderFile = findFunasrEncoder()
                val funasrLlmFile = findFunasrLlmModel()
                val translationModelFile = findTranslationModel()

                if (encoderFile == null || funasrLlmFile == null) {
                    withContext(Dispatchers.Main) {
                        setStatus("FunASR encoder or LLM not found in files/funasr/")
                    }
                    return@launch
                }

                // Load FunASR
                withContext(Dispatchers.Main) {
                    setStatus("Loading FunASR LLM: ${funasrLlmFile.name}")
                    loadFunasrModel(encoderFile.path, funasrLlmFile.path, useSharedModel = false)
                }

                // Load translation engine
                if (translationModelFile != null) {
                    withContext(Dispatchers.Main) {
                        setStatus("Loading translation engine: ${translationModelFile.name}")
                        loadLlmModel(translationModelFile)
                    }
                } else {
                    withContext(Dispatchers.Main) {
                        setStatus("Translation model not found in files/models/")
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load models", e)
                withContext(Dispatchers.Main) {
                    setStatus("Load failed: ${e.message}")
                    Toast.makeText(this@MainActivity, "Load failed: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    private fun setupLanguagePickers() {
        val adapter = ArrayAdapter(this, android.R.layout.simple_dropdown_item_1line, languages)
        sourceLangInput.setAdapter(adapter)
        targetLangInput.setAdapter(adapter)

        sourceLangInput.setOnItemClickListener { _, _, position, _ ->
            sourceLanguage = languages[position]
            updateLanguageLabels()
        }
        targetLangInput.setOnItemClickListener { _, _, position, _ ->
            targetLanguage = languages[position]
            updateLanguageLabels()
        }
        updateLanguageLabels()
    }

    private fun updateLanguageLabels() {
        sourceLangLayout.hint = "Source: $sourceLanguage"
        targetLangLayout.hint = "Target: $targetLanguage"
    }

    private fun swapLanguages() {
        if (sourceLanguage == "Auto") {
            Toast.makeText(this, "Cannot swap when source is Auto", Toast.LENGTH_SHORT).show()
            return
        }
        val temp = sourceLanguage
        sourceLanguage = targetLanguage
        targetLanguage = temp
        sourceLangInput.setText(sourceLanguage, false)
        targetLangInput.setText(targetLanguage, false)
        updateLanguageLabels()
    }

    private fun loadLlmModel(modelFile: File) {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                currentModelFile = modelFile
                val metadata = contentResolver.openFileDescriptor(Uri.fromFile(modelFile), "r")?.use { fd ->
                    GgufMetadataReader.create().readStructuredMetadata(FileInputStream(fd.fileDescriptor))
                }
                withContext(Dispatchers.Main) {
                    ggufTv.text = metadata?.toString() ?: modelFile.name
                    setStatus("Loading LLM model...")
                }
                engine.loadModel(modelFile.path)
                withContext(Dispatchers.Main) {
                    isModelReady = true
                    currentModelFile = modelFile
                    userInputEt.hint = "Type and send a message!"
                    userInputEt.isEnabled = true
                    userActionFab.setImageResource(R.drawable.outline_send_24)
                    userActionFab.isEnabled = true
                    setStatus("LLM model ready: ${modelFile.name}")
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    setStatus("LLM auto-load failed: ${e.message}")
                    Toast.makeText(this@MainActivity, "LLM load failed: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    private fun handleLoadFunasrClick() {
        val encoderFile = findFunasrEncoder()
        val funasrLlmFile = findFunasrLlmModel()
        if (encoderFile == null) {
            Toast.makeText(this, "FunASR encoder not found in files/funasr/", Toast.LENGTH_LONG).show()
            return
        }
        if (funasrLlmFile == null) {
            Toast.makeText(this, "FunASR LLM not found in files/funasr/", Toast.LENGTH_LONG).show()
            return
        }
        lifecycleScope.launch(Dispatchers.IO) {
            withContext(Dispatchers.Main) {
                setStatus("Loading FunASR models...")
                loadFunasrModel(encoderFile.path, funasrLlmFile.path, useSharedModel = false)
            }
        }
    }

    private fun loadFunasrModel(encoderPath: String, llmPath: String, useSharedModel: Boolean = false) {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                translator.loadFunasrModel(encoderPath, llmPath, useSharedModel)
                withContext(Dispatchers.Main) {
                    setStatus("FunASR loaded")
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    setStatus("FunASR load failed: ${e.message}")
                    Toast.makeText(this@MainActivity, "FunASR load failed: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    private fun setStatus(message: String) {
        statusTv.text = "Status: $message"
        Log.i(TAG, "Status: $message")
    }

    private fun requestPermissionsIfNeeded() {
        val permissions = mutableListOf<String>()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.BLUETOOTH_SCAN) != PackageManager.PERMISSION_GRANTED) {
                permissions.add(Manifest.permission.BLUETOOTH_SCAN)
            }
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.BLUETOOTH_CONNECT) != PackageManager.PERMISSION_GRANTED) {
                permissions.add(Manifest.permission.BLUETOOTH_CONNECT)
            }
        } else {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.BLUETOOTH) != PackageManager.PERMISSION_GRANTED) {
                permissions.add(Manifest.permission.BLUETOOTH)
            }
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.BLUETOOTH_ADMIN) != PackageManager.PERMISSION_GRANTED) {
                permissions.add(Manifest.permission.BLUETOOTH_ADMIN)
            }
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            permissions.add(Manifest.permission.RECORD_AUDIO)
        }
        if (permissions.isNotEmpty()) {
            permissionLauncher.launch(permissions.toTypedArray())
        }
    }

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { results ->
        val denied = results.filter { !it.value }.keys
        if (denied.isNotEmpty()) {
            Toast.makeText(this, "Permissions required: $denied", Toast.LENGTH_LONG).show()
        }
    }

    private fun observeTranslator() {
        lifecycleScope.launch {
            translator.state.collect { state ->
                withContext(Dispatchers.Main) {
                    updateUiForState(state)
                    when (state) {
                        is TranslatorOrchestrator.State.Listening -> setStatus("Listening...")
                        is TranslatorOrchestrator.State.Transcribing -> setStatus("Transcribing...")
                        is TranslatorOrchestrator.State.Translating -> setStatus("Translating...")
                        is TranslatorOrchestrator.State.Error -> setStatus("Error: ${state.message}")
                        else -> {}
                    }
                }
            }
        }
        lifecycleScope.launch {
            translator.events.collect { event ->
                withContext(Dispatchers.Main) {
                    when (event) {
                        is TranslatorOrchestrator.Event.FunasrLoaded -> {
                            setStatus("FunASR loaded")
                            Toast.makeText(this@MainActivity, "FunASR loaded", Toast.LENGTH_SHORT).show()
                        }
                        is TranslatorOrchestrator.Event.WhisperLoaded -> {
                            setStatus("Whisper loaded (legacy)")
                        }
                        is TranslatorOrchestrator.Event.PartialTranscription -> {
                            setStatus("Partial: ${event.text}")
                        }
                        is TranslatorOrchestrator.Event.TranscriptionResult -> {
                            setStatus("Transcribed: ${event.text}")
                            addMessage(event.text, true)
                        }
                        is TranslatorOrchestrator.Event.TranslationResult -> {
                            setStatus("Translated: ${event.translated}")
                            addMessage(event.translated, false)
                        }
                        is TranslatorOrchestrator.Event.QueueUpdate -> {
                            if (event.pending > 0) setStatus("Queued: ${event.pending}")
                        }
                    }
                }
            }
        }
        lifecycleScope.launch {
            translator.bleConnectionState.collect { state ->
                withContext(Dispatchers.Main) {
                    btnBle.text = when (state) {
                        is BleTranslatorService.ConnectionState.Disconnected -> "BLE"
                        is BleTranslatorService.ConnectionState.Scanning -> "Scanning"
                        is BleTranslatorService.ConnectionState.Connecting -> "Connecting"
                        is BleTranslatorService.ConnectionState.Connected -> state.deviceName ?: "Connected"
                        is BleTranslatorService.ConnectionState.Ready -> "BLE Ready"
                        is BleTranslatorService.ConnectionState.Error -> "BLE Error"
                    }
                }
            }
        }
    }

    private fun updateUiForState(state: TranslatorOrchestrator.State) {
        btnVoice.text = when (state) {
            is TranslatorOrchestrator.State.Listening -> "Stop Voice"
            else -> "Start Voice"
        }
        isVoiceActive = state is TranslatorOrchestrator.State.Listening
    }

    private fun handleBleAction() {
        val bluetoothManager = getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager
        val adapter = bluetoothManager.adapter
        if (adapter?.isEnabled != true) {
            Toast.makeText(this, "Please enable Bluetooth", Toast.LENGTH_SHORT).show()
            return
        }
        when (translator.bleConnectionState.value) {
            is BleTranslatorService.ConnectionState.Disconnected,
            is BleTranslatorService.ConnectionState.Error -> {
                showBleDeviceDialog(bluetoothManager)
            }
            is BleTranslatorService.ConnectionState.Scanning,
            is BleTranslatorService.ConnectionState.Connecting -> {
                translator.stopScanning()
            }
            is BleTranslatorService.ConnectionState.Connected,
            is BleTranslatorService.ConnectionState.Ready -> {
                translator.ble.disconnect()
            }
        }
    }

    private fun showBleDeviceDialog(bluetoothManager: BluetoothManager) {
        val adapter = bluetoothManager.adapter
        val paired = adapter?.bondedDevices?.toList() ?: emptyList()
        val names = paired.map { "${it.name ?: "Unknown"} (${it.address})" }.toTypedArray()
        AlertDialog.Builder(this)
            .setTitle("Select ESP32-S3")
            .setItems(names) { _, which ->
                val device = paired[which]
                translator.ble.startScan(device.address)
            }
            .setNegativeButton("Scan") { _, _ ->
                translator.startScanningForDevice()
            }
            .show()
    }

    private fun toggleVoiceChat() {
        if (!isModelReady) {
            Toast.makeText(this, "Please load a LLM model first", Toast.LENGTH_SHORT).show()
            return
        }
        if (isVoiceActive) {
            translator.stopVoiceChat()
        } else {
            translator.startVoiceChat(sourceLanguage = sourceLanguage, targetLanguage = targetLanguage)
        }
    }

    private val getContent = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri ->
        Log.i(TAG, "Selected file uri:\n $uri")
        uri?.let { handleSelectedModel(it) }
    }

    private fun handleSelectedModel(uri: Uri) {
        userActionFab.isEnabled = false
        userInputEt.hint = "Parsing GGUF..."
        setStatus("Parsing GGUF metadata...")
        ggufTv.text = "Parsing metadata from selected file \n$uri"

        lifecycleScope.launch(Dispatchers.IO) {
            contentResolver.openInputStream(uri)?.use { stream ->
                GgufMetadataReader.create().readStructuredMetadata(stream)
            }?.let { metadata ->
                Log.i(TAG, "GGUF parsed: \n$metadata")
                withContext(Dispatchers.Main) {
                    ggufTv.text = metadata.toString()
                    setStatus("Copying model file...")
                }

                val modelName = metadata.filename() + FILE_EXTENSION_GGUF
                contentResolver.openInputStream(uri)?.use { _ ->
                    copyFileToLocal(uri, modelName)
                }?.let { modelFile ->
                    withContext(Dispatchers.Main) { setStatus("Loading LLM model...") }
                    loadLlmModel(modelFile)
                }
            }
        }
    }

    private suspend fun copyFileToLocal(uri: Uri, fileName: String): File =
        withContext(Dispatchers.IO) {
            File(ensureModelsDirectory(), fileName).also { file ->
                if (!file.exists()) {
                    contentResolver.openInputStream(uri)?.use { input ->
                        FileOutputStream(file).use { output -> input.copyTo(output) }
                    }
                }
            }
        }

    private fun handleUserInput() {
        userInputEt.text.toString().also { userMsg ->
            if (userMsg.isEmpty()) {
                Toast.makeText(this, "Input message is empty!", Toast.LENGTH_SHORT).show()
            } else {
                userInputEt.text = null
                userInputEt.isEnabled = false
                userActionFab.isEnabled = false

                addMessage(userMsg, true)
                lastAssistantMsg.clear()
                addMessage("", false)

                generationJob = lifecycleScope.launch(Dispatchers.Default) {
                    engine.sendUserPrompt(userMsg)
                        .onCompletion {
                            withContext(Dispatchers.Main) {
                                userInputEt.isEnabled = true
                                userActionFab.isEnabled = true
                            }
                        }.collect { token ->
                            withContext(Dispatchers.Main) {
                                val messageCount = messages.size
                                check(messageCount > 0 && !messages[messageCount - 1].isUser)

                                messages.removeAt(messageCount - 1).copy(
                                    content = lastAssistantMsg.append(token).toString()
                                ).let { messages.add(it) }

                                messageAdapter.notifyItemChanged(messages.size - 1)
                            }
                        }
                }
            }
        }
    }

    private fun addMessage(text: String, isUser: Boolean) {
        messages.add(Message(UUID.randomUUID().toString(), text, isUser))
        messageAdapter.notifyItemInserted(messages.size - 1)
        messagesRv.scrollToPosition(messages.size - 1)
    }

    private fun ensureModelsDirectory() =
        File(filesDir, DIRECTORY_MODELS).also {
            if (it.exists() && !it.isDirectory) { it.delete() }
            if (!it.exists()) { it.mkdir() }
        }

    override fun onStop() {
        generationJob?.cancel()
        super.onStop()
    }

    override fun onDestroy() {
        translator.destroy()
        engine.destroy()
        super.onDestroy()
    }

    companion object {
        private val TAG = MainActivity::class.java.simpleName
        private const val DIRECTORY_MODELS = "models"
        private const val FILE_EXTENSION_GGUF = ".gguf"
    }
}

@OptIn(ExperimentalStdlibApi::class)
fun GgufMetadata.filename() = when {
    basic.name != null -> {
        basic.name?.let { name ->
            basic.sizeLabel?.let { size ->
                "$name-$size"
            } ?: name
        }
    }
    architecture?.architecture != null -> {
        architecture?.architecture?.let { arch ->
            basic.uuid?.let { uuid ->
                "$arch-$uuid"
            } ?: "$arch-${System.currentTimeMillis()}"
        }
    }
    else -> {
        "model-${System.currentTimeMillis().toHexString()}"
    }
}
