package com.arm.aichat.ble

import android.annotation.SuppressLint
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothGatt
import android.bluetooth.BluetoothGattCallback
import android.bluetooth.BluetoothGattCharacteristic
import android.bluetooth.BluetoothGattDescriptor
import android.bluetooth.BluetoothManager
import android.bluetooth.BluetoothProfile
import android.bluetooth.le.ScanCallback
import android.bluetooth.le.ScanFilter
import android.bluetooth.le.ScanResult
import android.bluetooth.le.ScanSettings
import android.content.Context
import android.os.ParcelUuid
import android.util.Log
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.flow
import java.util.UUID
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.util.concurrent.atomic.AtomicBoolean

private const val TAG = "BleTranslator"

object BleUuid {
    val SERVICE = UUID.fromString("abcd1000-1234-5678-1234-56789abcdef0")
    val AUDIO_UP = UUID.fromString("abcd1001-1234-5678-1234-56789abcdef0")
    val TEXT_DOWN = UUID.fromString("abcd1002-1234-5678-1234-56789abcdef0")
    val CONTROL = UUID.fromString("abcd1003-1234-5678-1234-56789abcdef0")
}

class BleTranslatorService(
    context: Context,
    private val autoReconnect: Boolean = true,
    private val reconnectDelayMs: Long = 2000
) {
    private val appContext = context.applicationContext
    private val bluetoothManager = appContext.getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager
    private val adapter: BluetoothAdapter? = bluetoothManager.adapter
    private val scanner = adapter?.bluetoothLeScanner

    private val scope = CoroutineScope(Dispatchers.IO)

    private var gatt: BluetoothGatt? = null
    private var textDownChar: BluetoothGattCharacteristic? = null

    private val _connectionState = MutableStateFlow<ConnectionState>(ConnectionState.Disconnected)
    val connectionState: StateFlow<ConnectionState> = _connectionState.asStateFlow()

    private val _receivedAudio = Channel<ByteArray>(Channel.BUFFERED)
    val receivedAudio: Flow<ByteArray> = flow {
        for (frame in _receivedAudio) emit(frame)
    }

    private val isScanning = AtomicBoolean(false)
    private var pendingConnectAddress: String? = null
    private var lastConnectedAddress: String? = null
    private var reconnectJob: Job? = null
    private var lastConnectedDeviceName: String? = null

    private val scanCallback = object : ScanCallback() {
        override fun onScanResult(callbackType: Int, result: ScanResult?) {
            result?.device?.let { device ->
                if (pendingConnectAddress == null || device.address == pendingConnectAddress) {
                    stopScan()
                    connect(device)
                }
            }
        }

        override fun onBatchScanResults(results: MutableList<ScanResult>?) {
            results?.firstOrNull()?.device?.let { device ->
                if (pendingConnectAddress == null || device.address == pendingConnectAddress) {
                    stopScan()
                    connect(device)
                }
            }
        }

        override fun onScanFailed(errorCode: Int) {
            Log.e(TAG, "Scan failed: $errorCode")
            _connectionState.value = ConnectionState.Error("Scan failed: $errorCode")
            isScanning.set(false)
        }
    }

    private val gattCallback = object : BluetoothGattCallback() {
        override fun onConnectionStateChange(gatt: BluetoothGatt, status: Int, newState: Int) {
            when (newState) {
                BluetoothProfile.STATE_CONNECTED -> {
                    Log.i(TAG, "Connected to ${gatt.device.address}")
                    lastConnectedAddress = gatt.device.address
                    lastConnectedDeviceName = gatt.device.name
                    _connectionState.value = ConnectionState.Connected(gatt.device.name, gatt.device.address)
                    gatt.requestMtu(512)
                }
                BluetoothProfile.STATE_DISCONNECTED -> {
                    Log.i(TAG, "Disconnected from ${gatt.device.address} (status=$status)")
                    this@BleTranslatorService.gatt = null
                    textDownChar = null
                    _connectionState.value = ConnectionState.Disconnected
                    if (autoReconnect && lastConnectedAddress != null) {
                        scheduleReconnect()
                    }
                }
            }
        }

        override fun onMtuChanged(gatt: BluetoothGatt, mtu: Int, status: Int) {
            Log.i(TAG, "MTU changed to $mtu")
            gatt.discoverServices()
        }

        override fun onServicesDiscovered(gatt: BluetoothGatt, status: Int) {
            if (status != BluetoothGatt.GATT_SUCCESS) {
                _connectionState.value = ConnectionState.Error("Service discovery failed: $status")
                return
            }
            val service = gatt.getService(BleUuid.SERVICE)
            if (service == null) {
                _connectionState.value = ConnectionState.Error("Translator service not found")
                return
            }
            textDownChar = service.getCharacteristic(BleUuid.TEXT_DOWN)
            service.getCharacteristic(BleUuid.AUDIO_UP)?.let { audioUp ->
                enableNotification(gatt, audioUp)
            }
            _connectionState.value = ConnectionState.Ready(lastConnectedDeviceName, lastConnectedAddress)
        }

        override fun onCharacteristicChanged(
            gatt: BluetoothGatt,
            characteristic: BluetoothGattCharacteristic,
            value: ByteArray
        ) {
            when (characteristic.uuid) {
                BleUuid.AUDIO_UP -> _receivedAudio.trySend(value)
                else -> Log.d(TAG, "Unknown char changed: ${characteristic.uuid}")
            }
        }

        @Suppress("DEPRECATION")
        override fun onCharacteristicWrite(
            gatt: BluetoothGatt,
            characteristic: BluetoothGattCharacteristic,
            status: Int
        ) {
            if (status != BluetoothGatt.GATT_SUCCESS) {
                Log.w(TAG, "Write failed for ${characteristic.uuid}: $status")
            }
        }
    }

    @SuppressLint("MissingPermission")
    fun startScan(targetAddress: String? = null) {
        pendingConnectAddress = targetAddress
        lastConnectedAddress = targetAddress ?: lastConnectedAddress
        _connectionState.value = ConnectionState.Scanning
        isScanning.set(true)
        val filters = listOf(
            ScanFilter.Builder()
                .setServiceUuid(ParcelUuid(BleUuid.SERVICE))
                .build()
        )
        val settings = ScanSettings.Builder()
            .setScanMode(ScanSettings.SCAN_MODE_LOW_LATENCY)
            .build()
        scanner?.startScan(filters, settings, scanCallback)
    }

    private fun scheduleReconnect() {
        reconnectJob?.cancel()
        reconnectJob = scope.launch {
            delay(reconnectDelayMs)
            if (_connectionState.value is ConnectionState.Disconnected) {
                Log.i(TAG, "Auto-reconnecting to $lastConnectedAddress")
                startScan(lastConnectedAddress)
            }
        }
    }

    @SuppressLint("MissingPermission")
    fun stopScan() {
        if (isScanning.compareAndSet(true, false)) {
            scanner?.stopScan(scanCallback)
        }
    }

    @SuppressLint("MissingPermission")
    fun connect(device: BluetoothDevice) {
        stopScan()
        lastConnectedAddress = device.address
        lastConnectedDeviceName = device.name
        _connectionState.value = ConnectionState.Connecting
        gatt = device.connectGatt(appContext, false, gattCallback)
    }

    @SuppressLint("MissingPermission")
    fun disconnect() {
        reconnectJob?.cancel()
        autoReconnectEnabled = false
        stopScan()
        gatt?.disconnect()
        gatt?.close()
        gatt = null
        textDownChar = null
        lastConnectedAddress = null
        lastConnectedDeviceName = null
        _connectionState.value = ConnectionState.Disconnected
    }

    private var autoReconnectEnabled = autoReconnect

    fun reconnectNow() {
        autoReconnectEnabled = autoReconnect
        if (lastConnectedAddress != null) {
            startScan(lastConnectedAddress)
        }
    }

    @SuppressLint("MissingPermission")
    fun sendText(text: String): Boolean {
        val gatt = this.gatt ?: return false
        val char = textDownChar ?: return false
        val bytes = text.toByteArray(Charsets.UTF_8)
        if (bytes.isEmpty()) return false

        return if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.TIRAMISU) {
            sendTextChunkedTiramisu(gatt, char, bytes)
        } else {
            sendTextChunkedLegacy(gatt, char, bytes)
        }
    }

    @SuppressLint("MissingPermission")
    private fun sendTextChunkedTiramisu(gatt: BluetoothGatt, char: BluetoothGattCharacteristic, bytes: ByteArray): Boolean {
        var offset = 0
        var success = true
        while (offset < bytes.size && success) {
            val end = (offset + MAX_CHUNK_SIZE).coerceAtMost(bytes.size)
            val chunk = bytes.copyOfRange(offset, end)
            val status = gatt.writeCharacteristic(char, chunk, BluetoothGattCharacteristic.WRITE_TYPE_DEFAULT)
            if (status != BluetoothGatt.GATT_SUCCESS) {
                Log.w(TAG, "Write failed at offset $offset: $status")
                success = false
            }
            offset = end
            if (offset < bytes.size) {
                Thread.sleep(WRITE_DELAY_MS)
            }
        }
        val endMarker = byteArrayOf(0x04)
        val endStatus = gatt.writeCharacteristic(char, endMarker, BluetoothGattCharacteristic.WRITE_TYPE_DEFAULT)
        return success && endStatus == BluetoothGatt.GATT_SUCCESS
    }

    @Suppress("DEPRECATION")
    @SuppressLint("MissingPermission")
    private fun sendTextChunkedLegacy(gatt: BluetoothGatt, char: BluetoothGattCharacteristic, bytes: ByteArray): Boolean {
        var offset = 0
        var success = true
        while (offset < bytes.size && success) {
            val end = (offset + MAX_CHUNK_SIZE).coerceAtMost(bytes.size)
            char.value = bytes.copyOfRange(offset, end)
            success = gatt.writeCharacteristic(char)
            offset = end
            if (offset < bytes.size) {
                Thread.sleep(WRITE_DELAY_MS)
            }
        }
        char.value = byteArrayOf(0x04)
        return success && gatt.writeCharacteristic(char)
    }

    @SuppressLint("MissingPermission")
    private fun enableNotification(gatt: BluetoothGatt, characteristic: BluetoothGattCharacteristic) {
        gatt.setCharacteristicNotification(characteristic, true)
        val descriptor = characteristic.getDescriptor(UUID.fromString("00002902-0000-1000-8000-00805f9b34fb"))
        descriptor?.let {
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.TIRAMISU) {
                gatt.writeDescriptor(it, BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE)
            } else {
                @Suppress("DEPRECATION")
                it.value = BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE
                @Suppress("DEPRECATION")
                gatt.writeDescriptor(it)
            }
        }
    }

    sealed class ConnectionState {
        object Disconnected : ConnectionState()
        object Scanning : ConnectionState()
        object Connecting : ConnectionState()
        data class Connected(val deviceName: String?, val address: String?) : ConnectionState()
        data class Ready(val deviceName: String?, val address: String?) : ConnectionState()
        data class Error(val message: String) : ConnectionState()
    }

    companion object {
        private const val MAX_CHUNK_SIZE = 200
        private const val WRITE_DELAY_MS = 20L
    }
}
