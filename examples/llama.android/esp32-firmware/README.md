# ESP32-S3 Translator Firmware

ESP-IDF project for the translator hardware.

## Hardware

- ESP32-S3-N6R8 (or any ESP32-S3 with PSRAM)
- INMP441 I2S MEMS microphone
- JDY-23 **not needed** - ESP32-S3 has built-in BLE 5.0
- LU6288 TTS module

## Wiring

### INMP441
```
INMP441      ESP32-S3
VDD     ->   3.3V
GND     ->   GND
SCK     ->   GPIO 6 (BCLK)
WS      ->   GPIO 7 (LRCK)
L/R     ->   GND (left channel)
SD      ->   GPIO 8 (DIN)
```

### LU6288
```
LU6288       ESP32-S3
VCC     ->   3.3V
GND     ->   GND
RX      ->   GPIO 17 (TX)
TX      ->   GPIO 18 (RX)
```

## Build & Flash

```bash
cd examples/llama.android/esp32-firmware
idf.py set-target esp32s3
idf.py add-dependency "espressif/opus^1.5.2"
idf.py build
idf.py flash monitor
```

## Data Flow

```
INMP441 -> I2S -> OPUS encoder -> BLE notify -> Android
Android -> BLE write -> ESP32 -> UART -> LU6288 -> speaker
```

## BLE Service

| Characteristic | UUID | Direction | Properties |
|---|---|---|---|
| Audio Up | 0xABCD1001 | ESP32 -> Android | NOTIFY |
| Text Down | 0xABCD1002 | Android -> ESP32 | WRITE |
| Control | 0xABCD1003 | Both | WRITE/INDICATE |

## Notes

- Audio: 16kHz, mono, 16-bit PCM
- OPUS: 16kbps CBR, 20ms frames (~40 bytes per packet)
- BLE device name: `Translator-S3`
