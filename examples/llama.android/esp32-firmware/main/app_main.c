#include "i2s_mic.h"
#include "opus_encoder.h"
#include "ble_server.h"
#include "tts_lu6288.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_log.h"

#define TAG "app"

#define OPUS_MAX_PACKET 150
#define AUDIO_QUEUE_LEN 32
#define TEXT_BUF_LEN    512

typedef struct {
    int len;
    uint8_t data[OPUS_MAX_PACKET];
} audio_packet_t;

static QueueHandle_t audio_queue;

static void audio_task(void *arg) {
    int16_t pcm[SAMPLES_PER_FRAME];
    uint8_t opus_buf[OPUS_MAX_PACKET];

    while (1) {
        int samples = i2s_mic_read(pcm, SAMPLES_PER_FRAME);
        if (samples != SAMPLES_PER_FRAME) {
            vTaskDelay(pdMS_TO_TICKS(1));
            continue;
        }

        int encoded = opus_encoder_encode(pcm, opus_buf, OPUS_MAX_PACKET);
        if (encoded > 0) {
            audio_packet_t pkt = { .len = encoded };
            memcpy(pkt.data, opus_buf, encoded);
            xQueueSend(audio_queue, &pkt, 0);
        }
    }
}

static void ble_tx_task(void *arg) {
    audio_packet_t pkt;
    while (1) {
        if (xQueueReceive(audio_queue, &pkt, portMAX_DELAY) == pdTRUE) {
            if (ble_server_is_connected()) {
                // Send multiple small frames together if possible
                ble_server_notify_audio(pkt.data, pkt.len);
            }
        }
    }
}

static void tts_task(void *arg) {
    char text[TEXT_BUF_LEN];
    while (1) {
        if (ble_server_get_received_text(text, sizeof(text))) {
            tts_lu6288_send_text(text);
        }
        vTaskDelay(pdMS_TO_TICKS(50));
    }
}

void app_main(void) {
    ESP_LOGI(TAG, "Translator ESP32-S3 firmware starting");

    audio_queue = xQueueCreate(AUDIO_QUEUE_LEN, sizeof(audio_packet_t));
    assert(audio_queue != NULL);

    i2s_mic_init();
    opus_encoder_init();
    tts_lu6288_init();
    ble_server_init();

    xTaskCreatePinnedToCore(audio_task, "audio", 4096, NULL, 5, NULL, 1);
    xTaskCreatePinnedToCore(ble_tx_task, "ble_tx", 4096, NULL, 4, NULL, 0);
    xTaskCreatePinnedToCore(tts_task, "tts", 4096, NULL, 3, NULL, 0);

    ESP_LOGI(TAG, "Tasks started");
}
