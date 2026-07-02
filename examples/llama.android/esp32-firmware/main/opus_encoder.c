#include "opus_encoder.h"
#include "i2s_mic.h"
#include "esp_log.h"
#include "opus.h"

#define TAG "opus_enc"

static OpusEncoder *encoder = NULL;

int opus_encoder_init(void) {
    int err = 0;
    encoder = opus_encoder_create(SAMPLE_RATE, 1, OPUS_APPLICATION_VOIP, &err);
    if (err != OPUS_OK) {
        ESP_LOGE(TAG, "opus_encoder_create failed: %d", err);
        return -1;
    }
    opus_encoder_ctl(encoder, OPUS_SET_BITRATE(16000));
    opus_encoder_ctl(encoder, OPUS_SET_COMPLEXITY(5));
    opus_encoder_ctl(encoder, OPUS_SET_SIGNAL(OPUS_SIGNAL_VOICE));
    opus_encoder_ctl(encoder, OPUS_SET_VBR(0));
    ESP_LOGI(TAG, "OPUS encoder initialized");
    return 0;
}

void opus_encoder_deinit(void) {
    if (encoder) {
        opus_encoder_destroy(encoder);
        encoder = NULL;
    }
}

int opus_encoder_encode(const int16_t *pcm, unsigned char *out_buf, int max_bytes) {
    if (!encoder) return -1;
    opus_int32 nb = opus_encode(encoder, pcm, SAMPLES_PER_FRAME, out_buf, max_bytes);
    if (nb < 0) {
        ESP_LOGE(TAG, "opus_encode failed: %d", (int)nb);
        return -1;
    }
    return (int)nb;
}
