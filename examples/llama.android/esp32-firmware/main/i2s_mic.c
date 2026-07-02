#include "i2s_mic.h"
#include "esp_log.h"
#include "driver/i2s_std.h"
#include "driver/gpio.h"

#define TAG "i2s_mic"

#define I2S_MIC_BCLK    GPIO_NUM_6
#define I2S_MIC_WS      GPIO_NUM_7
#define I2S_MIC_DIN     GPIO_NUM_8

static i2s_chan_handle_t rx_chan = NULL;

void i2s_mic_init(void) {
    i2s_chan_config_t chan_cfg = {
        .id = I2S_NUM_0,
        .role = I2S_ROLE_MASTER,
        .dma_desc_num = 6,
        .dma_frame_num = 240,
        .auto_clear = true,
    };
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &rx_chan));

    i2s_std_config_t std_cfg = {
        .clk_cfg = {
            .sample_rate_hz = SAMPLE_RATE,
            .clk_src = I2S_CLK_SRC_DEFAULT,
            .mclk_multiple = I2S_MCLK_MULTIPLE_256,
        },
        .slot_cfg = {
            .slot_mode = I2S_SLOT_MODE_MONO,
            .slot_mask = I2S_STD_SLOT_LEFT,
            .ws_width = I2S_DATA_BIT_WIDTH_16BIT,
            .data_bit_width = I2S_DATA_BIT_WIDTH_16BIT,
            .ws_pol = false,
            .bit_shift = true,
            .left_align = true,
            .big_endian = false,
            .bit_order_lsb = false,
        },
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = I2S_MIC_BCLK,
            .ws = I2S_MIC_WS,
            .dout = I2S_GPIO_UNUSED,
            .din = I2S_MIC_DIN,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv = false,
            },
        },
    };
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx_chan, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(rx_chan));
    ESP_LOGI(TAG, "I2S mic initialized");
}

int i2s_mic_read(int16_t *samples, size_t max_samples) {
    size_t bytes_read = 0;
    size_t bytes_to_read = max_samples * sizeof(int16_t);
    esp_err_t err = i2s_channel_read(rx_chan, samples, bytes_to_read, &bytes_read, portMAX_DELAY);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "I2S read failed: %d", err);
        return 0;
    }
    return bytes_read / sizeof(int16_t);
}
