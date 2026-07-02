#include "tts_lu6288.h"
#include "driver/uart.h"
#include "esp_log.h"
#include "string.h"

#define TAG "tts_lu6288"

#define TTS_UART_NUM    UART_NUM_1
#define TTS_TXD_PIN     GPIO_NUM_17
#define TTS_RXD_PIN     GPIO_NUM_18
#define TTS_BUF_SIZE    512

// Common control words for LU6288
static const uint8_t tts_prefix[] = {0xFD};
static const uint8_t tts_encoding_gb2312[] = {0x01};

void tts_lu6288_init(void) {
    uart_config_t uart_cfg = {
        .baud_rate = 115200,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .rx_flow_ctrl_thresh = 0,
        .source_clk = UART_SCLK_DEFAULT,
    };
    ESP_ERROR_CHECK(uart_param_config(TTS_UART_NUM, &uart_cfg));
    ESP_ERROR_CHECK(uart_set_pin(TTS_UART_NUM, TTS_TXD_PIN, TTS_RXD_PIN, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));
    ESP_ERROR_CHECK(uart_driver_install(TTS_UART_NUM, TTS_BUF_SIZE, TTS_BUF_SIZE, 0, NULL, 0));
    ESP_LOGI(TAG, "LU6288 UART initialized");
}

void tts_lu6288_send_text(const char *text) {
    if (text == NULL || text[0] == '\0') return;

    uint16_t text_len = strlen(text);
    uint16_t frame_len = text_len + 2; // encoding byte + text

    uint8_t frame[TTS_BUF_SIZE];
    frame[0] = 0xFD;
    frame[1] = (frame_len >> 8) & 0xFF;
    frame[2] = frame_len & 0xFF;
    frame[3] = 0x01; // GB2312 encoding
    memcpy(frame + 4, text, text_len);

    uart_write_bytes(TTS_UART_NUM, frame, 4 + text_len);
    ESP_LOGI(TAG, "TTS sent: %s", text);
}
