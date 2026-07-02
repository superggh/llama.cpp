#pragma once

#include <stdbool.h>
#include <stdint.h>

void ble_server_init(void);
bool ble_server_is_connected(void);
bool ble_server_notify_audio(const uint8_t *data, uint16_t len);
bool ble_server_get_received_text(char *out_buf, uint16_t max_len);
void ble_server_set_mtu(uint16_t mtu);
