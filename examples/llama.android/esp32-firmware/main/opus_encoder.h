#pragma once

#include <stdint.h>
#include <stddef.h>

int opus_encoder_init(void);
void opus_encoder_deinit(void);
int opus_encoder_encode(const int16_t *pcm, unsigned char *out_buf, int max_bytes);
