#pragma once

#include <stdint.h>
#include <stddef.h>

#define SAMPLE_RATE     16000
#define FRAME_MS        20
#define SAMPLES_PER_FRAME   (SAMPLE_RATE * FRAME_MS / 1000)
#define BYTES_PER_FRAME     (SAMPLES_PER_FRAME * sizeof(int16_t))

void i2s_mic_init(void);
int i2s_mic_read(int16_t *samples, size_t max_samples);
