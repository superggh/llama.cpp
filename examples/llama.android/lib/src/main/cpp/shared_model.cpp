#include "shared_model.h"
#include <android/log.h>

#define LOG_TAG "SharedModel"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)

static llama_model *g_shared_model = nullptr;
static int g_ref_count = 0;

llama_model *load_shared_model(const char *path) {
    if (!g_shared_model) {
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        g_shared_model = llama_model_load_from_file(path, mp);
        if (g_shared_model) {
            LOGI("Loaded shared model from %s", path);
        } else {
            LOGW("Failed to load shared model from %s", path);
            return nullptr;
        }
    } else {
        LOGI("Reusing existing shared model");
    }
    g_ref_count++;
    return g_shared_model;
}

llama_model *get_shared_model() {
    return g_shared_model;
}

void release_shared_model() {
    if (g_shared_model) {
        g_ref_count--;
        LOGI("Release shared model, ref_count=%d", g_ref_count);
        if (g_ref_count <= 0) {
            llama_model_free(g_shared_model);
            g_shared_model = nullptr;
            g_ref_count = 0;
            LOGI("Shared model freed");
        }
    }
}
