#include <jni.h>
#include <android/log.h>
#include <string.h>
#include "opus.h"

#define TAG "OpusJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_arm_aichat_audio_OpusDecoder_nativeCreate(
        JNIEnv *env, jobject thiz, jint sample_rate, jint channels) {
    int err = 0;
    OpusDecoder *decoder = opus_decoder_create(sample_rate, channels, &err);
    if (err != OPUS_OK) {
        LOGW("opus_decoder_create failed: %d", err);
        return 0;
    }
    return reinterpret_cast<jlong>(decoder);
}

JNIEXPORT void JNICALL
Java_com_arm_aichat_audio_OpusDecoder_nativeDestroy(
        JNIEnv *env, jobject thiz, jlong handle) {
    OpusDecoder *decoder = reinterpret_cast<OpusDecoder *>(handle);
    if (decoder) {
        opus_decoder_destroy(decoder);
    }
}

JNIEXPORT jint JNICALL
Java_com_arm_aichat_audio_OpusDecoder_nativeDecode(
        JNIEnv *env, jobject thiz, jlong handle,
        jbyteArray opus_data, jint opus_len,
        jshortArray pcm_out, jint frame_size) {
    OpusDecoder *decoder = reinterpret_cast<OpusDecoder *>(handle);
    if (!decoder) return -1;

    jbyte *opus_bytes = env->GetByteArrayElements(opus_data, NULL);
    jshort *pcm_samples = env->GetShortArrayElements(pcm_out, NULL);

    int decoded = opus_decode(decoder,
                              reinterpret_cast<const unsigned char *>(opus_bytes),
                              opus_len,
                              pcm_samples,
                              frame_size,
                              0);

    env->ReleaseByteArrayElements(opus_data, opus_bytes, JNI_ABORT);
    env->ReleaseShortArrayElements(pcm_out, pcm_samples, 0);

    return decoded;
}

} // extern "C"
