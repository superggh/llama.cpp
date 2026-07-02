cmake_minimum_required(VERSION 3.22.1)

project("ai-chat" VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}"   CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "" FORCE)

# --------------------------------------------------------------------------
# AI Chat library
# --------------------------------------------------------------------------

if(DEFINED ANDROID_ABI)
    message(STATUS "Detected Android ABI: ${ANDROID_ABI}")
    if(ANDROID_ABI STREQUAL "arm64-v8a")
        set(GGML_SYSTEM_ARCH "ARM")
        if(NOT DEFINED GGML_CPU_KLEIDIAI)
            set(GGML_CPU_KLEIDIAI ON)
        endif()
        set(GGML_OPENMP ON)
    elseif(ANDROID_ABI STREQUAL "x86_64")
        set(GGML_SYSTEM_ARCH "x86")
        set(GGML_CPU_KLEIDIAI OFF)
        set(GGML_OPENMP OFF)
    else()
        message(FATAL_ERROR "Unsupported ABI: ${ANDROID_ABI}")
    endif()
endif()

set(LLAMA_SRC ${CMAKE_CURRENT_LIST_DIR}/../../../../../../)
set(WHISPER_SRC ${CMAKE_CURRENT_LIST_DIR}/../../../../../../../whisper.cpp)
set(OPUS_SRC ${CMAKE_CURRENT_LIST_DIR}/../../../../../../../opus)

add_subdirectory(${LLAMA_SRC} build-llama)

set(OPUS_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(OPUS_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
set(OPUS_BUILD_SHARED_LIBRARY OFF CACHE BOOL "" FORCE)
set(OPUS_DISABLE_INTRINSICS ON CACHE BOOL "" FORCE)

if(EXISTS ${OPUS_SRC})
    message(STATUS "opus found at ${OPUS_SRC}")
    add_subdirectory(${OPUS_SRC} build-opus)
else()
    message(WARNING "opus not found at ${OPUS_SRC}, building without opus support")
endif()

if(EXISTS ${WHISPER_SRC})
    message(STATUS "whisper.cpp found at ${WHISPER_SRC}")
    add_library(${CMAKE_PROJECT_NAME} SHARED
            ai_chat.cpp
            shared_model.cpp
            whisper_jni.cpp
            opus_jni.cpp
            funasr_jni.cpp
            ${WHISPER_SRC}/src/whisper.cpp)

    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE
            GGML_SYSTEM_ARCH=${GGML_SYSTEM_ARCH}
            GGML_CPU_KLEIDIAI=$<BOOL:${GGML_CPU_KLEIDIAI}>
            GGML_OPENMP=$<BOOL:${GGML_OPENMP}>
            WHISPER_VERSION="1.9.1"
    )

    target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
            ${LLAMA_SRC}
            ${LLAMA_SRC}/common
            ${LLAMA_SRC}/include
            ${LLAMA_SRC}/ggml/include
            ${LLAMA_SRC}/ggml/src
            ${WHISPER_SRC}
            ${WHISPER_SRC}/include
            ${WHISPER_SRC}/src
            ${WHISPER_SRC}/ggml/include
            ${WHISPER_SRC}/ggml/src
            ${OPUS_SRC}/include)

    target_link_libraries(${CMAKE_PROJECT_NAME}
            llama
            llama-common
            opus
            android
            log)
else()
    message(WARNING "whisper.cpp not found at ${WHISPER_SRC}, building without whisper support")
    add_library(${CMAKE_PROJECT_NAME} SHARED
            ai_chat.cpp
            opus_jni.cpp)

    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE
            GGML_SYSTEM_ARCH=${GGML_SYSTEM_ARCH}
            GGML_CPU_KLEIDIAI=$<BOOL:${GGML_CPU_KLEIDIAI}>
            GGML_OPENMP=$<BOOL:${GGML_OPENMP}>
            WHISPER_VERSION="1.9.1"
    )

    target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
            ${LLAMA_SRC}
            ${LLAMA_SRC}/common
            ${LLAMA_SRC}/include
            ${LLAMA_SRC}/ggml/include
            ${LLAMA_SRC}/ggml/src
            ${OPUS_SRC}/include)

    target_link_libraries(${CMAKE_PROJECT_NAME}
            llama
            llama-common
            opus
            android
            log)
endif()
