#ifndef SHARED_MODEL_H
#define SHARED_MODEL_H

#include "llama.h"

llama_model *load_shared_model(const char *path);
llama_model *get_shared_model();
void release_shared_model();

#endif
