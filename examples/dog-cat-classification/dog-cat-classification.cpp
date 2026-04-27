#define _USE_MATH_DEFINES // for M_PI
#define _CRT_SECURE_NO_DEPRECATE // Disables ridiculous "unsafe" warnigns on Windows

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <thread>
#include <cinttypes>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

struct icls_hparams {
    // define the hparams for icls model
    // int32_t num_hidden = 128;
    // int32_t num_classes = 2;

    int32_t ftype = GGML_FTYPE_ALL_F32; // no quantization
};

struct icls_backbone_layer {
    // struct ggml_tensor * norm1_w;
    // struct ggml_tensor * norm1_b;
};

struct icls_backbone {
    std::vector<icls_backbone> layers;
};

struct icls_cls_head {
    // struct ggml_tensor * norm1_w;
    // struct ggml_tensor * norm1_b;
};

struct icls_state {
    struct ggml_context *ctx;
    ggml_gallocr_t       allocr = {};
};

struct icls_model {
    icls_hparams hparams;

    icls_backbone backbone;
    icls_cls_head head;

    struct ggml_context *ctx;
    std::map<std::string, struct ggml_tensor *> tensors;
};

struct icls_params {
    std::string model_path = "";
    std::string image_input_path = "";
    std::string result_output_path = "";
};

// load the model's weights from a file
bool icls_model_load(const icls_params & params, icls_model & model) {
    // load icls model weights
    return true;
}

int main(int argc, char ** argv){
    ggml_time_init();
    const int64_t t_main_start_us = ggml_time_us();
}