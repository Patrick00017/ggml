#define _USE_MATH_DEFINES // for M_PI
#define _CRT_SECURE_NO_DEPRECATE // Disables ridiculous "unsafe" warnigns on Windows

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"
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

struct conv2d_layer {
    struct ggml_tensor * weights;
    struct ggml_tensor * biases;
    struct ggml_tensor * scales;
    struct ggml_tensor * rolling_mean;
    struct ggml_tensor * rolling_variance;
    int padding = 1;
    bool batch_normalize = true;
    bool activate = true; // true for leaky relu, false for linear
};

struct icls_backbone {
    std::vector<conv2d_layer> layers;
};

struct icls_cls_head {
    // ("fc1", nn.Linear(1024, 512)),
    // ("relu1", nn.ReLU()),
    // ("fc2", nn.Linear(512, 256)),
    // ("relu2", nn.ReLU()),
    // ("fc3", nn.Linear(256, num_classes)),
    // ("output", nn.LogSoftmax(dim=1)),
    struct ggml_tensor * fc1_weight = nullptr;
    struct ggml_tensor * fc1_bias   = nullptr;
    struct ggml_tensor * fc2_weight = nullptr;
    struct ggml_tensor * fc2_bias   = nullptr;
    struct ggml_tensor * fc3_weight = nullptr;
    struct ggml_tensor * fc3_bias   = nullptr;
};

struct icls_model {
    icls_backbone backbone;
    icls_cls_head head;

    ggml_backend_t backend;
    ggml_backend_buffer_t buffer;
    struct ggml_context *ctx;
};

struct icls_params {
    std::string model_path = "";
    std::string image_input_path = "";
    std::string result_output_path = "";

    std::string backend = "cpu";
};

// load the model's weights from a file
bool icls_model_init_from_file(const std::string &fname, icls_model &model);
bool icls_image_load(const std::string & fname);
void icls_inference(icls_model &model, const std::string &image_path);
