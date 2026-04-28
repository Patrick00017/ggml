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
#include <iostream>

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

struct icls_cls_head {
    // ("fc1", nn.Linear(1024, 512)),
    // ("relu1", nn.ReLU()),
    // ("fc2", nn.Linear(512, 256)),
    // ("relu2", nn.ReLU()),
    // ("fc3", nn.Linear(256, num_classes)),
    // ("output", nn.LogSoftmax(dim=1)),
    struct ggml_tensor* fc1_weight = nullptr;
    struct ggml_tensor* fc1_bias = nullptr;
    struct ggml_tensor* fc2_weight = nullptr;
    struct ggml_tensor* fc2_bias = nullptr;
    struct ggml_tensor* fc3_weight = nullptr;
    struct ggml_tensor* fc3_bias = nullptr;
};

struct icls_backbone {
    conv2d_layer conv1;

    std::vector<conv2d_layer> layer1_0;
    std::vector<conv2d_layer> layer1_1;
    std::vector<conv2d_layer> layer1_2;

    std::vector<conv2d_layer> layer2_0;
    std::vector<conv2d_layer> layer2_1;
    std::vector<conv2d_layer> layer2_2;
    std::vector<conv2d_layer> layer2_3;

    std::vector<conv2d_layer> layer3_0;
    std::vector<conv2d_layer> layer3_1;
    std::vector<conv2d_layer> layer3_2;
    std::vector<conv2d_layer> layer3_3;
    std::vector<conv2d_layer> layer3_4;
    std::vector<conv2d_layer> layer3_5;

    std::vector<conv2d_layer> layer4_0;
    std::vector<conv2d_layer> layer4_1;
    std::vector<conv2d_layer> layer4_2;

    icls_cls_head classifer;
};

struct icls_model {
    icls_backbone backbone;
    icls_cls_head head;

    ggml_backend_t backend;
    ggml_backend_buffer_t buffer;
    struct ggml_context *ctx;
};

struct icls_params {
    std::string model = "D:/code/forked_project/ggml/examples/image_classification/weights/icls.gguf";
    std::string fname_inp = "input.jpg";
    std::string fname_out = "predictions.jpg";
    int         n_threads = std::max(1U, std::thread::hardware_concurrency() / 2);
    std::string device;
};

// load the model's weights from a file
bool icls_model_init_from_file(const std::string &fname, icls_model &model);
bool icls_image_load(const std::string & fname);
void icls_inference(icls_model &model, const std::string &image_path);


static void print_shape(int layer, const ggml_tensor * t)
{
    printf("Layer %2d output shape:  %3d x %3d x %4d x %3d\n", layer, (int)t->ne[0], (int)t->ne[1], (int)t->ne[2], (int)t->ne[3]);
}
