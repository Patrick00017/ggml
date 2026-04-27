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
    ggml_backend_sched_t backend_sched;
    std::vector<ggml_backend_t> backends;

    struct ggml_tensor * images     = nullptr;
    struct ggml_tensor * logits     = nullptr;

    icls_backbone backbone;
    icls_cls_head head;

    struct ggml_context *ctx;
    struct ggml_context * ctx_gguf    = nullptr;
    struct ggml_context * ctx_static  = nullptr;
    struct ggml_context * ctx_compute = nullptr;
    ggml_backend_buffer_t buf_gguf    = nullptr;
    ggml_backend_buffer_t buf_static  = nullptr;
    std::map<std::string, struct ggml_tensor *> tensors;

    icls_model(const std::string & backend_name)
    {
        std::vector<ggml_backend_dev_t> devices;
        const int ncores_logical = std::thread::hardware_concurrency();
        const int nthreads = std::min(ncores_logical, (ncores_logical + 4) / 2);

        // Add primary backend:
        if (!backend_name.empty()) {
            ggml_backend_dev_t dev = ggml_backend_dev_by_name(backend_name.c_str());
            if (dev == nullptr) {
                fprintf(stderr, "%s: ERROR: backend %s not found, available:\n", __func__, backend_name.c_str());
                for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                    ggml_backend_dev_t dev_i = ggml_backend_dev_get(i);
                    fprintf(stderr, "  - %s (%s)\n", ggml_backend_dev_name(dev_i), ggml_backend_dev_description(dev_i));
                }
                exit(1);
            }

            ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
            GGML_ASSERT(backend);

            if (ggml_backend_is_cpu(backend)) {
                ggml_backend_cpu_set_n_threads(backend, nthreads);
            }

            backends.push_back(backend);
            devices.push_back(dev);
        }

        // Add all available backends as fallback.
        // A "backend" is a stream on a physical device so there is no problem with adding multiple backends for the same device.
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);

            ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
            GGML_ASSERT(backend);

            if (ggml_backend_is_cpu(backend)) {
                ggml_backend_cpu_set_n_threads(backend, nthreads);
            }

            backends.push_back(backend);
            devices.push_back(dev);
        }

        // The order of the backends passed to ggml_backend_sched_new determines which backend is given priority.
        backend_sched = ggml_backend_sched_new(backends.data(), nullptr, backends.size(), GGML_DEFAULT_GRAPH_SIZE, false, true);
        fprintf(stderr, "%s: using %s (%s) as primary backend\n",
                __func__, ggml_backend_name(backends[0]), ggml_backend_dev_description(devices[0]));
        if (backends.size() >= 2) {
            fprintf(stderr, "%s: unsupported operations will be executed on the following fallback backends (in order of priority):\n", __func__);
            for (size_t i = 1; i < backends.size(); ++i) {
                fprintf(stderr, "%s:  - %s (%s)\n", __func__, ggml_backend_name(backends[i]), ggml_backend_dev_description(devices[i]));
            }
        }

        {
            const size_t size_meta = 1024*ggml_tensor_overhead();
            struct ggml_init_params params = {
                /*.mem_size   =*/ size_meta,
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ctx_static = ggml_init(params);
        }

        {
            // The compute context needs a total of 3 compute graphs: forward pass + backwards pass (with/without optimizer step).
            const size_t size_meta = GGML_DEFAULT_GRAPH_SIZE*ggml_tensor_overhead() + 3*ggml_graph_overhead();
            struct ggml_init_params params = {
                /*.mem_size   =*/ size_meta,
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ctx_compute = ggml_init(params);
        }
    }

    ~icls_model() {
        ggml_free(ctx_gguf);
        ggml_free(ctx_static);
        ggml_free(ctx_compute);

        ggml_backend_buffer_free(buf_gguf);
        ggml_backend_buffer_free(buf_static);
        ggml_backend_sched_free(backend_sched);
        for (ggml_backend_t backend : backends) {
            ggml_backend_free(backend);
        }
    }
};

struct icls_params {
    std::string model_path = "";
    std::string image_input_path = "";
    std::string result_output_path = "";
};

// load the model's weights from a file
icls_model icls_model_init_from_file(const icls_params & params, icls_model & model);
bool icls_image_load(const std::string & fname);
void icls_inference(icls_model &model, const std::string &image_path);

