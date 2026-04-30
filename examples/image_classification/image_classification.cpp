#include "image_classification.h"

#define GGUF_MODEL_PATH 

static ggml_backend_t create_backend(const icls_params& params) {
    ggml_backend_t backend = nullptr;

    if (!params.device.empty()) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_name(params.device.c_str());
        if (dev) {
            backend = ggml_backend_dev_init(dev, nullptr);
            if (!backend) {
                fprintf(stderr, "Failed to create backend for device %s\n", params.device.c_str());
                return nullptr;
            }
        }
    }

    // try to initialize a GPU backend first
    if (!backend) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    }

    // if there aren't GPU backends fallback to CPU backend
    if (!backend) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }

    if (backend) {
        fprintf(stderr, "%s: using %s backend\n", __func__, ggml_backend_name(backend));

        // set the number of threads
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (reg) {
            auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t)ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            if (ggml_backend_set_n_threads_fn) {
                ggml_backend_set_n_threads_fn(backend, params.n_threads);
            }
        }
    }

    return backend;
}

// --------------------- impl -----------------
bool icls_model_init_from_file(const std::string &fname, icls_model &model){
    struct ggml_context * tmp_ctx = nullptr;
    struct gguf_init_params gguf_params = {
        /*.no_alloc   =*/ false,
        /*.ctx        =*/ &tmp_ctx,
    };
    gguf_context * gguf_ctx = gguf_init_from_file(fname.c_str(), gguf_params);
    if (!gguf_ctx) {
        fprintf(stderr, "%s: gguf_init_from_file() failed\n", __func__);
        return false;
    }

    int num_tensors = gguf_get_n_tensors(gguf_ctx);
    struct ggml_init_params params {
            /*.mem_size   =*/ ggml_tensor_overhead() * num_tensors,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
    };
    model.ctx = ggml_init(params);
    for (int i = 0; i < num_tensors; i++) {
        const char * name = gguf_get_tensor_name(gguf_ctx, i);
        struct ggml_tensor * src = ggml_get_tensor(tmp_ctx, name);
        struct ggml_tensor * dst = ggml_dup_tensor(model.ctx, src);
        ggml_set_name(dst, name);

        // add some log
        std::cout << "src: " << src->name << " dst: " << dst->name << std::endl;
    }
    model.buffer = ggml_backend_alloc_ctx_tensors(model.ctx, model.backend);
    // copy tensors from main memory to backend
    for (struct ggml_tensor * cur = ggml_get_first_tensor(model.ctx); cur != NULL; cur = ggml_get_next_tensor(model.ctx, cur)) {
        struct ggml_tensor * src = ggml_get_tensor(tmp_ctx, ggml_get_name(cur));
        size_t n_size = ggml_nbytes(src);
        ggml_backend_tensor_set(cur, ggml_get_data(src), 0, n_size);
    }
    gguf_free(gguf_ctx);
    ggml_free(tmp_ctx);
    
    // load model.ctx tensors into backbone
    model.backbone.conv1.padding = 3;
    model.backbone.conv1.stride = 2;
    model.backbone.conv1.activate = true;

    // layer1.0
    model.backbone.layer1_0.resize(4); // 3 conv2d and 1 downsample
    model.backbone.layer1_0[1].padding = 1;
    model.backbone.layer1_0[2].activate = true;
    for (int i = 0; i < (int)model.backbone.layer1_0.size() - 1; i++) {
        char name[256];
        int index = i + 1;
        snprintf(name, sizeof(name), "layer1.0.conv%d.weight", index);
        model.backbone.layer1_0[i].weights = ggml_get_tensor(model.ctx, name);
        if (model.backbone.layer1_0[i].batch_normalize) {
            snprintf(name, sizeof(name), "layer1.0.bn%d.weight", index);
            model.backbone.layer1_0[i].bn_weight = ggml_get_tensor(model.ctx, name);
            snprintf(name, sizeof(name), "layer1.0.bn%d.bias", index);
            model.backbone.layer1_0[i].bias = ggml_get_tensor(model.ctx, name);
            snprintf(name, sizeof(name), "layer1.0.bn%d.running_mean", index);
            model.backbone.layer1_0[i].running_mean = ggml_get_tensor(model.ctx, name);
            snprintf(name, sizeof(name), "layer1.0.bn%d.running_var", index);
            model.backbone.layer1_0[i].running_var = ggml_get_tensor(model.ctx, name);
        }
    }
    // layer1.0 contains downsample
    model.backbone.layer1_0[model.backbone.layer1_0.size() - 1].weights = ggml_get_tensor(model.ctx, "layer1.0.downsample.0.weight");
    model.backbone.layer1_0[model.backbone.layer1_0.size() - 1].bn_weight = ggml_get_tensor(model.ctx, "layer1.0.downsample.1.weight");
    model.backbone.layer1_0[model.backbone.layer1_0.size() - 1].bias = ggml_get_tensor(model.ctx, "layer1.0.downsample.1.bias");
    model.backbone.layer1_0[model.backbone.layer1_0.size() - 1].running_mean = ggml_get_tensor(model.ctx, "layer1.0.downsample.1.running_mean");
    model.backbone.layer1_0[model.backbone.layer1_0.size() - 1].running_var = ggml_get_tensor(model.ctx, "layer1.0.downsample.1.running_var");

    // layer1.1
    model.backbone.layer1_1.resize(3);
    model.backbone.layer1_1[1].padding = 1;
    model.backbone.layer1_1[2].activate = true;
    for (int i = 0; i < (int)model.backbone.layer1_1.size(); i++) {
        char name[256];
        int index = i + 1;
        snprintf(name, sizeof(name), "layer1.1.conv%d.weight", index);
        model.backbone.layer1_1[i].weights = ggml_get_tensor(model.ctx, name);
        if (model.backbone.layer1_1[i].batch_normalize) {
            snprintf(name, sizeof(name), "layer1.1.bn%d.weight", index);
            model.backbone.layer1_1[i].bn_weight = ggml_get_tensor(model.ctx, name);
            snprintf(name, sizeof(name), "layer1.1.bn%d.bias", index);
            model.backbone.layer1_1[i].bias = ggml_get_tensor(model.ctx, name);
            snprintf(name, sizeof(name), "layer1.1.bn%d.running_mean", index);
            model.backbone.layer1_1[i].running_mean = ggml_get_tensor(model.ctx, name);
            snprintf(name, sizeof(name), "layer1.1.bn%d.running_var", index);
            model.backbone.layer1_1[i].running_var = ggml_get_tensor(model.ctx, name);
        }
    }

    // layer1.2
    model.backbone.layer1_2.resize(3);
    model.backbone.layer1_2[1].padding = 1;
    model.backbone.layer1_2[2].activate = true;
    for (int i = 0; i < (int)model.backbone.layer1_2.size(); i++) {
        char name[256];
        int index = i + 1;
        snprintf(name, sizeof(name), "layer1.2.conv%d.weight", index);
        model.backbone.layer1_2[i].weights = ggml_get_tensor(model.ctx, name);
        if (model.backbone.layer1_2[i].batch_normalize) {
            snprintf(name, sizeof(name), "layer1.2.bn%d.weight", index);
            model.backbone.layer1_2[i].bn_weight = ggml_get_tensor(model.ctx, name);
            snprintf(name, sizeof(name), "layer1.2.bn%d.bias", index);
            model.backbone.layer1_2[i].bias = ggml_get_tensor(model.ctx, name);
            snprintf(name, sizeof(name), "layer1.2.bn%d.running_mean", index);
            model.backbone.layer1_2[i].running_mean = ggml_get_tensor(model.ctx, name);
            snprintf(name, sizeof(name), "layer1.2.bn%d.running_var", index);
            model.backbone.layer1_2[i].running_var = ggml_get_tensor(model.ctx, name);
        }
    }

    // layer2.0
    model.backbone.layer2_0.resize(4); // 3 conv2d and 1 downsample
    model.backbone.layer2_0[1].stride = 2;
    model.backbone.layer2_0[1].padding = 1;
    model.backbone.layer2_0[2].activate = true;
    model.backbone.layer2_0[3].stride = 2;
    for (int i = 0; i < (int)model.backbone.layer2_0.size() - 1; i++) {
        char name[256];
        int index = i + 1;
        snprintf(name, sizeof(name), "layer2.0.conv%d.weight", index);
        model.backbone.layer2_0[i].weights = ggml_get_tensor(model.ctx, name);
        if (model.backbone.layer2_0[i].batch_normalize) {
            snprintf(name, sizeof(name), "layer2.0.bn%d.weight", index);
            model.backbone.layer2_0[i].bn_weight = ggml_get_tensor(model.ctx, name);
            snprintf(name, sizeof(name), "layer2.0.bn%d.bias", index);
            model.backbone.layer2_0[i].bias = ggml_get_tensor(model.ctx, name);
            snprintf(name, sizeof(name), "layer2.0.bn%d.running_mean", index);
            model.backbone.layer2_0[i].running_mean = ggml_get_tensor(model.ctx, name);
            snprintf(name, sizeof(name), "layer2.0.bn%d.running_var", index);
            model.backbone.layer2_0[i].running_var = ggml_get_tensor(model.ctx, name);
        }
    }
    // layer2.0 contains downsample
    model.backbone.layer2_0[model.backbone.layer2_0.size() - 1].weights = ggml_get_tensor(model.ctx, "layer2.0.downsample.0.weight");
    model.backbone.layer2_0[model.backbone.layer2_0.size() - 1].bn_weight = ggml_get_tensor(model.ctx, "layer2.0.downsample.1.weight");
    model.backbone.layer2_0[model.backbone.layer2_0.size() - 1].bias = ggml_get_tensor(model.ctx, "layer2.0.downsample.1.bias");
    model.backbone.layer2_0[model.backbone.layer2_0.size() - 1].running_mean = ggml_get_tensor(model.ctx, "layer2.0.downsample.1.running_mean");
    model.backbone.layer2_0[model.backbone.layer2_0.size() - 1].running_var = ggml_get_tensor(model.ctx, "layer2.0.downsample.1.running_var");

    // layer2.1
    model.backbone.layer2_1.resize(3);
    model.backbone.layer2_1[1].padding = 1;
    model.backbone.layer2_1[2].activate = true;
    for (int i = 0; i < (int)model.backbone.layer2_1.size(); i++) {
        char name[256];
        int index = i + 1;
        snprintf(name, sizeof(name), "layer2.1.conv%d.weight", index);
        model.backbone.layer2_1[i].weights = ggml_get_tensor(model.ctx, name);
        if (model.backbone.layer2_1[i].batch_normalize) {
            snprintf(name, sizeof(name), "layer2.1.bn%d.weight", index);
            model.backbone.layer2_1[i].bn_weight = ggml_get_tensor(model.ctx, name);
            snprintf(name, sizeof(name), "layer2.1.bn%d.bias", index);
            model.backbone.layer2_1[i].bias = ggml_get_tensor(model.ctx, name);
            snprintf(name, sizeof(name), "layer2.1.bn%d.running_mean", index);
            model.backbone.layer2_1[i].running_mean = ggml_get_tensor(model.ctx, name);
            snprintf(name, sizeof(name), "layer2.1.bn%d.running_var", index);
            model.backbone.layer2_1[i].running_var = ggml_get_tensor(model.ctx, name);
        }
    }

    // layer2.2
    model.backbone.layer2_2.resize(3);
    model.backbone.layer2_2[1].padding = 1;
    model.backbone.layer2_2[2].activate = true;
    for (int i = 0; i < (int)model.backbone.layer2_2.size(); i++) {
        char name[256];
        int index = i + 1;
        snprintf(name, sizeof(name), "layer2.2.conv%d.weight", index);
        model.backbone.layer2_2[i].weights = ggml_get_tensor(model.ctx, name);
        if (model.backbone.layer2_2[i].batch_normalize) {
            snprintf(name, sizeof(name), "layer2.2.bn%d.weight", index);
            model.backbone.layer2_2[i].bn_weight = ggml_get_tensor(model.ctx, name);
            snprintf(name, sizeof(name), "layer2.2.bn%d.bias", index);
            model.backbone.layer2_2[i].bias = ggml_get_tensor(model.ctx, name);
            snprintf(name, sizeof(name), "layer2.2.bn%d.running_mean", index);
            model.backbone.layer2_2[i].running_mean = ggml_get_tensor(model.ctx, name);
            snprintf(name, sizeof(name), "layer2.2.bn%d.running_var", index);
            model.backbone.layer2_2[i].running_var = ggml_get_tensor(model.ctx, name);
        }
    }

    // layer2.3
    model.backbone.layer2_3.resize(3);
    model.backbone.layer2_3[1].padding = 1;
    model.backbone.layer2_3[2].activate = true;
    for (int i = 0; i < (int)model.backbone.layer2_3.size(); i++) {
        char name[256];
        int index = i + 1;
        snprintf(name, sizeof(name), "layer2.3.conv%d.weight", index);
        model.backbone.layer2_3[i].weights = ggml_get_tensor(model.ctx, name);
        if (model.backbone.layer2_3[i].batch_normalize) {
            snprintf(name, sizeof(name), "layer2.3.bn%d.weight", index);
            model.backbone.layer2_3[i].bn_weight = ggml_get_tensor(model.ctx, name);
            snprintf(name, sizeof(name), "layer2.3.bn%d.bias", index);
            model.backbone.layer2_3[i].bias = ggml_get_tensor(model.ctx, name);
            snprintf(name, sizeof(name), "layer2.3.bn%d.running_mean", index);
            model.backbone.layer2_3[i].running_mean = ggml_get_tensor(model.ctx, name);
            snprintf(name, sizeof(name), "layer2.3.bn%d.running_var", index);
            model.backbone.layer2_3[i].running_var = ggml_get_tensor(model.ctx, name);
        }
    }

    // layer3.0
    model.backbone.layer3_0.resize(4); // 3 conv2d and 1 downsample
    model.backbone.layer3_0[1].stride = 2;
    model.backbone.layer3_0[1].padding = 1;
    model.backbone.layer3_0[2].activate = true;
    model.backbone.layer3_0[3].stride = 2;
    for (int i = 0; i < (int)model.backbone.layer3_0.size() - 1; i++) {
        char name[256];
        int index = i + 1;
        snprintf(name, sizeof(name), "layer3.0.conv%d.weight", index);
        model.backbone.layer3_0[i].weights = ggml_get_tensor(model.ctx, name);
        if (model.backbone.layer3_0[i].batch_normalize) {
            snprintf(name, sizeof(name), "layer3.0.bn%d.weight", index);
            model.backbone.layer3_0[i].bn_weight = ggml_get_tensor(model.ctx, name);
            snprintf(name, sizeof(name), "layer3.0.bn%d.bias", index);
            model.backbone.layer3_0[i].bias = ggml_get_tensor(model.ctx, name);
            snprintf(name, sizeof(name), "layer3.0.bn%d.running_mean", index);
            model.backbone.layer3_0[i].running_mean = ggml_get_tensor(model.ctx, name);
            snprintf(name, sizeof(name), "layer3.0.bn%d.running_var", index);
            model.backbone.layer3_0[i].running_var = ggml_get_tensor(model.ctx, name);
        }
    }
    // layer3.0 contains downsample
    model.backbone.layer3_0[model.backbone.layer3_0.size() - 1].weights = ggml_get_tensor(model.ctx, "layer3.0.downsample.0.weight");
    model.backbone.layer3_0[model.backbone.layer3_0.size() - 1].bn_weight = ggml_get_tensor(model.ctx, "layer3.0.downsample.1.weight");
    model.backbone.layer3_0[model.backbone.layer3_0.size() - 1].bias = ggml_get_tensor(model.ctx, "layer3.0.downsample.1.bias");
    model.backbone.layer3_0[model.backbone.layer3_0.size() - 1].running_mean = ggml_get_tensor(model.ctx, "layer3.0.downsample.1.running_mean");
    model.backbone.layer3_0[model.backbone.layer3_0.size() - 1].running_var = ggml_get_tensor(model.ctx, "layer3.0.downsample.1.running_var");

    // layer3.1
    model.backbone.layer3_1.resize(3);
    model.backbone.layer3_1[1].padding = 1;
    model.backbone.layer3_1[2].activate = true;
    for (int i = 0; i < (int)model.backbone.layer3_1.size(); i++) {
        char name[256];
        int index = i + 1;
        snprintf(name, sizeof(name), "layer3.1.conv%d.weight", index);
        model.backbone.layer3_1[i].weights = ggml_get_tensor(model.ctx, name);
        if (model.backbone.layer3_1[i].batch_normalize) {
            snprintf(name, sizeof(name), "layer3.1.bn%d.weight", index);
            model.backbone.layer3_1[i].bn_weight = ggml_get_tensor(model.ctx, name);
            snprintf(name, sizeof(name), "layer3.1.bn%d.bias", index);
            model.backbone.layer3_1[i].bias = ggml_get_tensor(model.ctx, name);
            snprintf(name, sizeof(name), "layer3.1.bn%d.running_mean", index);
            model.backbone.layer3_1[i].running_mean = ggml_get_tensor(model.ctx, name);
            snprintf(name, sizeof(name), "layer3.1.bn%d.running_var", index);
            model.backbone.layer3_1[i].running_var = ggml_get_tensor(model.ctx, name);
        }
    }

    // layer3.2
    model.backbone.layer3_2.resize(3);
    model.backbone.layer3_2[1].padding = 1;
    model.backbone.layer3_2[2].activate = true;
    for (int i = 0; i < (int)model.backbone.layer3_2.size(); i++) {
        char name[256];
        int index = i + 1;
        snprintf(name, sizeof(name), "layer3.2.conv%d.weight", index);
        model.backbone.layer3_2[i].weights = ggml_get_tensor(model.ctx, name);
        if (model.backbone.layer3_2[i].batch_normalize) {
            snprintf(name, sizeof(name), "layer3.2.bn%d.weight", index);
            model.backbone.layer3_2[i].bn_weight = ggml_get_tensor(model.ctx, name);
            snprintf(name, sizeof(name), "layer3.2.bn%d.bias", index);
            model.backbone.layer3_2[i].bias = ggml_get_tensor(model.ctx, name);
            snprintf(name, sizeof(name), "layer3.2.bn%d.running_mean", index);
            model.backbone.layer3_2[i].running_mean = ggml_get_tensor(model.ctx, name);
            snprintf(name, sizeof(name), "layer3.2.bn%d.running_var", index);
            model.backbone.layer3_2[i].running_var = ggml_get_tensor(model.ctx, name);
        }
    }

    // layer3.3
    model.backbone.layer3_3.resize(3);
    model.backbone.layer3_3[1].padding = 1;
    model.backbone.layer3_3[2].activate = true;
    for (int i = 0; i < (int)model.backbone.layer3_3.size(); i++) {
        char name[256];
        int index = i + 1;
        snprintf(name, sizeof(name), "layer3.3.conv%d.weight", index);
        model.backbone.layer3_3[i].weights = ggml_get_tensor(model.ctx, name);
        if (model.backbone.layer3_3[i].batch_normalize) {
            snprintf(name, sizeof(name), "layer3.3.bn%d.weight", index);
            model.backbone.layer3_3[i].bn_weight = ggml_get_tensor(model.ctx, name);
            snprintf(name, sizeof(name), "layer3.3.bn%d.bias", index);
            model.backbone.layer3_3[i].bias = ggml_get_tensor(model.ctx, name);
            snprintf(name, sizeof(name), "layer3.3.bn%d.running_mean", index);
            model.backbone.layer3_3[i].running_mean = ggml_get_tensor(model.ctx, name);
            snprintf(name, sizeof(name), "layer3.3.bn%d.running_var", index);
            model.backbone.layer3_3[i].running_var = ggml_get_tensor(model.ctx, name);
        }
    }
    model.backbone.layer3_4.resize(3);
    model.backbone.layer3_4[1].padding = 1;
    model.backbone.layer3_4[2].activate = true;
    for (int i = 0; i < (int)model.backbone.layer3_4.size(); i++) {
        char name[256];
        int index = i + 1;
        snprintf(name, sizeof(name), "layer3.4.conv%d.weight", index);
        model.backbone.layer3_4[i].weights = ggml_get_tensor(model.ctx, name);
        if (model.backbone.layer3_4[i].batch_normalize) {
            snprintf(name, sizeof(name), "layer3.4.bn%d.weight", index);
            model.backbone.layer3_4[i].bn_weight = ggml_get_tensor(model.ctx, name);
            snprintf(name, sizeof(name), "layer3.4.bn%d.bias", index);
            model.backbone.layer3_4[i].bias = ggml_get_tensor(model.ctx, name);
            snprintf(name, sizeof(name), "layer3.4.bn%d.running_mean", index);
            model.backbone.layer3_4[i].running_mean = ggml_get_tensor(model.ctx, name);
            snprintf(name, sizeof(name), "layer3.4.bn%d.running_var", index);
            model.backbone.layer3_4[i].running_var = ggml_get_tensor(model.ctx, name);
        }
    }

    // layer4.0
    model.backbone.layer4_0.resize(4); // 3 conv2d and 1 downsample
    model.backbone.layer4_0[1].stride = 2;
    model.backbone.layer4_0[1].padding = 1;
    model.backbone.layer4_0[2].activate = true;
    model.backbone.layer4_0[3].stride = 2;
    for (int i = 0; i < (int)model.backbone.layer4_0.size() - 1; i++) {
        char name[256];
        int index = i + 1;
        snprintf(name, sizeof(name), "layer4.0.conv%d.weight", index);
        model.backbone.layer4_0[i].weights = ggml_get_tensor(model.ctx, name);
        if (model.backbone.layer4_0[i].batch_normalize) {
            snprintf(name, sizeof(name), "layer4.0.bn%d.weight", index);
            model.backbone.layer4_0[i].bn_weight = ggml_get_tensor(model.ctx, name);
            snprintf(name, sizeof(name), "layer4.0.bn%d.bias", index);
            model.backbone.layer4_0[i].bias = ggml_get_tensor(model.ctx, name);
            snprintf(name, sizeof(name), "layer4.0.bn%d.running_mean", index);
            model.backbone.layer4_0[i].running_mean = ggml_get_tensor(model.ctx, name);
            snprintf(name, sizeof(name), "layer4.0.bn%d.running_var", index);
            model.backbone.layer4_0[i].running_var = ggml_get_tensor(model.ctx, name);
        }
    }
    // layer4.0 contains downsample
    model.backbone.layer4_0[model.backbone.layer4_0.size() - 1].weights = ggml_get_tensor(model.ctx, "layer4.0.downsample.0.weight");
    model.backbone.layer4_0[model.backbone.layer4_0.size() - 1].bn_weight = ggml_get_tensor(model.ctx, "layer4.0.downsample.1.weight");
    model.backbone.layer4_0[model.backbone.layer4_0.size() - 1].bias = ggml_get_tensor(model.ctx, "layer4.0.downsample.1.bias");
    model.backbone.layer4_0[model.backbone.layer4_0.size() - 1].running_mean = ggml_get_tensor(model.ctx, "layer4.0.downsample.1.running_mean");
    model.backbone.layer4_0[model.backbone.layer4_0.size() - 1].running_var = ggml_get_tensor(model.ctx, "layer4.0.downsample.1.running_var");

    // layer4.1
    model.backbone.layer4_1.resize(3);
    model.backbone.layer4_1[1].padding = 1;
    model.backbone.layer4_1[2].activate = true;
    for (int i = 0; i < (int)model.backbone.layer4_1.size(); i++) {
        char name[256];
        int index = i + 1;
        snprintf(name, sizeof(name), "layer4.1.conv%d.weight", index);
        model.backbone.layer4_1[i].weights = ggml_get_tensor(model.ctx, name);
        if (model.backbone.layer4_1[i].batch_normalize) {
            snprintf(name, sizeof(name), "layer4.1.bn%d.weight", index);
            model.backbone.layer4_1[i].bn_weight = ggml_get_tensor(model.ctx, name);
            snprintf(name, sizeof(name), "layer4.1.bn%d.bias", index);
            model.backbone.layer4_1[i].bias = ggml_get_tensor(model.ctx, name);
            snprintf(name, sizeof(name), "layer4.1.bn%d.running_mean", index);
            model.backbone.layer4_1[i].running_mean = ggml_get_tensor(model.ctx, name);
            snprintf(name, sizeof(name), "layer4.1.bn%d.running_var", index);
            model.backbone.layer4_1[i].running_var = ggml_get_tensor(model.ctx, name);
        }
    }

    // layer4.2
    model.backbone.layer4_2.resize(3);
    model.backbone.layer4_2[1].padding = 1;
    model.backbone.layer4_2[2].activate = true;
    for (int i = 0; i < (int)model.backbone.layer4_2.size(); i++) {
        char name[256];
        int index = i + 1;
        snprintf(name, sizeof(name), "layer4.2.conv%d.weight", index);
        model.backbone.layer4_2[i].weights = ggml_get_tensor(model.ctx, name);
        if (model.backbone.layer4_2[i].batch_normalize) {
            snprintf(name, sizeof(name), "layer4.2.bn%d.weight", index);
            model.backbone.layer4_2[i].bn_weight = ggml_get_tensor(model.ctx, name);
            snprintf(name, sizeof(name), "layer4.2.bn%d.bias", index);
            model.backbone.layer4_2[i].bias = ggml_get_tensor(model.ctx, name);
            snprintf(name, sizeof(name), "layer4.2.bn%d.running_mean", index);
            model.backbone.layer4_2[i].running_mean = ggml_get_tensor(model.ctx, name);
            snprintf(name, sizeof(name), "layer4.2.bn%d.running_var", index);
            model.backbone.layer4_2[i].running_var = ggml_get_tensor(model.ctx, name);
        }
    }
    

    return true;
}

int main(int argc, char ** argv){
    ggml_backend_load_all(); // important init
    ggml_time_init(); // important init
    const int64_t t_main_start_us = ggml_time_us();

    icls_model model;

    icls_params params; // temporary be default value
    // init backend
    model.backend = create_backend(params);
    if (!model.backend) {
        std::cout << "failed to create backend." << std::endl;
        return 1;
    }

    // load model
    icls_model_init_from_file(params.model, model);
}
