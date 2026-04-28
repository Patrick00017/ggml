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
    
    // TODO: load model.ctx tensors into backbone
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
