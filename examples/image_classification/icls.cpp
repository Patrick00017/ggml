#include "icls.h"

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
    
    // model.width  = 416;
    // model.height = 416;
    model.backbone.layers.resize(13);
    // model.conv2d_layers[7].padding = 0;
    // model.conv2d_layers[9].padding = 0;
    // model.conv2d_layers[9].batch_normalize = false;
    // model.conv2d_layers[9].activate = false;
    // model.conv2d_layers[10].padding = 0;
    // model.conv2d_layers[12].padding = 0;
    // model.conv2d_layers[12].batch_normalize = false;
    // model.conv2d_layers[12].activate = false;
    // for (int i = 0; i < (int)model.conv2d_layers.size(); i++) {
    //     char name[256];
    //     snprintf(name, sizeof(name), "l%d_weights", i);
    //     model.conv2d_layers[i].weights = ggml_get_tensor(model.ctx, name);
    //     snprintf(name, sizeof(name), "l%d_biases", i);
    //     model.conv2d_layers[i].biases = ggml_get_tensor(model.ctx, name);
    //     if (model.conv2d_layers[i].batch_normalize) {
    //         snprintf(name, sizeof(name), "l%d_scales", i);
    //         model.conv2d_layers[i].scales = ggml_get_tensor(model.ctx, name);
    //         snprintf(name, sizeof(name), "l%d_rolling_mean", i);
    //         model.conv2d_layers[i].rolling_mean = ggml_get_tensor(model.ctx, name);
    //         snprintf(name, sizeof(name), "l%d_rolling_variance", i);
    //         model.conv2d_layers[i].rolling_variance = ggml_get_tensor(model.ctx, name);
    //     }
    // }
    return true;
}

int main(int argc, char ** argv){
    ggml_time_init();
    const int64_t t_main_start_us = ggml_time_us();
}