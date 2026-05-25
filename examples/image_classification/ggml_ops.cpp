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

int get_tensor_dim_num(ggml_tensor* t){
    int dim_count = 0;
    for(int i=0;i<GGML_MAX_DIMS;i++){
        if(t->nb[i] > 0)
            dim_count++;
    }
    return dim_count;
}

ggml_tensor* aten_convolution_default(
    ggml_context* ctx,
    ggml_tensor* input, 
    ggml_tensor* weight, 
    ggml_tensor* bias,
    std::vector<int> stride,
    std::vector<int> padding,
    std::vector<int> dilation,
    bool is_transposed,
    std::vector<int> output_padding,
    int groups
){
    int dim_num = get_tensor_dim_num(weight);
    switch(dim_num){
        case 2: 
            return ggml_conv_2d(ctx, input, weight, stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1]);
        // leave here because ggml max dim is 4, maybe update
    }
    // default use conv2d
    return ggml_conv_2d(ctx, input, weight, stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1]);
}

ggml_tensor* aten_add_tensor(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b){
    return ggml_add(ctx, a, b);
}

std::vector<ggml_tensor*> aten__native_batch_norm_legit_functional_default(
    ggml_context* ctx,
    ggml_tensor* input,
    ggml_tensor* weight,
    ggml_tensor* bias,
    ggml_tensor* running_mean,
    ggml_tensor* running_var,
    bool training,
    float momentum,
    float eps
){
    // adjustify the shape free, no more memory will be allocated
    ggml_tensor* _running_mean = ggml_reshape_4d(ctx, running_mean, 1, 1, running_mean->ne[0], 1);
    ggml_tensor* _running_var = ggml_reshape_4d(ctx, running_var, 1, 1, running_var->ne[0], 1);
    ggml_tensor* _weight = ggml_reshape_4d(ctx, weight, 1, 1, weight->ne[0], 1);
    ggml_tensor* result = ggml_sub(ctx, input, ggml_repeat(ctx, _running_mean, input));
    result = ggml_div(ctx, result, ggml_sqrt(ctx, ggml_repeat(ctx, _running_var, result)));
    result = ggml_mul(ctx, result, ggml_repeat(ctx, _weight, result));
    result = ggml_add(ctx, result, bias);
    // actual return will be (output, save_mean, save_rstd, new_running_mean, new_running_var)
    return {result, result, result, result, result};
}

ggml_tensor* get_item(ggml_context* ctx, std::vector<ggml_tensor*> items, int index){
    return items[index];
}

ggml_tensor* aten_relu_default(ggml_context* ctx, ggml_tensor* input){
    return ggml_relu(ctx, input);
}

ggml_tensor* aten_max_pool2d_with_indices_default(
    ggml_context* ctx,
    ggml_tensor* input,
    std::vector<int> kernel_size,
    std::vector<int> stride,
    std::vector<int> padding,
    std::vector<int> dilation,
    bool ceil_mode
){
    ggml_tensor* result = ggml_pool_2d(ctx, input, GGML_OP_POOL_MAX, kernel_size[0], kernel_size[1], stride[0], stride[1], padding[0], padding[1]);
}

ggml_tensor* aten_mean_dim(
    ggml_context* ctx,
    ggml_tensor* input,
    std::vector<int> dims,
    bool keep_dims
){
    
}

ggml_tensor* aten_view_default(ggml_context* ctx, ggml_tensor* input, std::vector<int> shape){
    int dim_num = shape.size();
    switch(dim_num){
        case 1:
            return ggml_reshape_1d(ctx, input, shape[0]);
        case 2:
            return ggml_reshape_2d(ctx, input, shape[0], shape[1]);
        case 3:
            return ggml_reshape_3d(ctx, input, shape[0], shape[1], shape[2]);
        case 4:
            return ggml_reshape_4d(ctx, input, shape[0], shape[1], shape[2], shape[3]);
    }
    // if dim num is wrong, do nothing
    return input;
}

ggml_tensor* aten_permute_default(ggml_context* ctx, ggml_tensor* input, std::vector<int> dims){

}

ggml_tensor* aten_addmm_default(
    ggml_context* ctx,
    ggml_tensor* input,
    ggml_tensor* mat1,
    ggml_tensor* mat2,
    int beta = 1,
    int alpha = 1
){

}


