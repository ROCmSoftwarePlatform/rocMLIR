// RUN: rocmlir-gen -fut mlir_convolution_sigmoid_mul --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx | rocmlir-driver -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_convolution_sigmoid_mul_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2
// CHECK: [1 1 1]
module {
  func.func @mlir_convolution_sigmoid_mul(%arg0: !migraphx.shaped<4x56x122x122xf32, 833504x14884x122x1>, %arg1: !migraphx.shaped<4x14x1x1xf32, 14x1x1x1>, %arg2: !migraphx.shaped<56x14x1x1xf32, 14x1x1x1>) -> !migraphx.shaped<4x56x122x122xf32, 833504x14884x122x1> attributes {arch = "gfx942:sramecc+:xnack-", kernel = "mixr", num_cu = 304 : i64} {
    %0 = migraphx.convolution %arg1, %arg2 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : <4x14x1x1xf32, 14x1x1x1>, <56x14x1x1xf32, 14x1x1x1> -> <4x56x1x1xf32, 56x1x1x1>
    %1 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [4, 56, 122, 122]} : <4x56x1x1xf32, 56x1x1x1> -> <4x56x122x122xf32, 56x1x0x0>
    %2 = migraphx.sigmoid %1 : <4x56x122x122xf32, 56x1x0x0> -> <4x56x122x122xf32, 833504x14884x122x1>
    %3 = migraphx.mul %2, %arg0 : <4x56x122x122xf32, 833504x14884x122x1>, <4x56x122x122xf32, 833504x14884x122x1> -> <4x56x122x122xf32, 833504x14884x122x1>
    return %3 : !migraphx.shaped<4x56x122x122xf32, 833504x14884x122x1>
  }
}
