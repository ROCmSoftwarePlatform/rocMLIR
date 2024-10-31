// RUN: rocmlir-opt -split-input-file --migraphx-to-tosa %s | FileCheck %s

// CHECK-LABEL: @migraphx_literal_zero()
// CHECK-SAME: -> tensor<9408xi8> {
func.func @migraphx_literal_zero() -> !migraphx.shaped<64x3x7x7xsi8, 147x49x7x1> {
  // CHECK: %[[const:.+]] = "tosa.const"() <{value = dense<0> : tensor<64x3x7x7xi8>}> : () -> tensor<64x3x7x7xi8>
  // CHECK-NEXT: %[[reshape:.+]] = tosa.reshape %[[const]] {new_shape = array<i64: 9408>} : (tensor<64x3x7x7xi8>) -> tensor<9408xi8>
  // CHECK-NEXT: return %[[reshape]] : tensor<9408xi8>
  %0 = migraphx.literal (dense<0> : tensor<64x1xsi8>) : <64x3x7x7xsi8, 147x49x7x1>
  return %0 : !migraphx.shaped<64x3x7x7xsi8, 147x49x7x1>
}

// CHECK-LABEL: @migraphx_literal_negative()
// CHECK-SAME: -> tensor<9408xi8> {
func.func @migraphx_literal_negative() -> !migraphx.shaped<64x3x7x7xsi8, 147x49x7x1> {
  // CHECK: %[[const:.+]] = "tosa.const"() <{value = dense<-1> : tensor<64x3x7x7xi8>}> : () -> tensor<64x3x7x7xi8>
  // CHECK-NEXT: %[[reshape:.+]] = tosa.reshape %[[const]] {new_shape = array<i64: 9408>} : (tensor<64x3x7x7xi8>) -> tensor<9408xi8>
  // CHECK-NEXT: return %[[reshape]] : tensor<9408xi8>
  %0 = migraphx.literal (dense<-1> : tensor<64x1xsi8>) : <64x3x7x7xsi8, 147x49x7x1>
  return %0 : !migraphx.shaped<64x3x7x7xsi8, 147x49x7x1>
}

// CHECK-LABEL: @migraphx_convert_int4_signed
// CHECK: tosa.cast
// CHECK-SAME: (tensor<16xi4>) -> tensor<16xi8>
func.func @migraphx_convert_int4_signed(%arg0: !migraphx.shaped<16xsi4, 1>) -> !migraphx.shaped<16xi8, 1> {
  %0 = migraphx.convert %arg0 : <16xsi4, 1> to <16xi8, 1>
  return %0 : !migraphx.shaped<16xi8, 1>
}

// CHECK-LABEL: @migraphx_convert_int4_unsigned
// CHECK: tosa.custom
// CHECK-SAME: {domain_name = "rocmlir", implementation_attrs = "", operator_name = "unsigned_cast"} : (tensor<16xi4>) -> tensor<16xi8>
func.func @migraphx_convert_int4_unsigned(%arg0: !migraphx.shaped<16xui4, 1>) -> !migraphx.shaped<16xi8, 1> {
  %0 = migraphx.convert %arg0 : <16xui4, 1> to <16xi8, 1>
  return %0 : !migraphx.shaped<16xi8, 1>
}

// CHECK-LABEL: @migraphx_convert_int4_unsigned_to_float
// CHECK: tosa.custom
// CHECK-SAME: {domain_name = "rocmlir", implementation_attrs = "", operator_name = "unsigned_cast"} : (tensor<16xi4>) -> tensor<16xf32>
func.func @migraphx_convert_int4_unsigned_to_float(%arg0: !migraphx.shaped<16xui4, 1>) -> !migraphx.shaped<16xf32, 1> {
  %0 = migraphx.convert %arg0 : <16xui4, 1> to <16xf32, 1>
  return %0 : !migraphx.shaped<16xf32, 1>
}

// CHECK-LABEL: @migraphx_div_si32
// CHECK: tosa.int_div
// CHECK-SAME: (tensor<1x36x384x64xi32>, tensor<1x36x384x64xi32>) -> tensor<1x36x384x64xi32>
func.func @migraphx_div_si32(%arg0: !migraphx.shaped<1x36x384x64xsi32, 884736x24576x64x1>, %arg1: !migraphx.shaped<1x36x384x64xsi32, 884736x24576x64x1>) -> !migraphx.shaped<1x36x384x64xsi32, 884736x24576x64x1> attributes{kernel, arch = ""} {
  %0 = migraphx.div %arg0, %arg1 : <1x36x384x64xsi32, 884736x24576x64x1>, <1x36x384x64xsi32, 884736x24576x64x1> -> <1x36x384x64xsi32, 884736x24576x64x1>
  return %0 : !migraphx.shaped<1x36x384x64xsi32, 884736x24576x64x1>
}

// CHECK-LABEL: @migraphx_div_ui32
// CHECK: tosa.custom
// CHECK-SAME: {domain_name = "rocmlir", implementation_attrs = "", operator_name = "unsigned_div"} : (tensor<1x36x384x64xi32>, tensor<1x36x384x64xi32>) -> tensor<1x36x384x64xi32>
func.func @migraphx_div_ui32(%arg0: !migraphx.shaped<1x36x384x64xui32, 884736x24576x64x1>, %arg1: !migraphx.shaped<1x36x384x64xui32, 884736x24576x64x1>) -> !migraphx.shaped<1x36x384x64xui32, 884736x24576x64x1> attributes{kernel, arch = ""} {
  %0 = migraphx.div %arg0, %arg1 : <1x36x384x64xui32, 884736x24576x64x1>, <1x36x384x64xui32, 884736x24576x64x1> -> <1x36x384x64xui32, 884736x24576x64x1>
  return %0 : !migraphx.shaped<1x36x384x64xui32, 884736x24576x64x1>
}

// CHECK-LABEL: func @dequantize_scale_bias_ui32
// CHECK: tosa.custom %{{.*}} {domain_name = "rocmlir", implementation_attrs = "", operator_name = "unsigned_cast"} : (tensor<1x112x112x64xi32>) -> tensor<1x112x112x64xf32>
// CHECK: tosa.custom %{{.*}} {domain_name = "rocmlir", implementation_attrs = "", operator_name = "unsigned_cast"} : (tensor<64xi32>) -> tensor<64xf32>
// CHECK: tosa.sub
// CHECK: tosa.mul
func.func @dequantize_scale_bias_ui32(%arg: !migraphx.shaped<1x112x112x64xui32, 802816x7168x64x1>, %scale: !migraphx.shaped<64xf32, 1>, %bias: !migraphx.shaped<64xui32, 1>) -> !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1> attributes {kernel = "mixr"} {
  %1 = migraphx.dequantizelinear %arg, %scale, %bias : <1x112x112x64xui32, 802816x7168x64x1>, <64xf32, 1>, !migraphx.shaped<64xui32, 1> -> <1x112x112x64xf32, 802816x7168x64x1>
  return %1 : !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1>
}

// CHECK-LABEL: func @dequantize_scale_bias_si32
// CHECK: tosa.cast{{.*}}f32
// CHECK: tosa.cast{{.*}}f32
// CHECK: tosa.sub
// CHECK: tosa.mul
func.func @dequantize_scale_bias_si32(%arg: !migraphx.shaped<1x112x112x64xsi32, 802816x7168x64x1>, %scale: !migraphx.shaped<64xf32, 1>, %bias: !migraphx.shaped<64xsi32, 1>) -> !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1> attributes {kernel = "mixr"} {
  %1 = migraphx.dequantizelinear %arg, %scale, %bias : <1x112x112x64xsi32, 802816x7168x64x1>, <64xf32, 1>, !migraphx.shaped<64xsi32, 1> -> <1x112x112x64xf32, 802816x7168x64x1>
  return %1 : !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1>
}