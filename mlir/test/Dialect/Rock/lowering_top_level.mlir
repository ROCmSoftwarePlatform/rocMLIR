// This tests checks the following aspects of lowering component:
// * Can pass arguments correctly
// * Can pass arguments in the right sequence
// * Have, in most cases, the correct transformations
// * Have one gridwise_gemm
// * Can support F32 and F16

// RUN: rocmlir-opt -rock-conv-to-gemm %s | FileCheck %s

// CHECK-DAG: #[[$MAP_FILTER_FWD:transform_map[0-9]*]] = #rock.transform_map<{{.*}} bounds = [1, 72, 128] -> [1, 128, 8, 3, 3]>
// CHECK-DAG: #[[$MAP_INPUT1_FWD:transform_map[0-9]*]] = #rock.transform_map<{{.*}} bounds = [128, 1, 8, 32, 32] -> [128, 1, 8, 32, 32]>
// CHECK-DAG: #[[$MAP_INPUT2_FWD:transform_map[0-9]*]] = #rock.transform_map<{{.*}} bounds = [128, 1, 8, 3, 30, 3, 30] -> [128, 1, 8, 32, 32]>
// CHECK-DAG: #[[$MAP_INPUT3_FWD:transform_map[0-9]*]] = #rock.transform_map<{{.*}} bounds = [1, 72, 115200] -> [128, 1, 8, 3, 30, 3, 30]>
// CHECK-DAG: #[[$MAP_OUTPUT_FWD:transform_map[0-9]*]] = #rock.transform_map<{{.*}} bounds = [1, 128, 115200] -> [128, 1, 128, 30, 30]>

// CHECK-DAG: #[[$MAP_BWD_DATA_FIL1_NO_PAD:transform_map[0-9]*]] = {{.*}}by [<PassThrough ["g", "k", "c"] at [0, 1, 2] -> ["g", "k", "c"] at [0, 1, 2]>, <Embed{1, 1} ["0dot", "0tilda"] at [3, 4] -> ["0"] at [3]>, <Embed{1, 1} ["1dot", "1tilda"] at [5, 6] -> ["1"] at [4]>]
// CHECK-DAG: #[[$MAP_BWD_DATA_FIL2_NO_PAD:transform_map[0-9]*]] = {{.*}}by [<PassThrough ["g", "k", "c"] at [0, 1, 2] -> ["g", "k", "c"] at [0, 1, 2]>, <Slice{0, 1, 0, 1} ["0dotslice", "1dotslice"] at [3, 5] -> ["0dot", "1dot"] at [3, 5]>, <Slice{0, 1, 0, 1} ["0tildaslice", "1tildaslice"] at [4, 6] -> ["0tilda", "1tilda"] at [4, 6]>]
// CHECK-DAG: #[[$MAP_BWD_DATA_FIL3_NO_PAD:transform_map[0-9]*]] = {{.*}}by [<PassThrough ["gemmG"] at [0] -> ["g"] at [0]>, <Merge{1024, 1, 1} ["gemmK"] at [1] -> ["k", "0dotslice", "1dotslice"] at [1, 3, 5]>, <Merge{1024, 1, 1} ["gemmM"] at [2] -> ["c", "0tildaslice", "1tildaslice"] at [2, 4, 6]>]

// CHECK-DAG: #[[$MAP_BWD_DATA_IN1_NO_PAD:transform_map[0-9]*]] = {{.*}}by [<PassThrough ["gi", "ni", "ci"] at [1, 0, 2] -> ["gi", "ni", "ci"] at [1, 0, 2]>, <Pad{0, 0, 0, 0} ["0ipad", "1ipad"] at [3, 4] -> ["0i", "1i"] at [3, 4]>
// CHECK-DAG: #[[$MAP_BWD_DATA_IN2_NO_PAD:transform_map[0-9]*]] = {{.*}}by [<PassThrough ["gi", "ni", "ci"] at [1, 0, 2] -> ["gi", "ni", "ci"] at [1, 0, 2]>, <Embed{1, 1} ["0ftilda", "0itilda"] at [3, 4] -> ["0ipad"] at [3]>, <Embed{1, 1} ["1ftilda", "1itilda"] at [5, 6] -> ["1ipad"] at [4]>]
// CHECK-DAG: #[[$MAP_BWD_DATA_IN3_NO_PAD:transform_map[0-9]*]] = {{.*}}by [<PassThrough ["gi", "ni", "ci"] at [1, 0, 2] -> ["gi", "ni", "ci"] at [1, 0, 2]>, <Slice{0, 1, 0, 1} ["0slice", "1slice"] at [3, 5] -> ["0ftilda", "1ftilda"] at [3, 5]>, <Slice{0, 14, 0, 14} ["0islice", "1islice"] at [4, 6] -> ["0itilda", "1itilda"] at [4, 6]>]
// CHECK-DAG: #[[$MAP_BWD_DATA_IN4_NO_PAD:transform_map[0-9]*]] = {{.*}}by [<PassThrough ["gemmG"] at [0] -> ["gi"] at [1]>, <Merge{1024, 1, 1} ["gemmM"] at [1] -> ["ci", "0slice", "1slice"] at [2, 3, 5]>, <Merge{128, 14, 14} ["gemmN"] at [2] -> ["ni", "0islice", "1islice"] at [0, 4, 6]>]
// CHECK-DAG: #[[$MAP_BWD_DATA_OUT1_NO_PAD:transform_map[0-9]*]] = {{.*}}by [<PassThrough ["go", "no", "ko"] at [1, 0, 2] -> ["go", "no", "ko"] at [1, 0, 2]>, <Embed{-1, 1} ["0dot", "0tilda"] at [3, 4] -> ["0o"] at [3]>, <Embed{-1, 1} ["1dot", "1tilda"] at [5, 6] -> ["1o"] at [4]>]
// CHECK-DAG: #[[$MAP_BWD_DATA_OUT2_NO_PAD:transform_map[0-9]*]] = {{.*}}by [<PassThrough ["go", "no", "ko"] at [1, 0, 2] -> ["go", "no", "ko"] at [1, 0, 2]>, <Slice{0, 1, 0, 1} ["0slice", "1slice"] at [3, 5] -> ["0dot", "1dot"] at [3, 5]>, <Slice{0, 14, 0, 14} ["0islice", "1islice"] at [4, 6] -> ["0tilda", "1tilda"] at [4, 6]>]
// CHECK-DAG: #[[$MAP_BWD_DATA_OUT3_NO_PAD:transform_map[0-9]*]] = {{.*}}by [<PassThrough ["gemmG"] at [0] -> ["go"] at [1]>, <Merge{1024, 1, 1} ["gemmK"] at [1] -> ["ko", "0slice", "1slice"] at [2, 3, 5]>, <Merge{128, 14, 14} ["gemmN"] at [2] -> ["no", "0islice", "1islice"] at [0, 4, 6]>]

// CHECK-DAG: #[[$MAP_BWD_WEIGHT_FIL1:transform_map[0-9]*]] = {{.*}}by [<PassThrough ["gemmG"] at [0] -> ["g"] at [0]>, <PassThrough ["gemmM"] at [1] -> ["k"] at [1]>, <Merge{8, 3, 3} ["gemmN"] at [2] -> ["c", "0", "1"] at [2, 3, 4]>]
// CHECK-DAG: #[[$MAP_BWD_WEIGHT_IN3:transform_map[0-9]*]] = {{.*}}by [<PassThrough ["gemmG"] at [0] -> ["gi"] at [1]>, <Merge{128, 30, 30} ["gemmK"] at [1] -> ["ni", "0o", "1o"] at [0, 4, 6]>, <Merge{8, 3, 3} ["gemmN"] at [2] -> ["ci", "0", "1"] at [2, 3, 5]>]
// CHECK-DAG: #[[$MAP_BWD_WEIGHT_OUT:transform_map[0-9]*]] = {{.*}}by [<PassThrough ["gemmG"] at [0] -> ["go"] at [1]>, <Merge{128, 30, 30} ["gemmK"] at [1] -> ["no", "0o", "1o"] at [0, 3, 4]>, <PassThrough ["gemmM"] at [2] -> ["ko"] at [2]>]

#general_gemm_params0 = #rock.general_gemm_params<blockSize = 64, kPerBlock = 8, mPerBlock = 128, nPerBlock = 128, kPerThread = 1, mPerThread = 4, nPerThread = 4, kpack = 1, splitKFactor = 1>
#general_gemm_params1 = #rock.general_gemm_params<blockSize = 64, kPerBlock = 16, mPerBlock = 64, nPerBlock = 64, kPerThread = 1, mPerThread = 4, nPerThread = 4, kpack = 1, splitKFactor = 1>
#xdlops_gemm_params0 = #rock.xdlops_gemm_derived_params<kpackPerBlock = 8, mPerBlock = 64, nPerBlock = 64, kpack = 1, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, forceUnroll = true>
#xdlops_gemm_params1 = #rock.xdlops_gemm_derived_params<kpackPerBlock = 4, mPerBlock = 128, nPerBlock = 128, kpack = 4, mPerWave = 64, nPerWave = 64, mnPerXdl = 32, splitKFactor = 1, forceUnroll = true>

func.func @rock_conv(%filter : memref<1x128x8x3x3xf32>, %input : memref<128x1x8x32x32xf32>, %output : memref<128x1x128x30x30xf32>) {
  rock.conv(%filter, %input, %output) features = none {
    arch = "amdgcn-amd-amdhsa:gfx906",
    blockSize = 256 : i32,
    dilations = [1 : index,  1 : index],
    filter_layout = ["g", "k", "c", "0", "1"],
    gridSize = 900 : i32,
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    params = #general_gemm_params0,
    strides = [1 : index,  1 : index]
  } : memref<1x128x8x3x3xf32>, memref<128x1x8x32x32xf32>, memref<128x1x128x30x30xf32>
  return
}
// CHECK-LABEL: func.func {{@rock_conv.*%arg0.*%arg1.*%arg2}}
// CHECK-NOT:   rock.conv
// CHECK-NEXT:  %[[FILTER:.*]] = rock.transform %arg0 by #[[$MAP_FILTER_FWD]]
// CHECK-NEXT:  %[[IN1:.*]] = rock.transform %arg1 by #[[$MAP_INPUT1_FWD]]
// CHECK-NEXT:  %[[IN2:.*]] = rock.transform %[[IN1]] by #[[$MAP_INPUT2_FWD]]
// CHECK-NEXT:  %[[IN3:.*]] = rock.transform %[[IN2]] by #[[$MAP_INPUT3_FWD]]
// CHECK-NEXT:  %[[OUT:.*]] = rock.transform %arg2 by #[[$MAP_OUTPUT_FWD]]
// CHECK-NEXT:  rock.gemm %[[OUT]] = tr %[[FILTER]] * %[[IN3]]

func.func @rock_conv_f16(%filter : memref<1x128x8x3x3xf16>, %input : memref<128x1x8x32x32xf16>, %output : memref<128x1x128x30x30xf16>) {
  rock.conv(%filter, %input, %output) features = none {
    arch = "amdgcn-amd-amdhsa:gfx906",
    blockSize = 256 : i32,
    dilations = [1 : index,  1 : index],
    filter_layout = ["g", "k", "c", "0", "1"],
    gridSize = 900 : i32,
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    params = #general_gemm_params0,
    strides = [1 : index,  1 : index]
  } : memref<1x128x8x3x3xf16>, memref<128x1x8x32x32xf16>, memref<128x1x128x30x30xf16>
  return
}
// CHECK-LABEL: func.func {{@rock_conv_f16.*%arg0.*%arg1.*%arg2}}
// CHECK-NOT:   rock.conv
// CHECK-NEXT:  %[[FILTER:.*]] = rock.transform %arg0 by #[[$MAP_FILTER_FWD]]
// CHECK-NEXT:  %[[IN1:.*]] = rock.transform %arg1 by #[[$MAP_INPUT1_FWD]]
// CHECK-NEXT:  %[[IN2:.*]] = rock.transform %[[IN1]] by #[[$MAP_INPUT2_FWD]]
// CHECK-NEXT:  %[[IN3:.*]] = rock.transform %[[IN2]] by #[[$MAP_INPUT3_FWD]]
// CHECK-NEXT:  %[[OUT:.*]] = rock.transform %arg2 by #[[$MAP_OUTPUT_FWD]]
// CHECK-NEXT:  rock.gemm %[[OUT]] = tr %[[FILTER]] * %[[IN3]]

func.func @rock_conv_i8(%filter : memref<1x128x8x3x3xi8>, %input : memref<128x1x8x32x32xi8>, %output : memref<128x1x128x30x30xi32>) {
  rock.conv(%filter, %input, %output) features = mfma|dot|atomic_add|atomic_add_f16 {
    arch = "amdgcn-amd-amdhsa:gfx908",
    blockSize = 256 : i32,
    dilations = [1 : index,  1 : index],
    filter_layout = ["g", "k", "c", "0", "1"],
    gridSize = 3600 : i32,
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    params = #xdlops_gemm_params0,
    strides = [1 : index,  1 : index]
  } : memref<1x128x8x3x3xi8>, memref<128x1x8x32x32xi8>, memref<128x1x128x30x30xi32>
  return
}
// CHECK-LABEL: func.func {{@rock_conv_i8.*%arg0.*%arg1.*%arg2}}
// CHECK-NOT:   rock.conv
// CHECK-NEXT:  %[[FILTER:.*]] = rock.transform %arg0 by #[[$MAP_FILTER_FWD]]
// CHECK-NEXT:  %[[IN1:.*]] = rock.transform %arg1 by #[[$MAP_INPUT1_FWD]]
// CHECK-NEXT:  %[[IN2:.*]] = rock.transform %[[IN1]] by #[[$MAP_INPUT2_FWD]]
// CHECK-NEXT:  %[[IN3:.*]] = rock.transform %[[IN2]] by #[[$MAP_INPUT3_FWD]]
// CHECK-NEXT:  %[[OUT:.*]] = rock.transform %arg2 by #[[$MAP_OUTPUT_FWD]]
// CHECK-NEXT:  rock.gemm %[[OUT]] = tr %[[FILTER]] * %[[IN3]]


func.func @rock_conv_bwd_data(%filter: memref<1x1024x1024x1x1xf32>, %input: memref<128x1x1024x14x14xf32>, %output: memref<128x1x1024x14x14xf32>) attributes {kernel = 0 : i32} {
  rock.conv_bwd_data(%filter, %input, %output) features = mfma|dot|atomic_add|atomic_add_f16 {
    arch = "amdgcn-amd-amdhsa:gfx908",
    blockSize = 256 : i32,
    dilations = [1 : index, 1 : index],
    filter_layout = ["g", "k", "c", "0", "1"],
    gridSize = 900 : i32,
    kernelId = 0 : index,
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    params = #xdlops_gemm_params1,
    strides = [1 : index, 1 : index]
  } : memref<1x1024x1024x1x1xf32>, memref<128x1x1024x14x14xf32>, memref<128x1x1024x14x14xf32>
  return
}

// CHECK-LABEL: func.func {{@rock_conv_bwd_data.*%arg0.*%arg1.*%arg2}}
// CHECK-NOT:   rock.conv_bwd_data
// CHECK-NEXT:  %[[FIL1:.*]] = rock.transform %arg0 by #[[$MAP_BWD_DATA_FIL1_NO_PAD]]
// CHECK-NEXT:  %[[FIL2:.*]] = rock.transform %[[FIL1]] by #[[$MAP_BWD_DATA_FIL2_NO_PAD]]
// CHECK-NEXT:  %[[FIL3:.*]] = rock.transform %[[FIL2]] by #[[$MAP_BWD_DATA_FIL3_NO_PAD]]
// CHECK-NEXT:  %[[IN1:.*]] = rock.transform %arg1 by #[[$MAP_BWD_DATA_IN1_NO_PAD]]
// CHECK-NEXT:  %[[IN2:.*]] = rock.transform %[[IN1]] by #[[$MAP_BWD_DATA_IN2_NO_PAD]]
// CHECK-NEXT:  %[[IN3:.*]] = rock.transform %[[IN2]] by #[[$MAP_BWD_DATA_IN3_NO_PAD]]
// CHECK-NEXT:  %[[IN4:.*]] = rock.transform %[[IN3]] by #[[$MAP_BWD_DATA_IN4_NO_PAD]]
// CHECK-NEXT:  %[[OUT1:.*]] = rock.transform %arg2 by #[[$MAP_BWD_DATA_OUT1_NO_PAD]]
// CHECK-NEXT:  %[[OUT2:.*]] = rock.transform %[[OUT1]] by #[[$MAP_BWD_DATA_OUT2_NO_PAD]]
// CHECK-NEXT:  %[[OUT3:.*]] = rock.transform %[[OUT2]] by #[[$MAP_BWD_DATA_OUT3_NO_PAD]]
// CHECK-NEXT:  rock.gemm %[[IN4]] = tr %[[FIL3]] * %[[OUT3]]{{.*}}

func.func @rock_conv_bwd_data_f16(%filter: memref<1x1024x1024x1x1xf16>, %input: memref<128x1x1024x14x14xf16>, %output: memref<128x1x1024x14x14xf16>) attributes {kernel = 0 : i32} {
rock.conv_bwd_data(%filter, %input, %output) features = mfma|dot|atomic_add|atomic_add_f16 {
    arch = "amdgcn-amd-amdhsa:gfx908",
    blockSize = 256 : i32,
    dilations = [1 : index, 1 : index],
    filter_layout = ["g", "k", "c", "0", "1"],
    gridSize = 1568 : i32,
    kernelId = 0 : index,
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    params = #xdlops_gemm_params1,
    strides = [1 : index, 1 : index]
  } : memref<1x1024x1024x1x1xf16>, memref<128x1x1024x14x14xf16>, memref<128x1x1024x14x14xf16>
  return
}

// CHECK-LABEL: func.func {{@rock_conv_bwd_data_f16.*%arg0.*%arg1.*%arg2}}
// CHECK-NOT:   rock.conv_bwd_data
// CHECK-NEXT:  %[[FIL1:.*]] = rock.transform %arg0 by #[[$MAP_BWD_DATA_FIL1_NO_PAD]]
// CHECK-NEXT:  %[[FIL2:.*]] = rock.transform %[[FIL1]] by #[[$MAP_BWD_DATA_FIL2_NO_PAD]]
// CHECK-NEXT:  %[[FIL3:.*]] = rock.transform %[[FIL2]] by #[[$MAP_BWD_DATA_FIL3_NO_PAD]]
// CHECK-NEXT:  %[[IN1:.*]] = rock.transform %arg1 by #[[$MAP_BWD_DATA_IN1_NO_PAD]]
// CHECK-NEXT:  %[[IN2:.*]] = rock.transform %[[IN1]] by #[[$MAP_BWD_DATA_IN2_NO_PAD]]
// CHECK-NEXT:  %[[IN3:.*]] = rock.transform %[[IN2]] by #[[$MAP_BWD_DATA_IN3_NO_PAD]]
// CHECK-NEXT:  %[[IN4:.*]] = rock.transform %[[IN3]] by #[[$MAP_BWD_DATA_IN4_NO_PAD]]
// CHECK-NEXT:  %[[OUT1:.*]] = rock.transform %arg2 by #[[$MAP_BWD_DATA_OUT1_NO_PAD]]
// CHECK-NEXT:  %[[OUT2:.*]] = rock.transform %[[OUT1]] by #[[$MAP_BWD_DATA_OUT2_NO_PAD]]
// CHECK-NEXT:  %[[OUT3:.*]] = rock.transform %[[OUT2]] by #[[$MAP_BWD_DATA_OUT3_NO_PAD]]
// CHECK-NEXT:  rock.gemm %[[IN4]] = tr %[[FIL3]] * %[[OUT3]]{{.*}}

func.func @rock_conv_bwd_weight(%filter : memref<1x128x8x3x3xf32>, %input : memref<128x1x8x32x32xf32>, %output : memref<128x1x128x30x30xf32>) {
  rock.conv_bwd_weight(%filter, %input, %output) features = none {
    arch = "amdgcn-amd-amdhsa:gfx906",
    blockSize = 64 : i32,
    dilations = [1 : index, 1 : index],
    filter_layout = ["g", "k", "c", "0", "1"],
    gridSize = 4 : i32,
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    numCU = 64 : i32,
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    params = #general_gemm_params1,
    strides = [1 : index,  1 : index]
  } : memref<1x128x8x3x3xf32>, memref<128x1x8x32x32xf32>, memref<128x1x128x30x30xf32>
  return
}
// CHECK-LABEL: func.func {{@rock_conv_bwd_weight.*%arg0.*%arg1.*%arg2}}
// CHECK-NOT:   rock.conv_bwd_weight
// CHECK-NEXT:  %[[FIL1:.*]] = rock.transform %arg0 by #[[$MAP_BWD_WEIGHT_FIL1]]
// CHECK-NEXT:  %[[IN1:.*]] = rock.transform %arg1 by #[[$MAP_INPUT1_FWD]]
// CHECK-NEXT:  %[[IN2:.*]] = rock.transform %[[IN1]] by #[[$MAP_INPUT2_FWD]]
// CHECK-NEXT:  %[[IN3:.*]] = rock.transform %[[IN2]] by #[[$MAP_BWD_WEIGHT_IN3]]
// CHECK-NEXT:  %[[OUT:.*]] = rock.transform %arg2 by #[[$MAP_BWD_WEIGHT_OUT]]
// CHECK-NEXT:  rock.gemm %[[FIL1]] = tr %[[OUT]] * %[[IN3]]{{.*}}

func.func @rock_conv_bwd_weight_f16(%filter : memref<1x128x8x3x3xf16>, %input : memref<128x1x8x32x32xf16>, %output : memref<128x1x128x30x30xf16>) {
  rock.conv_bwd_weight(%filter, %input, %output) features = none {
    arch = "amdgcn-amd-amdhsa:gfx906",
    blockSize = 64 : i32,
    dilations = [1 : index,  1 : index],
    filter_layout = ["g", "k", "c", "0", "1"],
    gridSize = 4 : i32,
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    numCU = 64 : i32,
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    params = #general_gemm_params1,
    strides = [1 : index,  1 : index]
  } : memref<1x128x8x3x3xf16>, memref<128x1x8x32x32xf16>, memref<128x1x128x30x30xf16>
  return
}
// CHECK-LABEL: func.func {{@rock_conv_bwd_weight_f16.*%arg0.*%arg1.*%arg2}}
// CHECK-NOT:   rock.conv_bwd_weight
// CHECK-NEXT:  %[[FIL1:.*]] = rock.transform %arg0 by #[[$MAP_BWD_WEIGHT_FIL1]]
// CHECK-NEXT:  %[[IN1:.*]] = rock.transform %arg1 by #[[$MAP_INPUT1_FWD]]
// CHECK-NEXT:  %[[IN2:.*]] = rock.transform %[[IN1]] by #[[$MAP_INPUT2_FWD]]
// CHECK-NEXT:  %[[IN3:.*]] = rock.transform %[[IN2]] by #[[$MAP_BWD_WEIGHT_IN3]]
// CHECK-NEXT:  %[[OUT:.*]] = rock.transform %arg2 by #[[$MAP_BWD_WEIGHT_OUT]]
// CHECK-NEXT:  rock.gemm %[[FIL1]] = tr %[[OUT]] * %[[IN3]]{{.*}}
