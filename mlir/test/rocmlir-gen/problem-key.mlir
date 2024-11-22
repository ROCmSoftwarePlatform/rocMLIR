// RUN: rocmlir-gen --arch gfx942 --operation attention -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -t f32 -g 1 | rocmlir-gen --emit-tuning-key - | FileCheck %s  --check-prefixes=CHECK_1
// CHECK_1: -t f32 -transQ false -transK false -transV false -transO false -g 1 -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32
// RUN: rocmlir-gen --arch gfx942 --operation attention -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -t f16 -g 4 | rocmlir-gen --emit-tuning-key - | FileCheck %s  --check-prefixes=CHECK_2
// CHECK_2: -t f16 -transQ false -transK false -transV false -transO false -g 4 -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32
// RUN: rocmlir-gen --arch gfx942 --operation attention -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -t i8 -g 8 | rocmlir-gen --emit-tuning-key - | FileCheck %s  --check-prefixes=CHECK_3
// CHECK_3: -t i8 -transQ false -transK false -transV false -transO false -g 8 -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32
// RUN: rocmlir-gen --arch gfx942 --operation attention -num_heads_q 4 -num_heads_kv 4 -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -t i8 -g 8 | rocmlir-gen --emit-tuning-key - | FileCheck %s  --check-prefixes=CHECK_4
// CHECK_4: -t i8 -transQ false -transK false -transV false -transO false -g 32 -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32
// RUN: rocmlir-gen --arch gfx942 --operation attention -num_heads_q 4 -num_heads_kv 2 -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -t i8 -g 8 | rocmlir-gen --emit-tuning-key - | FileCheck %s  --check-prefixes=CHECK_5
// CHECK_5: -t i8 -transQ false -transK false -transV false -transO false -g 32 -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32
// RUN: rocmlir-gen --arch gfx942 --operation attention -current_seq_len=16 -num_heads_q 4 -num_heads_kv 2 -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -t i8 -g 1 | rocmlir-gen --emit-tuning-key - | FileCheck %s  --check-prefixes=CHECK_6
// CHECK_6: -t i8 -transQ false -transK false -transV false -transO false -g 4 -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32
// RUN: rocmlir-gen --arch gfx942 --operation attention -current_seq_len=16,16,17,1,30,40,38,12 -num_heads_q 4 -num_heads_kv 2 -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -t i8 -g 8 | rocmlir-gen --emit-tuning-key - | FileCheck %s  --check-prefixes=CHECK_7
// CHECK_7: -t i8 -transQ false -transK false -transV false -transO false -g 32 -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32

// Checking numCU

// RUN: rocmlir-gen --arch gfx942 --num_cu 304 --operation attention -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -t f16 -g 4 | rocmlir-gen --emit-tuning-key - | FileCheck %s  --check-prefixes=CHECK_NUMCU
// CHECK_NUMCU: 304
