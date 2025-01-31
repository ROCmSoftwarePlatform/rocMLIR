; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py UTC_ARGS: --version 5

; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1030 %s -o - | FileCheck %s

; Testing codegen for memcpy with scalar reads.


define void @memcpy_p1_p4_sz16_align_4_4(ptr addrspace(1) align 4 %dst, ptr addrspace(4) align 4 readonly inreg %src) {
; CHECK-LABEL: memcpy_p1_p4_sz16_align_4_4:
; CHECK:       ; %bb.0: ; %entry
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    s_load_dwordx4 s[4:7], s[16:17], 0x0
; CHECK-NEXT:    s_waitcnt lgkmcnt(0)
; CHECK-NEXT:    v_mov_b32_e32 v2, s4
; CHECK-NEXT:    v_mov_b32_e32 v3, s5
; CHECK-NEXT:    v_mov_b32_e32 v4, s6
; CHECK-NEXT:    v_mov_b32_e32 v5, s7
; CHECK-NEXT:    global_store_dwordx4 v[0:1], v[2:5], off
; CHECK-NEXT:    s_setpc_b64 s[30:31]
entry:
  tail call void @llvm.memcpy.p1.p4.i64(ptr addrspace(1) noundef nonnull align 4 %dst, ptr addrspace(4) noundef nonnull align 4 %src, i64 16, i1 false)
  ret void
}

define void @memcpy_p1_p4_sz31_align_4_4(ptr addrspace(1) align 4 %dst, ptr addrspace(4) align 4 readonly inreg %src) {
; CHECK-LABEL: memcpy_p1_p4_sz31_align_4_4:
; CHECK:       ; %bb.0: ; %entry
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    s_load_dwordx4 s[4:7], s[16:17], 0x0
; CHECK-NEXT:    v_mov_b32_e32 v6, 0
; CHECK-NEXT:    s_waitcnt lgkmcnt(0)
; CHECK-NEXT:    v_mov_b32_e32 v2, s4
; CHECK-NEXT:    v_mov_b32_e32 v3, s5
; CHECK-NEXT:    v_mov_b32_e32 v4, s6
; CHECK-NEXT:    v_mov_b32_e32 v5, s7
; CHECK-NEXT:    global_store_dwordx4 v[0:1], v[2:5], off
; CHECK-NEXT:    global_load_dwordx4 v[2:5], v6, s[16:17] offset:15
; CHECK-NEXT:    s_waitcnt vmcnt(0)
; CHECK-NEXT:    global_store_dwordx4 v[0:1], v[2:5], off offset:15
; CHECK-NEXT:    s_setpc_b64 s[30:31]
entry:
  tail call void @llvm.memcpy.p1.p4.i64(ptr addrspace(1) noundef nonnull align 4 %dst, ptr addrspace(4) noundef nonnull align 4 %src, i64 31, i1 false)
  ret void
}

define void @memcpy_p1_p4_sz32_align_4_4(ptr addrspace(1) align 4 %dst, ptr addrspace(4) align 4 readonly inreg %src) {
; CHECK-LABEL: memcpy_p1_p4_sz32_align_4_4:
; CHECK:       ; %bb.0: ; %entry
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    s_load_dwordx8 s[4:11], s[16:17], 0x0
; CHECK-NEXT:    s_waitcnt lgkmcnt(0)
; CHECK-NEXT:    v_mov_b32_e32 v2, s4
; CHECK-NEXT:    v_mov_b32_e32 v3, s5
; CHECK-NEXT:    v_mov_b32_e32 v4, s6
; CHECK-NEXT:    v_mov_b32_e32 v5, s7
; CHECK-NEXT:    v_mov_b32_e32 v6, s8
; CHECK-NEXT:    v_mov_b32_e32 v7, s9
; CHECK-NEXT:    v_mov_b32_e32 v8, s10
; CHECK-NEXT:    v_mov_b32_e32 v9, s11
; CHECK-NEXT:    global_store_dwordx4 v[0:1], v[2:5], off
; CHECK-NEXT:    global_store_dwordx4 v[0:1], v[6:9], off offset:16
; CHECK-NEXT:    s_setpc_b64 s[30:31]
entry:
  tail call void @llvm.memcpy.p1.p4.i64(ptr addrspace(1) noundef nonnull align 4 %dst, ptr addrspace(4) noundef nonnull align 4 %src, i64 32, i1 false)
  ret void
}

declare void @llvm.memcpy.p1.p4.i64(ptr addrspace(1) noalias nocapture writeonly, ptr addrspace(4) noalias nocapture readonly, i64, i1 immarg) #2

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }

