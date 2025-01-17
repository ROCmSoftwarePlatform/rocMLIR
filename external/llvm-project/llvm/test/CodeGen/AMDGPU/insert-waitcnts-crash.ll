; NOTE: Assertions have been autogenerated by utils/update_mir_test_checks.py UTC_ARGS: --version 4
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1030 -stop-after=si-insert-waitcnts -verify-machineinstrs < %s | FileCheck %s

declare fastcc void @bar()

define fastcc i32 @foo() {
  ; CHECK-LABEL: name: foo
  ; CHECK: bb.0 (%ir-block.0):
  ; CHECK-NEXT:   successors: %bb.1(0x80000000)
  ; CHECK-NEXT:   liveins: $sgpr12, $sgpr13, $sgpr14, $sgpr15, $vgpr31, $sgpr4_sgpr5, $sgpr6_sgpr7, $sgpr8_sgpr9, $sgpr10_sgpr11, $sgpr30_sgpr31
  ; CHECK-NEXT: {{  $}}
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION llvm_def_aspace_cfa $sgpr32_lo16, 0, 6
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION llvm_register_pair $pc_reg, $sgpr30_lo16, 32, $sgpr31_lo16, 32
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr0_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr1_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr2_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr3_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr4_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr5_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr6_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr7_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr8_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr9_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr10_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr11_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr12_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr13_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr14_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr15_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr16_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr17_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr18_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr19_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr20_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr21_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr22_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr23_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr24_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr25_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr26_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr27_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr28_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr29_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr30_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr31_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr32_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr33_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr34_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr35_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr36_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr37_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr38_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr39_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr48_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr49_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr50_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr51_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr52_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr53_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr54_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr55_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr64_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr65_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr66_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr67_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr68_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr69_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr70_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr71_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr80_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr81_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr82_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr83_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr84_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr85_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr86_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr87_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr96_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr97_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr98_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr99_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr100_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr101_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr102_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr103_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr112_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr113_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr114_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr115_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr116_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr117_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr118_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr119_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr128_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr129_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr130_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr131_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr132_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr133_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr134_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr135_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr144_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr145_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr146_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr147_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr148_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr149_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr150_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr151_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr160_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr161_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr162_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr163_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr164_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr165_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr166_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr167_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr176_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr177_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr178_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr179_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr180_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr181_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr182_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr183_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr192_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr193_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr194_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr195_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr196_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr197_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr198_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr199_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr208_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr209_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr210_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr211_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr212_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr213_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr214_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr215_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr224_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr225_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr226_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr227_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr228_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr229_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr230_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr231_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr240_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr241_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr242_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr243_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr244_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr245_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr246_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $vgpr247_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $sgpr0_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $sgpr1_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $sgpr2_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $sgpr3_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $sgpr4_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $sgpr5_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $sgpr6_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $sgpr7_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $sgpr8_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $sgpr9_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $sgpr10_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $sgpr11_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $sgpr12_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $sgpr13_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $sgpr14_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $sgpr15_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $sgpr16_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $sgpr17_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $sgpr18_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $sgpr19_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $sgpr20_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $sgpr21_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $sgpr22_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $sgpr23_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $sgpr24_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $sgpr25_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $sgpr26_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $sgpr27_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $sgpr28_lo16
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION undefined $sgpr29_lo16
  ; CHECK-NEXT:   S_WAITCNT 0
  ; CHECK-NEXT:   $sgpr16 = S_MOV_B32 $sgpr33
  ; CHECK-NEXT:   $sgpr33 = S_MOV_B32 $sgpr32
  ; CHECK-NEXT:   $sgpr17 = S_OR_SAVEEXEC_B32 -1, implicit-def $exec, implicit-def dead $scc, implicit $exec
  ; CHECK-NEXT:   BUFFER_STORE_DWORD_OFFSET $vgpr40, $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr33, 0, 0, 0, implicit $exec :: (store (s32) into %stack.1, addrspace 5)
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION offset $vgpr40_lo16, 0
  ; CHECK-NEXT:   $exec_lo = S_MOV_B32 killed $sgpr17
  ; CHECK-NEXT:   $vgpr40 = V_WRITELANE_B32 killed $sgpr16, 2, undef $vgpr40
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION llvm_vector_registers $sgpr33_lo16, $vgpr40_lo16, 2, 32
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION def_cfa_register $sgpr33_lo16
  ; CHECK-NEXT:   $vgpr40 = V_WRITELANE_B32 killed $sgpr30, 0, $vgpr40, implicit-def $sgpr30_sgpr31, implicit $sgpr30_sgpr31
  ; CHECK-NEXT:   $sgpr32 = frame-setup S_ADDK_I32 $sgpr32, 512, implicit-def dead $scc
  ; CHECK-NEXT:   $vgpr40 = V_WRITELANE_B32 killed $sgpr31, 1, $vgpr40, implicit $sgpr30_sgpr31
  ; CHECK-NEXT:   frame-setup CFI_INSTRUCTION llvm_vector_registers $pc_reg, $vgpr40_lo16, 0, 32, $vgpr40_lo16, 1, 32
  ; CHECK-NEXT:   BUNDLE implicit-def $sgpr16_sgpr17, implicit-def $sgpr16, implicit-def $sgpr16_lo16, implicit-def $sgpr16_hi16, implicit-def $sgpr17, implicit-def $sgpr17_lo16, implicit-def $sgpr17_hi16, implicit-def $scc {
  ; CHECK-NEXT:     $sgpr16_sgpr17 = S_GETPC_B64
  ; CHECK-NEXT:     $sgpr16 = S_ADD_U32 internal $sgpr16, target-flags(amdgpu-gotprel32-lo) @bar + 4, implicit-def $scc
  ; CHECK-NEXT:     $sgpr17 = S_ADDC_U32 internal $sgpr17, target-flags(amdgpu-gotprel32-hi) @bar + 12, implicit-def $scc, implicit internal $scc
  ; CHECK-NEXT:   }
  ; CHECK-NEXT:   S_WAITCNT_VSCNT undef $sgpr_null, 0
  ; CHECK-NEXT:   BUFFER_GL1_INV implicit $exec
  ; CHECK-NEXT:   BUFFER_GL0_INV implicit $exec
  ; CHECK-NEXT:   renamable $sgpr16_sgpr17 = S_LOAD_DWORDX2_IMM killed renamable $sgpr16_sgpr17, 0, 0 :: (dereferenceable invariant load (s64) from got, addrspace 4)
  ; CHECK-NEXT:   S_WAITCNT 49279
  ; CHECK-NEXT:   dead $sgpr30_sgpr31 = SI_CALL killed renamable $sgpr16_sgpr17, @bar, csr_amdgpu, implicit killed $sgpr4_sgpr5, implicit killed $sgpr6_sgpr7, implicit killed $sgpr8_sgpr9, implicit killed $sgpr10_sgpr11, implicit killed $sgpr12, implicit killed $sgpr13, implicit killed $sgpr14, implicit killed $sgpr15, implicit killed $vgpr31, implicit $sgpr0_sgpr1_sgpr2_sgpr3
  ; CHECK-NEXT:   $vcc_lo = S_MOV_B32 $exec_lo
  ; CHECK-NEXT: {{  $}}
  ; CHECK-NEXT: bb.1 (%ir-block.1):
  ; CHECK-NEXT:   successors: %bb.2(0x04000000), %bb.1(0x7c000000)
  ; CHECK-NEXT:   liveins: $vcc_lo
  ; CHECK-NEXT: {{  $}}
  ; CHECK-NEXT:   S_CBRANCH_VCCNZ %bb.1, implicit $vcc_lo
  ; CHECK-NEXT: {{  $}}
  ; CHECK-NEXT: bb.2.DummyReturnBlock:
  ; CHECK-NEXT:   $sgpr30 = V_READLANE_B32 $vgpr40, 0, implicit-def $sgpr30_sgpr31
  ; CHECK-NEXT:   $sgpr31 = V_READLANE_B32 $vgpr40, 1
  ; CHECK-NEXT:   $sgpr4 = V_READLANE_B32 $vgpr40, 2
  ; CHECK-NEXT:   $sgpr5 = S_OR_SAVEEXEC_B32 -1, implicit-def $exec, implicit-def dead $scc, implicit $exec
  ; CHECK-NEXT:   $vgpr40 = BUFFER_LOAD_DWORD_OFFSET $sgpr0_sgpr1_sgpr2_sgpr3, $sgpr33, 0, 0, 0, implicit $exec :: (load (s32) from %stack.1, addrspace 5)
  ; CHECK-NEXT:   $exec_lo = S_MOV_B32 killed $sgpr5
  ; CHECK-NEXT:   $sgpr32 = frame-destroy S_ADDK_I32 $sgpr32, -512, implicit-def dead $scc
  ; CHECK-NEXT:   frame-destroy CFI_INSTRUCTION def_cfa_register $sgpr32_lo16
  ; CHECK-NEXT:   $sgpr33 = S_MOV_B32 killed $sgpr4
  ; CHECK-NEXT:   S_WAITCNT 16240
  ; CHECK-NEXT:   S_SETPC_B64_return undef $sgpr30_sgpr31, implicit undef $vgpr0
  fence acquire
  call fastcc void @bar()
  br label %1

1:
  br label %1
}
