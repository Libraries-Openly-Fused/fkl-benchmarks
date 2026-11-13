
e18_stencil_hip.0.hip-amdgcn-amd-amdhsa--gfx1200:	file format elf64-amdgpu
	.amdgcn_target "amdgpu-amd-amdhsa-unknown-gfx1200"

Disassembly of section .text:

0000000000006c00 <__cxa_pure_virtual>:
	s_wait_loadcnt_dscnt 0x0                                   // 000000006C00: BFC80000
	s_wait_expcnt 0x0                                          // 000000006C04: BFC40000
	s_wait_samplecnt 0x0                                       // 000000006C08: BFC20000
	s_wait_bvhcnt 0x0                                          // 000000006C0C: BFC30000
	s_wait_kmcnt 0x0                                           // 000000006C10: BFC70000
	s_trap 2                                                   // 000000006C14: BF900002
		...

0000000000006c80 <__cxa_deleted_virtual>:
	s_wait_loadcnt_dscnt 0x0                                   // 000000006C80: BFC80000
	s_wait_expcnt 0x0                                          // 000000006C84: BFC40000
	s_wait_samplecnt 0x0                                       // 000000006C88: BFC20000
	s_wait_bvhcnt 0x0                                          // 000000006C8C: BFC30000
	s_wait_kmcnt 0x0                                           // 000000006C90: BFC70000
	s_trap 2                                                   // 000000006C94: BF900002
		...

0000000000006d00 <_ZN6e18amd7gpuSpinEy>:
	s_load_b64 s[0:1], s[0:1], 0x0                             // 000000006D00: F4002000 F8000000
	s_sendmsg_rtn_b64 s[2:3], sendmsg(MSG_RTN_GET_REALTIME)    // 000000006D08: BE824D83

0000000000006d0c <L0>:
	s_sendmsg_rtn_b64 s[4:5], sendmsg(MSG_RTN_GET_REALTIME)    // 000000006D0C: BE844D83
	s_wait_kmcnt 0x0                                           // 000000006D10: BFC70000
	s_wait_alu depctr_sa_sdst(0)                               // 000000006D14: BF88FF9E
	s_sub_nc_u64 s[4:5], s[4:5], s[2:3]                        // 000000006D18: AA040204
	s_wait_alu depctr_sa_sdst(0)                               // 000000006D1C: BF88FF9E
	v_cmp_lt_u64_e64 s4, s[4:5], s[0:1]                        // 000000006D20: D4590004 02000004
	s_and_b32 vcc_lo, exec_lo, s4                              // 000000006D28: 8B6A047E
	s_cbranch_vccnz L0                                         // 000000006D2C: BFA4FFF7
	s_endpgm                                                   // 000000006D30: BFB00000
		...

0000000000006e00 <_ZN2fk21linearFilterDPPKernelILNS_7ParArchE3ENS_22LinearFilterDPPDetailsINS_6Float3ELi16ELi8ELi7ELi7EEENS_5TupleIJNS_29ReadBackInstantiableOperationINS_14ResizeCompleteILNS_11AspectRatioE1ENS_28TernaryInstantiableOperationINS_19InterpolateCompleteILNS_17InterpolationTypeE1ENS_25ReadInstantiableOperationINS_13PerThreadReadILNS_2NDE2ENS_6Uchar3EEEEEEEEELNS_14ResizeSamplingE0EEEEENSC_INSD_ILSE_2ES3_EEEEEEENS_26UnaryInstantiableOperationINS_3MulIS3_S3_S3_NS_9UnaryTypeEEEEENSQ_INS_3AddIS3_S3_S3_SS_EEEENS_26WriteInstantiableOperationINS_15FusedOperation_IvJNSQ_INS_12SaturateCastIS3_SF_EEEENSY_INS_14PerThreadWriteILSE_2ESF_EEEEEEEEEEEvT0_T1_T2_T3_T4_>:
	s_clause 0x3                                               // 000000006E00: BF850003
	s_load_b32 s10, s[0:1], 0x8c                               // 000000006E04: F4000280 F800008C
	s_load_b128 s[4:7], s[0:1], 0x0                            // 000000006E0C: F4004100 F8000000
	s_load_b64 s[8:9], s[0:1], 0x68                            // 000000006E14: F4002200 F8000068
	s_load_b64 s[2:3], s[0:1], 0x48                            // 000000006E1C: F4002080 F8000048
	v_bfe_u32 v5, v0, 10, 10                                   // 000000006E24: D6100005 02291500
	v_and_b32_e32 v0, 0x3ff, v0                                // 000000006E2C: 360000FF 000003FF
	s_lshl_b32 s15, ttmp9, 4                                   // 000000006E34: 840F8475
	s_lshl_b32 s16, ttmp7, 3                                   // 000000006E38: 84108373
	s_mov_b32 s19, exec_lo                                     // 000000006E3C: BE93007E
	s_wait_kmcnt 0x0                                           // 000000006E40: BFC70000
	s_and_b32 s11, s10, 0xffff                                 // 000000006E44: 8B0BFF0A 0000FFFF
	s_add_co_i32 s14, s6, 15                                   // 000000006E4C: 810E8F06
	v_mad_u32_u24 v1, v5, s11, v0                              // 000000006E50: D60B0001 04001705
	s_add_co_i32 s18, s7, 7                                    // 000000006E58: 81128707
	s_lshr_b32 s17, s10, 16                                    // 000000006E5C: 8511900A
	s_mul_i32 s18, s18, s14                                    // 000000006E60: 96120E12
	s_mul_i32 s17, s17, s11                                    // 000000006E64: 96110B11
	v_cmpx_gt_i32_e32 s18, v1                                  // 000000006E68: 7D880212
	s_cbranch_execz L2                                         // 000000006E6C: BFA500D5
	s_abs_i32 s20, s14                                         // 000000006E70: BE94150E
	s_add_nc_u64 s[22:23], s[0:1], 24                          // 000000006E74: A9969800
	s_cvt_f32_u32 s10, s20                                     // 000000006E78: BE8A6514
	s_clause 0x1                                               // 000000006E7C: BF850001
	s_load_b64 s[28:29], s[0:1], 0x10                          // 000000006E80: F4002700 F8000010
	s_load_b64 s[30:31], s[0:1], 0x38                          // 000000006E88: F4002780 F8000038
	v_mul_lo_u32 v2, v1, 12                                    // 000000006E90: D72C0002 02011901
	v_s_rcp_f32 s26, s10                                       // 000000006E98: D684001A 0201000A
	s_clause 0x2                                               // 000000006EA0: BF850002
	s_load_b64 s[10:11], s[22:23], 0x8                         // 000000006EA4: F400228B F8000008
	s_load_b64 s[12:13], s[22:23], 0x18                        // 000000006EAC: F400230B F8000018
	s_load_b32 s21, s[22:23], 0x28                             // 000000006EB4: F400054B F8000028
	s_sub_co_i32 s34, 0, s20                                   // 000000006EBC: 81A21480
	v_mov_b32_e32 v3, v1                                       // 000000006EC0: 7E060301
	s_add_co_i32 s22, s4, -1                                   // 000000006EC4: 8116C104
	s_add_co_i32 s23, s5, -1                                   // 000000006EC8: 8117C105
	s_ashr_i32 s24, s14, 31                                    // 000000006ECC: 86189F0E
	s_mov_b32 s25, 0                                           // 000000006ED0: BE990080
	s_mul_f32 s27, s26, 0x4f7ffffe                             // 000000006ED4: A21BFF1A 4F7FFFFE
	s_sub_co_i32 s26, 0, s14                                   // 000000006EDC: 819A0E80
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(SALU_CYCLE_3)// 000000006EE0: BF87059A
	s_cvt_u32_f32 s33, s27                                     // 000000006EE4: BEA1671B
	s_mul_i32 s34, s34, s33                                    // 000000006EE8: 96222122
	s_wait_kmcnt 0x0                                           // 000000006EEC: BFC70000
	s_sub_co_i32 s27, s15, s28                                 // 000000006EF0: 819B1C0F
	s_mul_hi_u32 s34, s33, s34                                 // 000000006EF4: 96A22221
	s_sub_co_i32 s28, s16, s29                                 // 000000006EF8: 819C1D10
	s_add_co_i32 s29, s30, -1                                  // 000000006EFC: 811DC11E
	s_add_co_i32 s30, s31, -1                                  // 000000006F00: 811EC11F
	s_add_co_i32 s31, s33, s34                                 // 000000006F04: 811F2221
	s_mul_i32 s33, s17, 12                                     // 000000006F08: 96218C11

0000000000006f0c <L1>:
	v_sub_nc_u32_e32 v4, 0, v3                                 // 000000006F0C: 4C080680
	v_ashrrev_i32_e32 v6, 31, v3                               // 000000006F10: 340C069F
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000006F14: BF870112
	v_max_i32_e32 v4, v4, v3                                   // 000000006F18: 24080704
	v_xor_b32_e32 v6, s24, v6                                  // 000000006F1C: 3A0C0C18
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000006F20: BF870092
	v_mul_hi_u32 v7, v4, s31                                   // 000000006F24: D72D0007 02003F04
	v_mul_lo_u32 v8, v7, s20                                   // 000000006F2C: D72C0008 02002907
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000006F34: BF870091
	v_sub_nc_u32_e32 v4, v4, v8                                // 000000006F38: 4C081104
	v_subrev_nc_u32_e32 v8, s20, v4                            // 000000006F3C: 4E100814
	v_cmp_le_u32_e32 vcc_lo, s20, v4                           // 000000006F40: 7C960814
	s_wait_alu depctr_va_vcc(0)                                // 000000006F44: BF88FF9D
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000006F48: BF870092
	v_dual_cndmask_b32 v4, v4, v8 :: v_dual_add_nc_u32 v9, 1, v7// 000000006F4C: CA601104 04080E81
	v_cndmask_b32_e32 v7, v7, v9, vcc_lo                       // 000000006F54: 020E1307
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000006F58: BF870112
	v_cmp_le_u32_e32 vcc_lo, s20, v4                           // 000000006F5C: 7C960814
	v_add_nc_u32_e32 v8, 1, v7                                 // 000000006F60: 4A100E81
	s_wait_alu depctr_va_vcc(0)                                // 000000006F64: BF88FF9D
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000006F68: BF870091
	v_cndmask_b32_e32 v4, v7, v8, vcc_lo                       // 000000006F6C: 02081107
	v_xor_b32_e32 v4, v4, v6                                   // 000000006F70: 3A080D04
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000006F74: BF870091
	v_sub_nc_u32_e32 v4, v4, v6                                // 000000006F78: 4C080D04
	v_mul_lo_u32 v6, s26, v4                                   // 000000006F7C: D72C0006 0202081A
	v_add_nc_u32_e32 v4, s28, v4                               // 000000006F84: 4A08081C
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 000000006F88: BF870221
	v_min_i32_e32 v7, s23, v4                                  // 000000006F8C: 220E0817
	v_cmp_lt_i32_e32 vcc_lo, -1, v4                            // 000000006F90: 7C8208C1
	v_add3_u32 v4, s27, v3, v6                                 // 000000006F94: D6550004 041A061B
	s_wait_alu depctr_va_vcc(0)                                // 000000006F9C: BF88FF9D
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000006FA0: BF870113
	v_cndmask_b32_e32 v6, 0, v7, vcc_lo                        // 000000006FA4: 020C0E80
	v_min_i32_e32 v7, s22, v4                                  // 000000006FA8: 220E0816
	v_cmp_lt_i32_e32 vcc_lo, -1, v4                            // 000000006FAC: 7C8208C1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000006FB0: BF8700A3
	v_cvt_f32_i32_e32 v4, v6                                   // 000000006FB4: 7E080B06
	s_wait_alu depctr_va_vcc(0)                                // 000000006FB8: BF88FF9D
	v_dual_cndmask_b32 v6, 0, v7 :: v_dual_mul_f32 v7, s11, v4 // 000000006FBC: CA460E80 0606080B
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000006FC4: BF870111
	v_cvt_f32_i32_e32 v14, v6                                  // 000000006FC8: 7E1C0B06
	v_floor_f32_e32 v6, v7                                     // 000000006FCC: 7E0C4907
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000006FD0: BF870112
	v_mul_f32_e32 v7, s10, v14                                 // 000000006FD4: 100E1C0A
	v_cvt_i32_f32_e32 v15, v6                                  // 000000006FD8: 7E1E1106
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000006FDC: BF870112
	v_floor_f32_e32 v6, v7                                     // 000000006FE0: 7E0C4907
	v_add_nc_u32_e32 v16, 1, v15                               // 000000006FE4: 4A201E81
	v_mul_lo_u32 v7, s21, v15                                  // 000000006FE8: D72C0007 02021E15
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000006FF0: BF870193
	v_cvt_i32_f32_e32 v17, v6                                  // 000000006FF4: 7E221106
	v_min_i32_e32 v8, s30, v16                                 // 000000006FF8: 2210201E
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000006FFC: BF870212
	v_add_nc_u32_e32 v18, 1, v17                               // 000000007000: 4A242281
	v_add_co_u32 v6, s34, s12, v7                              // 000000007004: D7002206 02020E0C
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_2)// 00000000700C: BF870143
	v_mul_lo_u32 v10, v8, s21                                  // 000000007010: D72C000A 02002B08
	s_wait_alu depctr_va_sdst(0)                               // 000000007018: BF88F19F
	v_add_co_ci_u32_e64 v7, null, s13, 0, s34                  // 00000000701C: D5207C07 0089000D
	v_min_i32_e32 v12, s29, v18                                // 000000007024: 2218241D
	v_mad_co_i64_i32 v[8:9], null, v17, 3, v[6:7]              // 000000007028: D6FF7C08 04190711
	s_delay_alu instid0(VALU_DEP_4)                            // 000000007030: BF870004
	v_add_co_u32 v10, s34, s12, v10                            // 000000007034: D700220A 0202140C
	s_wait_alu depctr_va_sdst(0)                               // 00000000703C: BF88F19F
	v_add_co_ci_u32_e64 v11, null, s13, 0, s34                 // 000000007040: D5207C0B 0089000D
	v_mad_co_i64_i32 v[6:7], null, v12, 3, v[6:7]              // 000000007048: D6FF7C06 0419070C
	global_load_u8 v19, v[8:9], off offset:2                   // 000000007050: EE04007C 00000013 00000208
	v_mad_co_i64_i32 v[12:13], null, v12, 3, v[10:11]          // 00000000705C: D6FF7C0C 0429070C
	v_mad_co_i64_i32 v[10:11], null, v17, 3, v[10:11]          // 000000007064: D6FF7C0A 04290711
	s_clause 0x6                                               // 00000000706C: BF850006
	global_load_u8 v20, v[6:7], off offset:2                   // 000000007070: EE04007C 00000014 00000206
	global_load_u16 v21, v[12:13], off                         // 00000000707C: EE04807C 00000015 0000000C
	global_load_u16 v22, v[10:11], off                         // 000000007088: EE04807C 00000016 0000000A
	global_load_u16 v6, v[6:7], off                            // 000000007094: EE04807C 00000006 00000006
	global_load_u16 v7, v[8:9], off                            // 0000000070A0: EE04807C 00000007 00000008
	global_load_u8 v8, v[10:11], off offset:2                  // 0000000070AC: EE04007C 00000008 0000020A
	global_load_u8 v9, v[12:13], off offset:2                  // 0000000070B8: EE04007C 00000009 0000020C
	v_cvt_f32_i32_e32 v10, v15                                 // 0000000070C4: 7E140B0F
	v_cvt_f32_i32_e32 v11, v16                                 // 0000000070C8: 7E160B10
	v_cvt_f32_i32_e32 v12, v17                                 // 0000000070CC: 7E180B11
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 0000000070D0: BF870193
	v_fma_f32 v10, s11, v4, -v10                               // 0000000070D4: D613000A 842A080B
	v_fma_f32 v4, -s11, v4, v11                                // 0000000070DC: D6130004 242E080B
	v_cvt_f32_i32_e32 v11, v18                                 // 0000000070E4: 7E160B12
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)// 0000000070E8: BF8701A4
	v_fma_f32 v12, s10, v14, -v12                              // 0000000070EC: D613000C 84321C0A
	v_add_nc_u32_e32 v3, s17, v3                               // 0000000070F4: 4A060611
	v_fma_f32 v11, -s10, v14, v11                              // 0000000070F8: D613000B 242E1C0A
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 000000007100: BF870223
	v_mul_f32_e32 v13, v12, v4                                 // 000000007104: 101A090C
	v_mul_f32_e32 v12, v10, v12                                // 000000007108: 1018190A
	v_cmp_le_i32_e32 vcc_lo, s18, v3                           // 00000000710C: 7C860612
	s_or_b32 s25, vcc_lo, s25                                  // 000000007110: 8C19196A
	s_wait_loadcnt 0x6                                         // 000000007114: BFC00006
	v_cvt_f32_ubyte0_e32 v14, v20                              // 000000007118: 7E1C2314
	v_mul_f32_e32 v4, v4, v11                                  // 00000000711C: 10081704
	s_wait_loadcnt 0x4                                         // 000000007120: BFC00004
	v_cvt_f32_ubyte1_e32 v17, v22                              // 000000007124: 7E222516
	s_wait_loadcnt 0x3                                         // 000000007128: BFC00003
	v_cvt_f32_ubyte0_e32 v16, v6                               // 00000000712C: 7E202306
	v_mul_f32_e32 v10, v10, v11                                // 000000007130: 1014170A
	v_cvt_f32_ubyte0_e32 v11, v19                              // 000000007134: 7E162313
	s_wait_loadcnt 0x2                                         // 000000007138: BFC00002
	v_cvt_f32_ubyte0_e32 v15, v7                               // 00000000713C: 7E1E2307
	v_cvt_f32_ubyte1_e32 v7, v7                                // 000000007140: 7E0E2507
	v_cvt_f32_ubyte1_e32 v6, v6                                // 000000007144: 7E0C2506
	s_wait_loadcnt 0x1                                         // 000000007148: BFC00001
	v_cvt_f32_ubyte0_e32 v8, v8                                // 00000000714C: 7E102308
	v_mul_f32_e32 v11, v4, v11                                 // 000000007150: 10161704
	v_cvt_f32_ubyte0_e32 v18, v21                              // 000000007154: 7E242315
	v_cvt_f32_ubyte1_e32 v19, v21                              // 000000007158: 7E262515
	s_delay_alu instid0(VALU_DEP_3)                            // 00000000715C: BF870003
	v_fmac_f32_e32 v11, v13, v14                               // 000000007160: 56161D0D
	v_mul_f32_e32 v15, v4, v15                                 // 000000007164: 101E1F04
	v_mul_f32_e32 v4, v4, v7                                   // 000000007168: 10080F04
	v_cvt_f32_ubyte0_e32 v14, v22                              // 00000000716C: 7E1C2316
	s_wait_loadcnt 0x0                                         // 000000007170: BFC00000
	v_cvt_f32_ubyte0_e32 v7, v9                                // 000000007174: 7E0E2309
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000007178: BF870093
	v_dual_fmac_f32 v4, v13, v6 :: v_dual_fmac_f32 v15, v13, v16// 00000000717C: C8000D0D 040E210D
	v_dual_fmac_f32 v11, v10, v8 :: v_dual_fmac_f32 v4, v10, v17// 000000007184: C800110A 0B04230A
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000718C: BF870112
	v_fmac_f32_e32 v15, v10, v14                               // 000000007190: 561E1D0A
	v_fmac_f32_e32 v11, v12, v7                                // 000000007194: 56160F0C
	s_delay_alu instid0(VALU_DEP_2)                            // 000000007198: BF870002
	v_dual_fmac_f32 v4, v12, v19 :: v_dual_fmac_f32 v15, v12, v18// 00000000719C: C800270C 040E250C
	ds_store_2addr_b32 v2, v15, v4 offset1:1                   // 0000000071A4: D8380100 00040F02
	ds_store_b32 v2, v11 offset:8                              // 0000000071AC: D8340008 00000B02
	v_add_nc_u32_e32 v2, s33, v2                               // 0000000071B4: 4A040421
	s_wait_alu depctr_sa_sdst(0)                               // 0000000071B8: BF88FF9E
	s_and_not1_b32 exec_lo, exec_lo, s25                       // 0000000071BC: 917E197E
	s_cbranch_execnz L1                                        // 0000000071C0: BFA6FF52

00000000000071c4 <L2>:
	s_or_b32 exec_lo, exec_lo, s19                             // 0000000071C4: 8C7E137E
	s_clause 0x1                                               // 0000000071C8: BF850001
	s_load_b32 s11, s[0:1], 0x58                               // 0000000071CC: F40002C0 F8000058
	s_load_b32 s10, s[0:1], 0x78                               // 0000000071D4: F4000280 F8000078
	s_mul_i32 s0, s7, s6                                       // 0000000071DC: 96000607
	s_mov_b32 s1, exec_lo                                      // 0000000071E0: BE81007E
	v_cmpx_gt_i32_e32 s0, v1                                   // 0000000071E4: 7D880200
	s_cbranch_execz L4                                         // 0000000071E8: BFA50054
	s_abs_i32 s12, s6                                          // 0000000071EC: BE8C1506
	s_wait_alu depctr_sa_sdst(0)                               // 0000000071F0: BF88FF9E
	s_cvt_f32_u32 s13, s12                                     // 0000000071F4: BE8D650C
	s_wait_alu depctr_sa_sdst(0)                               // 0000000071F8: BF88FF9E
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)// 0000000071FC: BF87029A
	v_s_rcp_f32 s13, s13                                       // 000000007200: D684000D 0201000D
	s_mul_f32 s13, s13, 0x4f7ffffe                             // 000000007208: A20DFF0D 4F7FFFFE
	s_wait_alu depctr_sa_sdst(0)                               // 000000007210: BF88FF9E
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000007214: BF8704BA
	s_cvt_u32_f32 s19, s13                                     // 000000007218: BE93670D
	s_sub_co_i32 s13, 0, s12                                   // 00000000721C: 818D0C80
	s_wait_alu depctr_sa_sdst(0)                               // 000000007220: BF88FF9E
	s_mul_i32 s18, s13, s19                                    // 000000007224: 9612130D
	s_ashr_i32 s13, s6, 31                                     // 000000007228: 860D9F06
	s_wait_alu depctr_sa_sdst(0)                               // 00000000722C: BF88FF9E
	s_mul_hi_u32 s20, s19, s18                                 // 000000007230: 96941213
	s_mov_b32 s18, 0                                           // 000000007234: BE920080
	s_wait_alu depctr_sa_sdst(0)                               // 000000007238: BF88FF9E
	s_add_co_i32 s19, s19, s20                                 // 00000000723C: 81131413

0000000000007240 <L3>:
	v_sub_nc_u32_e32 v2, 0, v1                                 // 000000007240: 4C040280
	v_ashrrev_i32_e32 v3, 31, v1                               // 000000007244: 3406029F
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000007248: BF870112
	v_max_i32_e32 v2, v2, v1                                   // 00000000724C: 24040302
	v_xor_b32_e32 v3, s13, v3                                  // 000000007250: 3A06060D
	s_wait_alu depctr_sa_sdst(0)                               // 000000007254: BF88FF9E
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000007258: BF870092
	v_mul_hi_u32 v4, v2, s19                                   // 00000000725C: D72D0004 02002702
	v_mul_lo_u32 v6, v4, s12                                   // 000000007264: D72C0006 02001904
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000726C: BF870091
	v_sub_nc_u32_e32 v2, v2, v6                                // 000000007270: 4C040D02
	v_subrev_nc_u32_e32 v6, s12, v2                            // 000000007274: 4E0C040C
	v_cmp_le_u32_e32 vcc_lo, s12, v2                           // 000000007278: 7C96040C
	s_wait_alu depctr_va_vcc(0)                                // 00000000727C: BF88FF9D
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000007280: BF870092
	v_dual_cndmask_b32 v2, v2, v6 :: v_dual_add_nc_u32 v7, 1, v4// 000000007284: CA600D02 02060881
	v_cndmask_b32_e32 v4, v4, v7, vcc_lo                       // 00000000728C: 02080F04
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000007290: BF870112
	v_cmp_le_u32_e32 vcc_lo, s12, v2                           // 000000007294: 7C96040C
	v_add_nc_u32_e32 v6, 1, v4                                 // 000000007298: 4A0C0881
	s_wait_alu depctr_va_vcc(0)                                // 00000000729C: BF88FF9D
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000072A0: BF8700A1
	v_cndmask_b32_e32 v2, v4, v6, vcc_lo                       // 0000000072A4: 02040D04
	v_mul_lo_u32 v6, v1, 12                                    // 0000000072A8: D72C0006 02011901
	v_add_nc_u32_e32 v7, 0xe70, v6                             // 0000000072B0: 4A0E0CFF 00000E70
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000072B8: BF870093
	v_xor_b32_e32 v2, v2, v3                                   // 0000000072BC: 3A040702
	v_sub_nc_u32_e32 v2, v2, v3                                // 0000000072C0: 4C040702
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 0000000072C4: BF870131
	v_mul_lo_u32 v3, v2, s6                                    // 0000000072C8: D72C0003 02000D02
	s_wait_kmcnt 0x0                                           // 0000000072D0: BFC70000
	v_mul_lo_u32 v2, v2, s11                                   // 0000000072D4: D72C0002 02001702
	v_sub_nc_u32_e32 v4, v1, v3                                // 0000000072DC: 4C080701
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)// 0000000072E0: BF870142
	v_add_co_u32 v2, s20, s2, v2                               // 0000000072E4: D7001402 02020402
	s_wait_alu depctr_va_sdst(0)                               // 0000000072EC: BF88F19F
	v_add_co_ci_u32_e64 v3, null, s3, 0, s20                   // 0000000072F0: D5207C03 00510003
	v_add_nc_u32_e32 v1, s17, v1                               // 0000000072F8: 4A020211
	v_mad_co_u64_u32 v[2:3], null, v4, 12, v[2:3]              // 0000000072FC: D6FE7C02 04091904
	s_delay_alu instid0(VALU_DEP_2)                            // 000000007304: BF870002
	v_cmp_le_i32_e32 vcc_lo, s0, v1                            // 000000007308: 7C860200
	s_or_b32 s18, vcc_lo, s18                                  // 00000000730C: 8C12126A
	global_load_b96 v[2:4], v[2:3], off                        // 000000007310: EE05807C 00000002 00000002
	s_wait_loadcnt 0x0                                         // 00000000731C: BFC00000
	ds_store_2addr_b32 v7, v2, v3 offset1:1                    // 000000007320: D8380100 00030207
	ds_store_b32 v6, v4 offset:3704                            // 000000007328: D8340E78 00000406
	s_wait_alu depctr_sa_sdst(0)                               // 000000007330: BF88FF9E
	s_and_not1_b32 exec_lo, exec_lo, s18                       // 000000007334: 917E127E
	s_cbranch_execnz L3                                        // 000000007338: BFA6FFC1

000000000000733c <L4>:
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000733C: 8C7E017E
	s_wait_dscnt 0x0                                           // 000000007340: BFC60000
	s_barrier_signal -1                                        // 000000007344: BE804EC1
	v_add_nc_u32_e32 v3, s15, v0                               // 000000007348: 4A06000F
	v_add_nc_u32_e32 v4, s16, v5                               // 00000000734C: 4A080A10
	v_cmp_gt_u32_e32 vcc_lo, 16, v0                            // 000000007350: 7C980090
	v_cmp_gt_u32_e64 s0, 8, v5                                 // 000000007354: D44C0000 02020A88
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 00000000735C: BF870214
	v_cmp_gt_i32_e64 s1, s4, v3                                // 000000007360: D4440001 02020604
	v_cmp_gt_i32_e64 s2, s5, v4                                // 000000007368: D4440002 02020805
	s_and_b32 s0, vcc_lo, s0                                   // 000000007370: 8B00006A
	s_wait_alu depctr_sa_sdst(0)                               // 000000007374: BF88FF9E
	s_and_b32 s0, s0, s1                                       // 000000007378: 8B000100
	s_wait_alu depctr_sa_sdst(0)                               // 00000000737C: BF88FF9E
	s_and_b32 s0, s0, s2                                       // 000000007380: 8B000200
	s_barrier_wait 0xffff                                      // 000000007384: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000007388: EE0AC07C 00040000 00000000
	s_wait_alu depctr_sa_sdst(0)                               // 000000007394: BF88FF9E
	s_and_saveexec_b32 s1, s0                                  // 000000007398: BE812000
	s_cbranch_execz L25                                        // 00000000739C: BFA50110
	s_cmp_lt_i32 s7, 1                                         // 0000000073A0: BF048107
	s_cbranch_scc1 L8                                          // 0000000073A4: BFA20036
	v_mul_u32_u24_e32 v0, 12, v0                               // 0000000073A8: 1600008C
	v_dual_mov_b32 v8, 0 :: v_dual_mov_b32 v7, 0               // 0000000073AC: CA100080 08060080
	v_mov_b32_e32 v6, 0                                        // 0000000073B4: 7E0C0280
	s_cmp_gt_i32 s6, 0                                         // 0000000073B8: BF028006
	s_mov_b32 s0, 0                                            // 0000000073BC: BE800080
	s_cselect_b32 s1, -1, 0                                    // 0000000073C0: 980180C1
	s_branch L6                                                // 0000000073C4: BFA00004

00000000000073c8 <L5>:
	s_add_co_i32 s0, s0, 1                                     // 0000000073C8: 81008100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000073CC: BF88FF9E
	s_cmp_eq_u32 s0, s7                                        // 0000000073D0: BF060700
	s_cbranch_scc1 L9                                          // 0000000073D4: BFA2002D

00000000000073d8 <L6>:
	s_wait_alu depctr_sa_sdst(0)                               // 0000000073D8: BF88FF9E
	s_and_b32 s2, s1, exec_lo                                  // 0000000073DC: 8B027E01
	s_cselect_b32 s2, 1, 0                                     // 0000000073E0: 98028081
	s_wait_alu depctr_sa_sdst(0)                               // 0000000073E4: BF88FF9E
	s_cmp_lg_u32 s2, 1                                         // 0000000073E8: BF078102
	s_cbranch_scc1 L5                                          // 0000000073EC: BFA2FFF6
	v_add_nc_u32_e32 v1, s0, v5                                // 0000000073F0: 4A020A00
	s_mul_i32 s2, s0, s6                                       // 0000000073F4: 96020600
	s_mov_b32 s3, 0                                            // 0000000073F8: BE830080
	s_wait_alu depctr_sa_sdst(0)                               // 0000000073FC: BF88FF9E
	s_mul_i32 s2, s2, 12                                       // 000000007400: 96028C02
	s_wait_alu depctr_sa_sdst(0)                               // 000000007404: BF88FF9E
	s_addk_co_i32 s2, 0xe70                                    // 000000007408: B7820E70
	v_mul_lo_u32 v1, v1, s14                                   // 00000000740C: D72C0001 02001D01
	s_delay_alu instid0(VALU_DEP_1)                            // 000000007414: BF870001
	v_mad_co_u64_u32 v[1:2], null, v1, 12, v[0:1]              // 000000007418: D6FE7C01 04011901

0000000000007420 <L7>:
	s_mul_i32 s4, s3, 12                                       // 000000007420: 96048C03
	s_add_co_i32 s3, s3, 1                                     // 000000007424: 81038103
	s_wait_alu depctr_sa_sdst(0)                               // 000000007428: BF88FF9E
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000742C: BF870001
	v_add_nc_u32_e32 v2, s4, v1                                // 000000007430: 4A040204
	s_add_co_i32 s4, s2, s4                                    // 000000007434: 81040402
	s_cmp_eq_u32 s3, s6                                        // 000000007438: BF060603
	s_wait_alu depctr_sa_sdst(0)                               // 00000000743C: BF88FF9E
	v_mov_b32_e32 v13, s4                                      // 000000007440: 7E1A0204
	ds_load_2addr_b32 v[9:10], v2 offset1:1                    // 000000007444: D8DC0100 09000002
	ds_load_b32 v2, v2 offset:8                                // 00000000744C: D8D80008 02000002
	ds_load_2addr_b32 v[11:12], v13 offset1:1                  // 000000007454: D8DC0100 0B00000D
	ds_load_b32 v13, v13 offset:8                              // 00000000745C: D8D80008 0D00000D
	s_wait_dscnt 0x1                                           // 000000007464: BFC60001
	v_dual_fmac_f32 v8, v9, v11 :: v_dual_fmac_f32 v7, v10, v12// 000000007468: C8001709 0806190A
	s_wait_dscnt 0x0                                           // 000000007470: BFC60000
	v_fmac_f32_e32 v6, v2, v13                                 // 000000007474: 560C1B02
	s_cbranch_scc0 L7                                          // 000000007478: BFA1FFE9
	s_branch L5                                                // 00000000747C: BFA0FFD2

0000000000007480 <L8>:
	v_dual_mov_b32 v8, 0 :: v_dual_mov_b32 v7, 0               // 000000007480: CA100080 08060080
	v_mov_b32_e32 v6, 0                                        // 000000007488: 7E0C0280

000000000000748c <L9>:
	v_mov_b16_e32 v0.l, 0xff                                   // 00000000748C: 7E0038FF 000000FF
	s_mov_b32 s2, exec_lo                                      // 000000007494: BE82007E
	s_delay_alu instid0(VALU_DEP_3)                            // 000000007498: BF870003
	v_cmpx_nlt_f32_e32 0x437f0000, v8                          // 00000000749C: 7D3C10FF 437F0000
	s_cbranch_execz L14                                        // 0000000074A4: BFA50036
	v_mov_b16_e32 v0.l, 0                                      // 0000000074A8: 7E003880
	s_mov_b32 s3, exec_lo                                      // 0000000074AC: BE83007E
	v_cmpx_ngt_f32_e32 0, v8                                   // 0000000074B0: 7D361080
	s_cbranch_execz L13                                        // 0000000074B4: BFA50030
	s_mov_b32 s4, exec_lo                                      // 0000000074B8: BE84007E
	v_cmpx_o_f32_e32 v8, v8                                    // 0000000074BC: 7D2E1108
	s_cbranch_execz L12                                        // 0000000074C0: BFA5002A
	v_add_f32_e32 v0, v8, v8                                   // 0000000074C4: 06001108
	v_cmp_nlg_f32_e32 vcc_lo, 0, v8                            // 0000000074C8: 7C341080
	v_cmp_neq_f32_e64 s1, 0, v8                                // 0000000074CC: D41D0001 02021080
	s_delay_alu instid0(VALU_DEP_3)                            // 0000000074D4: BF870003
	v_cmp_neq_f32_e64 s0, v0, v8                               // 0000000074D8: D41D0000 02021100
	s_or_b32 s0, vcc_lo, s0                                    // 0000000074E0: 8C00006A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000074E4: BF88FF9E
	s_and_b32 s1, s1, s0                                       // 0000000074E8: 8B010001
	s_wait_alu depctr_sa_sdst(0)                               // 0000000074EC: BF88FF9E
	s_and_saveexec_b32 s0, s1                                  // 0000000074F0: BE802001
	s_cbranch_execz L11                                        // 0000000074F4: BFA5001B
	v_cvt_i32_f32_e32 v1, v8                                   // 0000000074F8: 7E021108
	s_mov_b32 s1, exec_lo                                      // 0000000074FC: BE81007E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000007500: BF870091
	v_cvt_f32_i32_e32 v0, v1                                   // 000000007504: 7E000B01
	v_sub_f32_e32 v2, v8, v0                                   // 000000007508: 08040108
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000750C: BF870001
	v_cmpx_ngt_f32_e32 0.5, v2                                 // 000000007510: 7D3604F0
	s_cbranch_execz L10                                        // 000000007514: BFA50010
	s_mov_b32 s5, exec_lo                                      // 000000007518: BE85007E
	v_cmpx_nlt_f32_e32 0.5, v2                                 // 00000000751C: 7D3C04F0
	s_wait_alu depctr_sa_sdst(0)                               // 000000007520: BF88FF9E
	s_xor_b32 s5, exec_lo, s5                                  // 000000007524: 8D05057E
	v_dual_add_f32 v2, 1.0, v0 :: v_dual_and_b32 v1, 1, v1     // 000000007528: C92400F2 02000281
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000007530: BF870121
	v_cmp_eq_u32_e32 vcc_lo, 0, v1                             // 000000007534: 7C940280
	s_wait_alu depctr_va_vcc(0)                                // 000000007538: BF88FF9D
	v_cndmask_b32_e32 v0, v2, v0, vcc_lo                       // 00000000753C: 02000102
	s_wait_alu depctr_sa_sdst(0)                               // 000000007540: BF88FF9E
	s_and_not1_saveexec_b32 s5, s5                             // 000000007544: BE853005
	s_delay_alu instid0(VALU_DEP_1)                            // 000000007548: BF870001
	v_add_f32_e32 v0, 1.0, v0                                  // 00000000754C: 060000F2
	s_wait_alu depctr_sa_sdst(0)                               // 000000007550: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s5                              // 000000007554: 8C7E057E

0000000000007558 <L10>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000007558: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000755C: 8C7E017E
	v_mov_b32_e32 v8, v0                                       // 000000007560: 7E100300

0000000000007564 <L11>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000007564: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s0                              // 000000007568: 8C7E007E

000000000000756c <L12>:
	s_wait_alu depctr_sa_sdst(0)                               // 00000000756C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s4                              // 000000007570: 8C7E047E
	v_cvt_i32_f32_e32 v0, v8                                   // 000000007574: 7E001108

0000000000007578 <L13>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000007578: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s3                              // 00000000757C: 8C7E037E

0000000000007580 <L14>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000007580: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s2                              // 000000007584: 8C7E027E
	v_mov_b32_e32 v1, 0xff00                                   // 000000007588: 7E0202FF 0000FF00
	s_mov_b32 s2, exec_lo                                      // 000000007590: BE82007E
	v_cmpx_nlt_f32_e32 0x437f0000, v7                          // 000000007594: 7D3C0EFF 437F0000
	s_cbranch_execz L19                                        // 00000000759C: BFA50039
	v_mov_b32_e32 v1, 0                                        // 0000000075A0: 7E020280
	s_mov_b32 s3, exec_lo                                      // 0000000075A4: BE83007E
	v_cmpx_ngt_f32_e32 0, v7                                   // 0000000075A8: 7D360E80
	s_cbranch_execz L18                                        // 0000000075AC: BFA50033
	s_mov_b32 s4, exec_lo                                      // 0000000075B0: BE84007E
	v_cmpx_o_f32_e32 v7, v7                                    // 0000000075B4: 7D2E0F07
	s_cbranch_execz L17                                        // 0000000075B8: BFA5002A
	v_add_f32_e32 v1, v7, v7                                   // 0000000075BC: 06020F07
	v_cmp_nlg_f32_e32 vcc_lo, 0, v7                            // 0000000075C0: 7C340E80
	v_cmp_neq_f32_e64 s1, 0, v7                                // 0000000075C4: D41D0001 02020E80
	s_delay_alu instid0(VALU_DEP_3)                            // 0000000075CC: BF870003
	v_cmp_neq_f32_e64 s0, v1, v7                               // 0000000075D0: D41D0000 02020F01
	s_or_b32 s0, vcc_lo, s0                                    // 0000000075D8: 8C00006A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000075DC: BF88FF9E
	s_and_b32 s1, s1, s0                                       // 0000000075E0: 8B010001
	s_wait_alu depctr_sa_sdst(0)                               // 0000000075E4: BF88FF9E
	s_and_saveexec_b32 s0, s1                                  // 0000000075E8: BE802001
	s_cbranch_execz L16                                        // 0000000075EC: BFA5001B
	v_cvt_i32_f32_e32 v2, v7                                   // 0000000075F0: 7E041107
	s_mov_b32 s1, exec_lo                                      // 0000000075F4: BE81007E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000075F8: BF870091
	v_cvt_f32_i32_e32 v1, v2                                   // 0000000075FC: 7E020B02
	v_sub_f32_e32 v5, v7, v1                                   // 000000007600: 080A0307
	s_delay_alu instid0(VALU_DEP_1)                            // 000000007604: BF870001
	v_cmpx_ngt_f32_e32 0.5, v5                                 // 000000007608: 7D360AF0
	s_cbranch_execz L15                                        // 00000000760C: BFA50010
	s_mov_b32 s5, exec_lo                                      // 000000007610: BE85007E
	v_cmpx_nlt_f32_e32 0.5, v5                                 // 000000007614: 7D3C0AF0
	s_wait_alu depctr_sa_sdst(0)                               // 000000007618: BF88FF9E
	s_xor_b32 s5, exec_lo, s5                                  // 00000000761C: 8D05057E
	v_dual_add_f32 v5, 1.0, v1 :: v_dual_and_b32 v2, 1, v2     // 000000007620: C92402F2 05020481
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000007628: BF870121
	v_cmp_eq_u32_e32 vcc_lo, 0, v2                             // 00000000762C: 7C940480
	s_wait_alu depctr_va_vcc(0)                                // 000000007630: BF88FF9D
	v_cndmask_b32_e32 v1, v5, v1, vcc_lo                       // 000000007634: 02020305
	s_wait_alu depctr_sa_sdst(0)                               // 000000007638: BF88FF9E
	s_and_not1_saveexec_b32 s5, s5                             // 00000000763C: BE853005
	s_delay_alu instid0(VALU_DEP_1)                            // 000000007640: BF870001
	v_add_f32_e32 v1, 1.0, v1                                  // 000000007644: 060202F2
	s_wait_alu depctr_sa_sdst(0)                               // 000000007648: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s5                              // 00000000764C: 8C7E057E

0000000000007650 <L15>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000007650: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s1                              // 000000007654: 8C7E017E
	v_mov_b32_e32 v7, v1                                       // 000000007658: 7E0E0301

000000000000765c <L16>:
	s_wait_alu depctr_sa_sdst(0)                               // 00000000765C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s0                              // 000000007660: 8C7E007E

0000000000007664 <L17>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000007664: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s4                              // 000000007668: 8C7E047E
	v_cvt_i32_f32_e32 v1, v7                                   // 00000000766C: 7E021107
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000007670: BF870091
	v_cvt_u32_u16_e32 v1, v1.l                                 // 000000007674: 7E02D701
	v_lshlrev_b32_e32 v1, 8, v1                                // 000000007678: 30020288

000000000000767c <L18>:
	s_wait_alu depctr_sa_sdst(0)                               // 00000000767C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s3                              // 000000007680: 8C7E037E

0000000000007684 <L19>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000007684: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s2                              // 000000007688: 8C7E027E
	v_mov_b32_e32 v2, 0xff0000                                 // 00000000768C: 7E0402FF 00FF0000
	s_mov_b32 s2, exec_lo                                      // 000000007694: BE82007E
	v_cmpx_nlt_f32_e32 0x437f0000, v6                          // 000000007698: 7D3C0CFF 437F0000
	s_cbranch_execz L24                                        // 0000000076A0: BFA50038
	v_mov_b32_e32 v2, 0                                        // 0000000076A4: 7E040280
	s_mov_b32 s3, exec_lo                                      // 0000000076A8: BE83007E
	v_cmpx_ngt_f32_e32 0, v6                                   // 0000000076AC: 7D360C80
	s_cbranch_execz L23                                        // 0000000076B0: BFA50032
	s_mov_b32 s4, exec_lo                                      // 0000000076B4: BE84007E
	v_cmpx_o_f32_e32 v6, v6                                    // 0000000076B8: 7D2E0D06
	s_cbranch_execz L22                                        // 0000000076BC: BFA5002A
	v_add_f32_e32 v2, v6, v6                                   // 0000000076C0: 06040D06
	v_cmp_nlg_f32_e32 vcc_lo, 0, v6                            // 0000000076C4: 7C340C80
	v_cmp_neq_f32_e64 s1, 0, v6                                // 0000000076C8: D41D0001 02020C80
	s_delay_alu instid0(VALU_DEP_3)                            // 0000000076D0: BF870003
	v_cmp_neq_f32_e64 s0, v2, v6                               // 0000000076D4: D41D0000 02020D02
	s_or_b32 s0, vcc_lo, s0                                    // 0000000076DC: 8C00006A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000076E0: BF88FF9E
	s_and_b32 s1, s1, s0                                       // 0000000076E4: 8B010001
	s_wait_alu depctr_sa_sdst(0)                               // 0000000076E8: BF88FF9E
	s_and_saveexec_b32 s0, s1                                  // 0000000076EC: BE802001
	s_cbranch_execz L21                                        // 0000000076F0: BFA5001B
	v_cvt_i32_f32_e32 v5, v6                                   // 0000000076F4: 7E0A1106
	s_mov_b32 s1, exec_lo                                      // 0000000076F8: BE81007E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000076FC: BF870091
	v_cvt_f32_i32_e32 v2, v5                                   // 000000007700: 7E040B05
	v_sub_f32_e32 v6, v6, v2                                   // 000000007704: 080C0506
	s_delay_alu instid0(VALU_DEP_1)                            // 000000007708: BF870001
	v_cmpx_ngt_f32_e32 0.5, v6                                 // 00000000770C: 7D360CF0
	s_cbranch_execz L20                                        // 000000007710: BFA50010
	s_mov_b32 s5, exec_lo                                      // 000000007714: BE85007E
	v_cmpx_nlt_f32_e32 0.5, v6                                 // 000000007718: 7D3C0CF0
	s_wait_alu depctr_sa_sdst(0)                               // 00000000771C: BF88FF9E
	s_xor_b32 s5, exec_lo, s5                                  // 000000007720: 8D05057E
	v_dual_add_f32 v6, 1.0, v2 :: v_dual_and_b32 v5, 1, v5     // 000000007724: C92404F2 06040A81
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 00000000772C: BF870121
	v_cmp_eq_u32_e32 vcc_lo, 0, v5                             // 000000007730: 7C940A80
	s_wait_alu depctr_va_vcc(0)                                // 000000007734: BF88FF9D
	v_cndmask_b32_e32 v2, v6, v2, vcc_lo                       // 000000007738: 02040506
	s_wait_alu depctr_sa_sdst(0)                               // 00000000773C: BF88FF9E
	s_and_not1_saveexec_b32 s5, s5                             // 000000007740: BE853005
	s_delay_alu instid0(VALU_DEP_1)                            // 000000007744: BF870001
	v_add_f32_e32 v2, 1.0, v2                                  // 000000007748: 060404F2
	s_wait_alu depctr_sa_sdst(0)                               // 00000000774C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s5                              // 000000007750: 8C7E057E

0000000000007754 <L20>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000007754: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s1                              // 000000007758: 8C7E017E
	v_mov_b32_e32 v6, v2                                       // 00000000775C: 7E0C0302

0000000000007760 <L21>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000007760: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s0                              // 000000007764: 8C7E007E

0000000000007768 <L22>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000007768: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s4                              // 00000000776C: 8C7E047E
	v_cvt_i32_f32_e32 v2, v6                                   // 000000007770: 7E041106
	s_delay_alu instid0(VALU_DEP_1)                            // 000000007774: BF870001
	v_lshlrev_b32_e32 v2, 16, v2                               // 000000007778: 30040490

000000000000777c <L23>:
	s_wait_alu depctr_sa_sdst(0)                               // 00000000777C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s3                              // 000000007780: 8C7E037E

0000000000007784 <L24>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000007784: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s2                              // 000000007788: 8C7E027E
	s_wait_kmcnt 0x0                                           // 00000000778C: BFC70000
	v_mul_lo_u32 v4, s10, v4                                   // 000000007790: D72C0004 0202080A
	v_and_or_b32 v0, 0xff00, v1, v0                            // 000000007798: D6570000 040202FF 0000FF00
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 0000000077A4: BF8700B2
	v_add_co_u32 v4, s0, s8, v4                                // 0000000077A8: D7000004 02020808
	s_wait_alu depctr_va_sdst(0)                               // 0000000077B0: BF88F19F
	v_add_co_ci_u32_e64 v5, null, s9, 0, s0                    // 0000000077B4: D5207C05 00010009
	v_mad_co_i64_i32 v[3:4], null, v3, 3, v[4:5]               // 0000000077BC: D6FF7C03 04110703
	s_clause 0x1                                               // 0000000077C4: BF850001
	global_store_d16_hi_b8 v[3:4], v2, off offset:2            // 0000000077C8: EE09007C 01000000 00000203
	global_store_b16 v[3:4], v0, off                           // 0000000077D4: EE06407C 00000000 00000003

00000000000077e0 <L25>:
	s_endpgm                                                   // 0000000077E0: BFB00000
		...

0000000000007800 <_ZN2fk21linearFilterDPPKernelILNS_7ParArchE3ENS_22LinearFilterDPPDetailsINS_6Float3ELi16ELi8ELi7ELi7EEENS_5TupleIJNS_29ReadBackInstantiableOperationINS_14ResizeCompleteILNS_11AspectRatioE1ENS_28TernaryInstantiableOperationINS_19InterpolateCompleteILNS_17InterpolationTypeE1ENS_25ReadInstantiableOperationINS_13PerThreadReadILNS_2NDE2ENS_6Uchar3EEEEEEEEELNS_14ResizeSamplingE1EEEEENSC_INSD_ILSE_2ES3_EEEEEEENS_26UnaryInstantiableOperationINS_3MulIS3_S3_S3_NS_9UnaryTypeEEEEENSQ_INS_3AddIS3_S3_S3_SS_EEEENS_26WriteInstantiableOperationINS_15FusedOperation_IvJNSQ_INS_12SaturateCastIS3_SF_EEEENSY_INS_14PerThreadWriteILSE_2ESF_EEEEEEEEEEEvT0_T1_T2_T3_T4_>:
	s_clause 0x3                                               // 000000007800: BF850003
	s_load_b32 s10, s[0:1], 0x8c                               // 000000007804: F4000280 F800008C
	s_load_b128 s[4:7], s[0:1], 0x0                            // 00000000780C: F4004100 F8000000
	s_load_b64 s[8:9], s[0:1], 0x68                            // 000000007814: F4002200 F8000068
	s_load_b64 s[2:3], s[0:1], 0x48                            // 00000000781C: F4002080 F8000048
	v_bfe_u32 v4, v0, 10, 10                                   // 000000007824: D6100004 02291500
	v_and_b32_e32 v0, 0x3ff, v0                                // 00000000782C: 360000FF 000003FF
	s_lshl_b32 s15, ttmp9, 4                                   // 000000007834: 840F8475
	s_lshl_b32 s16, ttmp7, 3                                   // 000000007838: 84108373
	s_mov_b32 s19, exec_lo                                     // 00000000783C: BE93007E
	s_wait_kmcnt 0x0                                           // 000000007840: BFC70000
	s_and_b32 s11, s10, 0xffff                                 // 000000007844: 8B0BFF0A 0000FFFF
	s_add_co_i32 s14, s6, 15                                   // 00000000784C: 810E8F06
	v_mad_u32_u24 v1, v4, s11, v0                              // 000000007850: D60B0001 04001704
	s_add_co_i32 s18, s7, 7                                    // 000000007858: 81128707
	s_lshr_b32 s17, s10, 16                                    // 00000000785C: 8511900A
	s_mul_i32 s18, s18, s14                                    // 000000007860: 96120E12
	s_mul_i32 s17, s17, s11                                    // 000000007864: 96110B11
	v_cmpx_gt_i32_e32 s18, v1                                  // 000000007868: 7D880212
	s_cbranch_execz L27                                        // 00000000786C: BFA500DB
	s_abs_i32 s20, s14                                         // 000000007870: BE94150E
	s_add_nc_u64 s[22:23], s[0:1], 24                          // 000000007874: A9969800
	s_cvt_f32_u32 s10, s20                                     // 000000007878: BE8A6514
	s_clause 0x1                                               // 00000000787C: BF850001
	s_load_b64 s[28:29], s[0:1], 0x10                          // 000000007880: F4002700 F8000010
	s_load_b64 s[30:31], s[0:1], 0x38                          // 000000007888: F4002780 F8000038
	v_mul_lo_u32 v2, v1, 12                                    // 000000007890: D72C0002 02011901
	v_s_rcp_f32 s26, s10                                       // 000000007898: D684001A 0201000A
	s_clause 0x2                                               // 0000000078A0: BF850002
	s_load_b64 s[10:11], s[22:23], 0x8                         // 0000000078A4: F400228B F8000008
	s_load_b64 s[12:13], s[22:23], 0x18                        // 0000000078AC: F400230B F8000018
	s_load_b32 s21, s[22:23], 0x28                             // 0000000078B4: F400054B F8000028
	s_sub_co_i32 s34, 0, s20                                   // 0000000078BC: 81A21480
	v_mov_b32_e32 v3, v1                                       // 0000000078C0: 7E060301
	s_add_co_i32 s22, s4, -1                                   // 0000000078C4: 8116C104
	s_add_co_i32 s23, s5, -1                                   // 0000000078C8: 8117C105
	s_ashr_i32 s24, s14, 31                                    // 0000000078CC: 86189F0E
	s_mov_b32 s25, 0                                           // 0000000078D0: BE990080
	s_mul_f32 s27, s26, 0x4f7ffffe                             // 0000000078D4: A21BFF1A 4F7FFFFE
	s_sub_co_i32 s26, 0, s14                                   // 0000000078DC: 819A0E80
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(SALU_CYCLE_3)// 0000000078E0: BF87059A
	s_cvt_u32_f32 s33, s27                                     // 0000000078E4: BEA1671B
	s_mul_i32 s34, s34, s33                                    // 0000000078E8: 96222122
	s_wait_kmcnt 0x0                                           // 0000000078EC: BFC70000
	s_sub_co_i32 s27, s15, s28                                 // 0000000078F0: 819B1C0F
	s_mul_hi_u32 s34, s33, s34                                 // 0000000078F4: 96A22221
	s_sub_co_i32 s28, s16, s29                                 // 0000000078F8: 819C1D10
	s_add_co_i32 s29, s30, -1                                  // 0000000078FC: 811DC11E
	s_add_co_i32 s30, s31, -1                                  // 000000007900: 811EC11F
	s_add_co_i32 s31, s33, s34                                 // 000000007904: 811F2221
	s_mul_i32 s33, s17, 12                                     // 000000007908: 96218C11

000000000000790c <L26>:
	v_sub_nc_u32_e32 v5, 0, v3                                 // 00000000790C: 4C0A0680
	v_ashrrev_i32_e32 v6, 31, v3                               // 000000007910: 340C069F
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000007914: BF870112
	v_max_i32_e32 v5, v5, v3                                   // 000000007918: 240A0705
	v_xor_b32_e32 v6, s24, v6                                  // 00000000791C: 3A0C0C18
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000007920: BF870092
	v_mul_hi_u32 v7, v5, s31                                   // 000000007924: D72D0007 02003F05
	v_mul_lo_u32 v8, v7, s20                                   // 00000000792C: D72C0008 02002907
	v_add_nc_u32_e32 v9, 1, v7                                 // 000000007934: 4A120E81
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000007938: BF870092
	v_sub_nc_u32_e32 v5, v5, v8                                // 00000000793C: 4C0A1105
	v_subrev_nc_u32_e32 v8, s20, v5                            // 000000007940: 4E100A14
	v_cmp_le_u32_e32 vcc_lo, s20, v5                           // 000000007944: 7C960A14
	s_wait_alu depctr_va_vcc(0)                                // 000000007948: BF88FF9D
	v_cndmask_b32_e32 v7, v7, v9, vcc_lo                       // 00000000794C: 020E1307
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000007950: BF870091
	v_dual_cndmask_b32 v5, v5, v8 :: v_dual_add_nc_u32 v8, 1, v7// 000000007954: CA601105 05080E81
	v_cmp_le_u32_e32 vcc_lo, s20, v5                           // 00000000795C: 7C960A14
	s_wait_alu depctr_va_vcc(0)                                // 000000007960: BF88FF9D
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000007964: BF870092
	v_cndmask_b32_e32 v5, v7, v8, vcc_lo                       // 000000007968: 020A1107
	v_xor_b32_e32 v5, v5, v6                                   // 00000000796C: 3A0A0D05
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000007970: BF870091
	v_sub_nc_u32_e32 v5, v5, v6                                // 000000007974: 4C0A0D05
	v_mul_lo_u32 v6, s26, v5                                   // 000000007978: D72C0006 02020A1A
	v_add_nc_u32_e32 v5, s28, v5                               // 000000007980: 4A0A0A1C
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 000000007984: BF870221
	v_min_i32_e32 v7, s23, v5                                  // 000000007988: 220E0A17
	v_cmp_lt_i32_e32 vcc_lo, -1, v5                            // 00000000798C: 7C820AC1
	v_add3_u32 v5, s27, v3, v6                                 // 000000007990: D6550005 041A061B
	s_wait_alu depctr_va_vcc(0)                                // 000000007998: BF88FF9D
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000799C: BF870113
	v_cndmask_b32_e32 v6, 0, v7, vcc_lo                        // 0000000079A0: 020C0E80
	v_min_i32_e32 v7, s22, v5                                  // 0000000079A4: 220E0A16
	v_cmp_lt_i32_e32 vcc_lo, -1, v5                            // 0000000079A8: 7C820AC1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000079AC: BF8700A3
	v_cvt_f32_i32_e32 v5, v6                                   // 0000000079B0: 7E0A0B06
	s_wait_alu depctr_va_vcc(0)                                // 0000000079B4: BF88FF9D
	v_dual_cndmask_b32 v6, 0, v7 :: v_dual_add_f32 v5, 0.5, v5 // 0000000079B8: CA480E80 06040AF0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000079C0: BF870111
	v_cvt_f32_i32_e32 v6, v6                                   // 0000000079C4: 7E0C0B06
	v_fma_f32 v5, s11, v5, -0.5                                // 0000000079C8: D6130005 03C60A0B
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000079D0: BF870112
	v_add_f32_e32 v6, 0.5, v6                                  // 0000000079D4: 060C0CF0
	v_cmp_le_f32_e32 vcc_lo, 0, v5                             // 0000000079D8: 7C260A80
	s_wait_alu depctr_va_vcc(0)                                // 0000000079DC: BF88FF9D
	v_cndmask_b32_e32 v13, 0, v5, vcc_lo                       // 0000000079E0: 021A0A80
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000079E4: BF870093
	v_fma_f32 v5, s10, v6, -0.5                                // 0000000079E8: D6130005 03C60C0A
	v_cmp_le_f32_e32 vcc_lo, 0, v5                             // 0000000079F0: 7C260A80
	s_wait_alu depctr_va_vcc(0)                                // 0000000079F4: BF88FF9D
	v_cndmask_b32_e32 v14, 0, v5, vcc_lo                       // 0000000079F8: 021C0A80
	v_floor_f32_e32 v6, v13                                    // 0000000079FC: 7E0C490D
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000007A00: BF870112
	v_floor_f32_e32 v5, v14                                    // 000000007A04: 7E0A490E
	v_cvt_i32_f32_e32 v15, v6                                  // 000000007A08: 7E1E1106
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000007A0C: BF870112
	v_cvt_i32_f32_e32 v17, v5                                  // 000000007A10: 7E221105
	v_add_nc_u32_e32 v16, 1, v15                               // 000000007A14: 4A201E81
	v_mul_lo_u32 v6, s21, v15                                  // 000000007A18: D72C0006 02021E15
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000007A20: BF870193
	v_add_nc_u32_e32 v18, 1, v17                               // 000000007A24: 4A242281
	v_min_i32_e32 v7, s30, v16                                 // 000000007A28: 220E201E
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000007A2C: BF870113
	v_add_co_u32 v5, s34, s12, v6                              // 000000007A30: D7002205 02020C0C
	v_mul_lo_u32 v9, v7, s21                                   // 000000007A38: D72C0009 02002B07
	s_wait_alu depctr_va_sdst(0)                               // 000000007A40: BF88F19F
	v_add_co_ci_u32_e64 v6, null, s13, 0, s34                  // 000000007A44: D5207C06 0089000D
	v_min_i32_e32 v11, s29, v18                                // 000000007A4C: 2216241D
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000007A50: BF870212
	v_mad_co_i64_i32 v[7:8], null, v17, 3, v[5:6]              // 000000007A54: D6FF7C07 04150711
	v_add_co_u32 v9, s34, s12, v9                              // 000000007A5C: D7002209 0202120C
	s_wait_alu depctr_va_sdst(0)                               // 000000007A64: BF88F19F
	v_add_co_ci_u32_e64 v10, null, s13, 0, s34                 // 000000007A68: D5207C0A 0089000D
	v_mad_co_i64_i32 v[5:6], null, v11, 3, v[5:6]              // 000000007A70: D6FF7C05 0415070B
	global_load_u8 v19, v[7:8], off offset:2                   // 000000007A78: EE04007C 00000013 00000207
	v_mad_co_i64_i32 v[11:12], null, v11, 3, v[9:10]           // 000000007A84: D6FF7C0B 0425070B
	v_mad_co_i64_i32 v[9:10], null, v17, 3, v[9:10]            // 000000007A8C: D6FF7C09 04250711
	s_clause 0x6                                               // 000000007A94: BF850006
	global_load_u8 v20, v[5:6], off offset:2                   // 000000007A98: EE04007C 00000014 00000205
	global_load_u16 v21, v[11:12], off                         // 000000007AA4: EE04807C 00000015 0000000B
	global_load_u16 v22, v[9:10], off                          // 000000007AB0: EE04807C 00000016 00000009
	global_load_u16 v5, v[5:6], off                            // 000000007ABC: EE04807C 00000005 00000005
	global_load_u16 v6, v[7:8], off                            // 000000007AC8: EE04807C 00000006 00000007
	global_load_u8 v7, v[9:10], off offset:2                   // 000000007AD4: EE04007C 00000007 00000209
	global_load_u8 v8, v[11:12], off offset:2                  // 000000007AE0: EE04007C 00000008 0000020B
	v_cvt_f32_i32_e32 v11, v17                                 // 000000007AEC: 7E160B11
	v_cvt_f32_i32_e32 v10, v16                                 // 000000007AF0: 7E140B10
	v_add_nc_u32_e32 v3, s17, v3                               // 000000007AF4: 4A060611
	v_cvt_f32_i32_e32 v9, v15                                  // 000000007AF8: 7E120B0F
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_3)// 000000007AFC: BF8701C4
	v_sub_f32_e32 v11, v14, v11                                // 000000007B00: 0816170E
	v_cvt_f32_i32_e32 v12, v18                                 // 000000007B04: 7E180B12
	v_sub_f32_e32 v10, v10, v13                                // 000000007B08: 08141B0A
	v_cmp_le_i32_e32 vcc_lo, s18, v3                           // 000000007B0C: 7C860612
	v_dual_sub_f32 v12, v12, v14 :: v_dual_sub_f32 v9, v13, v9 // 000000007B10: C94A1D0C 0C08130D
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000007B18: BF870123
	v_mul_f32_e32 v13, v11, v10                                // 000000007B1C: 101A150B
	s_or_b32 s25, vcc_lo, s25                                  // 000000007B20: 8C19196A
	v_mul_f32_e32 v10, v10, v12                                // 000000007B24: 1014190A
	s_wait_loadcnt 0x6                                         // 000000007B28: BFC00006
	v_cvt_f32_ubyte0_e32 v14, v20                              // 000000007B2C: 7E1C2314
	s_wait_loadcnt 0x5                                         // 000000007B30: BFC00005
	v_cvt_f32_ubyte0_e32 v18, v21                              // 000000007B34: 7E242315
	s_wait_loadcnt 0x4                                         // 000000007B38: BFC00004
	v_cvt_f32_ubyte1_e32 v17, v22                              // 000000007B3C: 7E222516
	s_wait_loadcnt 0x3                                         // 000000007B40: BFC00003
	v_cvt_f32_ubyte0_e32 v16, v5                               // 000000007B44: 7E202305
	s_wait_loadcnt 0x2                                         // 000000007B48: BFC00002
	v_cvt_f32_ubyte0_e32 v15, v6                               // 000000007B4C: 7E1E2306
	v_cvt_f32_ubyte1_e32 v6, v6                                // 000000007B50: 7E0C2506
	v_mul_f32_e32 v11, v9, v11                                 // 000000007B54: 10161709
	v_cvt_f32_ubyte1_e32 v5, v5                                // 000000007B58: 7E0A2505
	s_wait_loadcnt 0x1                                         // 000000007B5C: BFC00001
	v_cvt_f32_ubyte0_e32 v7, v7                                // 000000007B60: 7E0E2307
	s_wait_loadcnt 0x0                                         // 000000007B64: BFC00000
	v_cvt_f32_ubyte0_e32 v8, v8                                // 000000007B68: 7E102308
	v_dual_mul_f32 v6, v10, v6 :: v_dual_mul_f32 v9, v9, v12   // 000000007B6C: C8C60D0A 06081909
	v_cvt_f32_ubyte0_e32 v12, v19                              // 000000007B74: 7E182313
	v_mul_f32_e32 v15, v10, v15                                // 000000007B78: 101E1F0A
	v_cvt_f32_ubyte1_e32 v19, v21                              // 000000007B7C: 7E262515
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000007B80: BF870193
	v_mul_f32_e32 v12, v10, v12                                // 000000007B84: 1018190A
	v_dual_fmac_f32 v15, v13, v16 :: v_dual_fmac_f32 v6, v13, v5// 000000007B88: C800210D 0F060B0D
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)// 000000007B90: BF8701A2
	v_fmac_f32_e32 v12, v13, v14                               // 000000007B94: 56181D0D
	v_cvt_f32_ubyte0_e32 v14, v22                              // 000000007B98: 7E1C2316
	v_fmac_f32_e32 v6, v9, v17                                 // 000000007B9C: 560C2309
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000007BA0: BF870191
	v_fmac_f32_e32 v6, v11, v19                                // 000000007BA4: 560C270B
	v_dual_fmac_f32 v12, v9, v7 :: v_dual_fmac_f32 v15, v9, v14// 000000007BA8: C8000F09 0C0E1D09
	s_delay_alu instid0(VALU_DEP_1)                            // 000000007BB0: BF870001
	v_dual_fmac_f32 v12, v11, v8 :: v_dual_fmac_f32 v15, v11, v18// 000000007BB4: C800110B 0C0E250B
	ds_store_2addr_b32 v2, v15, v6 offset1:1                   // 000000007BBC: D8380100 00060F02
	ds_store_b32 v2, v12 offset:8                              // 000000007BC4: D8340008 00000C02
	v_add_nc_u32_e32 v2, s33, v2                               // 000000007BCC: 4A040421
	s_wait_alu depctr_sa_sdst(0)                               // 000000007BD0: BF88FF9E
	s_and_not1_b32 exec_lo, exec_lo, s25                       // 000000007BD4: 917E197E
	s_cbranch_execnz L26                                       // 000000007BD8: BFA6FF4C

0000000000007bdc <L27>:
	s_or_b32 exec_lo, exec_lo, s19                             // 000000007BDC: 8C7E137E
	s_clause 0x1                                               // 000000007BE0: BF850001
	s_load_b32 s11, s[0:1], 0x58                               // 000000007BE4: F40002C0 F8000058
	s_load_b32 s10, s[0:1], 0x78                               // 000000007BEC: F4000280 F8000078
	s_mul_i32 s0, s7, s6                                       // 000000007BF4: 96000607
	s_mov_b32 s1, exec_lo                                      // 000000007BF8: BE81007E
	v_cmpx_gt_i32_e32 s0, v1                                   // 000000007BFC: 7D880200
	s_cbranch_execz L29                                        // 000000007C00: BFA50054
	s_abs_i32 s12, s6                                          // 000000007C04: BE8C1506
	s_wait_alu depctr_sa_sdst(0)                               // 000000007C08: BF88FF9E
	s_cvt_f32_u32 s13, s12                                     // 000000007C0C: BE8D650C
	s_wait_alu depctr_sa_sdst(0)                               // 000000007C10: BF88FF9E
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)// 000000007C14: BF87029A
	v_s_rcp_f32 s13, s13                                       // 000000007C18: D684000D 0201000D
	s_mul_f32 s13, s13, 0x4f7ffffe                             // 000000007C20: A20DFF0D 4F7FFFFE
	s_wait_alu depctr_sa_sdst(0)                               // 000000007C28: BF88FF9E
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000007C2C: BF8704BA
	s_cvt_u32_f32 s19, s13                                     // 000000007C30: BE93670D
	s_sub_co_i32 s13, 0, s12                                   // 000000007C34: 818D0C80
	s_wait_alu depctr_sa_sdst(0)                               // 000000007C38: BF88FF9E
	s_mul_i32 s18, s13, s19                                    // 000000007C3C: 9612130D
	s_ashr_i32 s13, s6, 31                                     // 000000007C40: 860D9F06
	s_wait_alu depctr_sa_sdst(0)                               // 000000007C44: BF88FF9E
	s_mul_hi_u32 s20, s19, s18                                 // 000000007C48: 96941213
	s_mov_b32 s18, 0                                           // 000000007C4C: BE920080
	s_wait_alu depctr_sa_sdst(0)                               // 000000007C50: BF88FF9E
	s_add_co_i32 s19, s19, s20                                 // 000000007C54: 81131413

0000000000007c58 <L28>:
	v_sub_nc_u32_e32 v2, 0, v1                                 // 000000007C58: 4C040280
	v_ashrrev_i32_e32 v3, 31, v1                               // 000000007C5C: 3406029F
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000007C60: BF870112
	v_max_i32_e32 v2, v2, v1                                   // 000000007C64: 24040302
	v_xor_b32_e32 v3, s13, v3                                  // 000000007C68: 3A06060D
	s_wait_alu depctr_sa_sdst(0)                               // 000000007C6C: BF88FF9E
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000007C70: BF870092
	v_mul_hi_u32 v5, v2, s19                                   // 000000007C74: D72D0005 02002702
	v_mul_lo_u32 v6, v5, s12                                   // 000000007C7C: D72C0006 02001905
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000007C84: BF870091
	v_sub_nc_u32_e32 v2, v2, v6                                // 000000007C88: 4C040D02
	v_subrev_nc_u32_e32 v6, s12, v2                            // 000000007C8C: 4E0C040C
	v_cmp_le_u32_e32 vcc_lo, s12, v2                           // 000000007C90: 7C96040C
	s_wait_alu depctr_va_vcc(0)                                // 000000007C94: BF88FF9D
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000007C98: BF870092
	v_dual_cndmask_b32 v2, v2, v6 :: v_dual_add_nc_u32 v7, 1, v5// 000000007C9C: CA600D02 02060A81
	v_cndmask_b32_e32 v5, v5, v7, vcc_lo                       // 000000007CA4: 020A0F05
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000007CA8: BF870112
	v_cmp_le_u32_e32 vcc_lo, s12, v2                           // 000000007CAC: 7C96040C
	v_add_nc_u32_e32 v6, 1, v5                                 // 000000007CB0: 4A0C0A81
	s_wait_alu depctr_va_vcc(0)                                // 000000007CB4: BF88FF9D
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000007CB8: BF870091
	v_cndmask_b32_e32 v2, v5, v6, vcc_lo                       // 000000007CBC: 02040D05
	v_xor_b32_e32 v2, v2, v3                                   // 000000007CC0: 3A040702
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000007CC4: BF870091
	v_sub_nc_u32_e32 v2, v2, v3                                // 000000007CC8: 4C040702
	v_mul_lo_u32 v3, v2, s6                                    // 000000007CCC: D72C0003 02000D02
	s_wait_kmcnt 0x0                                           // 000000007CD4: BFC70000
	v_mul_lo_u32 v2, v2, s11                                   // 000000007CD8: D72C0002 02001702
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000007CE0: BF870112
	v_sub_nc_u32_e32 v5, v1, v3                                // 000000007CE4: 4C0A0701
	v_add_co_u32 v2, s20, s2, v2                               // 000000007CE8: D7001402 02020402
	s_wait_alu depctr_va_sdst(0)                               // 000000007CF0: BF88F19F
	v_add_co_ci_u32_e64 v3, null, s3, 0, s20                   // 000000007CF4: D5207C03 00510003
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 000000007CFC: BF8700C1
	v_mad_co_u64_u32 v[2:3], null, v5, 12, v[2:3]              // 000000007D00: D6FE7C02 04091905
	global_load_b96 v[5:7], v[2:3], off                        // 000000007D08: EE05807C 00000005 00000002
	v_mul_lo_u32 v2, v1, 12                                    // 000000007D14: D72C0002 02011901
	v_add_nc_u32_e32 v1, s17, v1                               // 000000007D1C: 4A020211
	v_cmp_le_i32_e32 vcc_lo, s0, v1                            // 000000007D20: 7C860200
	s_delay_alu instid0(VALU_DEP_3)                            // 000000007D24: BF870003
	v_add_nc_u32_e32 v3, 0xe70, v2                             // 000000007D28: 4A0604FF 00000E70
	s_wait_loadcnt 0x0                                         // 000000007D30: BFC00000
	ds_store_2addr_b32 v3, v5, v6 offset1:1                    // 000000007D34: D8380100 00060503
	ds_store_b32 v2, v7 offset:3704                            // 000000007D3C: D8340E78 00000702
	s_or_b32 s18, vcc_lo, s18                                  // 000000007D44: 8C12126A
	s_wait_alu depctr_sa_sdst(0)                               // 000000007D48: BF88FF9E
	s_and_not1_b32 exec_lo, exec_lo, s18                       // 000000007D4C: 917E127E
	s_cbranch_execnz L28                                       // 000000007D50: BFA6FFC1

0000000000007d54 <L29>:
	s_or_b32 exec_lo, exec_lo, s1                              // 000000007D54: 8C7E017E
	s_wait_dscnt 0x0                                           // 000000007D58: BFC60000
	s_barrier_signal -1                                        // 000000007D5C: BE804EC1
	v_add_nc_u32_e32 v3, s15, v0                               // 000000007D60: 4A06000F
	v_add_nc_u32_e32 v5, s16, v4                               // 000000007D64: 4A0A0810
	v_cmp_gt_u32_e32 vcc_lo, 16, v0                            // 000000007D68: 7C980090
	v_cmp_gt_u32_e64 s0, 8, v4                                 // 000000007D6C: D44C0000 02020888
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000007D74: BF870214
	v_cmp_gt_i32_e64 s1, s4, v3                                // 000000007D78: D4440001 02020604
	v_cmp_gt_i32_e64 s2, s5, v5                                // 000000007D80: D4440002 02020A05
	s_and_b32 s0, vcc_lo, s0                                   // 000000007D88: 8B00006A
	s_wait_alu depctr_sa_sdst(0)                               // 000000007D8C: BF88FF9E
	s_and_b32 s0, s0, s1                                       // 000000007D90: 8B000100
	s_wait_alu depctr_sa_sdst(0)                               // 000000007D94: BF88FF9E
	s_and_b32 s0, s0, s2                                       // 000000007D98: 8B000200
	s_barrier_wait 0xffff                                      // 000000007D9C: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000007DA0: EE0AC07C 00040000 00000000
	s_wait_alu depctr_sa_sdst(0)                               // 000000007DAC: BF88FF9E
	s_and_saveexec_b32 s1, s0                                  // 000000007DB0: BE812000
	s_cbranch_execz L50                                        // 000000007DB4: BFA50110
	s_cmp_lt_i32 s7, 1                                         // 000000007DB8: BF048107
	s_cbranch_scc1 L33                                         // 000000007DBC: BFA20036
	v_mul_u32_u24_e32 v0, 12, v0                               // 000000007DC0: 1600008C
	v_dual_mov_b32 v8, 0 :: v_dual_mov_b32 v7, 0               // 000000007DC4: CA100080 08060080
	v_mov_b32_e32 v6, 0                                        // 000000007DCC: 7E0C0280
	s_cmp_gt_i32 s6, 0                                         // 000000007DD0: BF028006
	s_mov_b32 s0, 0                                            // 000000007DD4: BE800080
	s_cselect_b32 s1, -1, 0                                    // 000000007DD8: 980180C1
	s_branch L31                                               // 000000007DDC: BFA00004

0000000000007de0 <L30>:
	s_add_co_i32 s0, s0, 1                                     // 000000007DE0: 81008100
	s_wait_alu depctr_sa_sdst(0)                               // 000000007DE4: BF88FF9E
	s_cmp_eq_u32 s0, s7                                        // 000000007DE8: BF060700
	s_cbranch_scc1 L34                                         // 000000007DEC: BFA2002D

0000000000007df0 <L31>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000007DF0: BF88FF9E
	s_and_b32 s2, s1, exec_lo                                  // 000000007DF4: 8B027E01
	s_cselect_b32 s2, 1, 0                                     // 000000007DF8: 98028081
	s_wait_alu depctr_sa_sdst(0)                               // 000000007DFC: BF88FF9E
	s_cmp_lg_u32 s2, 1                                         // 000000007E00: BF078102
	s_cbranch_scc1 L30                                         // 000000007E04: BFA2FFF6
	v_add_nc_u32_e32 v1, s0, v4                                // 000000007E08: 4A020800
	s_mul_i32 s2, s0, s6                                       // 000000007E0C: 96020600
	s_mov_b32 s3, 0                                            // 000000007E10: BE830080
	s_wait_alu depctr_sa_sdst(0)                               // 000000007E14: BF88FF9E
	s_mul_i32 s2, s2, 12                                       // 000000007E18: 96028C02
	s_wait_alu depctr_sa_sdst(0)                               // 000000007E1C: BF88FF9E
	s_addk_co_i32 s2, 0xe70                                    // 000000007E20: B7820E70
	v_mul_lo_u32 v1, v1, s14                                   // 000000007E24: D72C0001 02001D01
	s_delay_alu instid0(VALU_DEP_1)                            // 000000007E2C: BF870001
	v_mad_co_u64_u32 v[1:2], null, v1, 12, v[0:1]              // 000000007E30: D6FE7C01 04011901

0000000000007e38 <L32>:
	s_mul_i32 s4, s3, 12                                       // 000000007E38: 96048C03
	s_add_co_i32 s3, s3, 1                                     // 000000007E3C: 81038103
	s_wait_alu depctr_sa_sdst(0)                               // 000000007E40: BF88FF9E
	s_delay_alu instid0(VALU_DEP_1)                            // 000000007E44: BF870001
	v_add_nc_u32_e32 v2, s4, v1                                // 000000007E48: 4A040204
	s_add_co_i32 s4, s2, s4                                    // 000000007E4C: 81040402
	s_cmp_eq_u32 s3, s6                                        // 000000007E50: BF060603
	s_wait_alu depctr_sa_sdst(0)                               // 000000007E54: BF88FF9E
	v_mov_b32_e32 v13, s4                                      // 000000007E58: 7E1A0204
	ds_load_2addr_b32 v[9:10], v2 offset1:1                    // 000000007E5C: D8DC0100 09000002
	ds_load_b32 v2, v2 offset:8                                // 000000007E64: D8D80008 02000002
	ds_load_2addr_b32 v[11:12], v13 offset1:1                  // 000000007E6C: D8DC0100 0B00000D
	ds_load_b32 v13, v13 offset:8                              // 000000007E74: D8D80008 0D00000D
	s_wait_dscnt 0x1                                           // 000000007E7C: BFC60001
	v_dual_fmac_f32 v8, v9, v11 :: v_dual_fmac_f32 v7, v10, v12// 000000007E80: C8001709 0806190A
	s_wait_dscnt 0x0                                           // 000000007E88: BFC60000
	v_fmac_f32_e32 v6, v2, v13                                 // 000000007E8C: 560C1B02
	s_cbranch_scc0 L32                                         // 000000007E90: BFA1FFE9
	s_branch L30                                               // 000000007E94: BFA0FFD2

0000000000007e98 <L33>:
	v_dual_mov_b32 v8, 0 :: v_dual_mov_b32 v7, 0               // 000000007E98: CA100080 08060080
	v_mov_b32_e32 v6, 0                                        // 000000007EA0: 7E0C0280

0000000000007ea4 <L34>:
	v_mov_b16_e32 v0.l, 0xff                                   // 000000007EA4: 7E0038FF 000000FF
	s_mov_b32 s2, exec_lo                                      // 000000007EAC: BE82007E
	s_delay_alu instid0(VALU_DEP_3)                            // 000000007EB0: BF870003
	v_cmpx_nlt_f32_e32 0x437f0000, v8                          // 000000007EB4: 7D3C10FF 437F0000
	s_cbranch_execz L39                                        // 000000007EBC: BFA50036
	v_mov_b16_e32 v0.l, 0                                      // 000000007EC0: 7E003880
	s_mov_b32 s3, exec_lo                                      // 000000007EC4: BE83007E
	v_cmpx_ngt_f32_e32 0, v8                                   // 000000007EC8: 7D361080
	s_cbranch_execz L38                                        // 000000007ECC: BFA50030
	s_mov_b32 s4, exec_lo                                      // 000000007ED0: BE84007E
	v_cmpx_o_f32_e32 v8, v8                                    // 000000007ED4: 7D2E1108
	s_cbranch_execz L37                                        // 000000007ED8: BFA5002A
	v_add_f32_e32 v0, v8, v8                                   // 000000007EDC: 06001108
	v_cmp_nlg_f32_e32 vcc_lo, 0, v8                            // 000000007EE0: 7C341080
	v_cmp_neq_f32_e64 s1, 0, v8                                // 000000007EE4: D41D0001 02021080
	s_delay_alu instid0(VALU_DEP_3)                            // 000000007EEC: BF870003
	v_cmp_neq_f32_e64 s0, v0, v8                               // 000000007EF0: D41D0000 02021100
	s_or_b32 s0, vcc_lo, s0                                    // 000000007EF8: 8C00006A
	s_wait_alu depctr_sa_sdst(0)                               // 000000007EFC: BF88FF9E
	s_and_b32 s1, s1, s0                                       // 000000007F00: 8B010001
	s_wait_alu depctr_sa_sdst(0)                               // 000000007F04: BF88FF9E
	s_and_saveexec_b32 s0, s1                                  // 000000007F08: BE802001
	s_cbranch_execz L36                                        // 000000007F0C: BFA5001B
	v_cvt_i32_f32_e32 v1, v8                                   // 000000007F10: 7E021108
	s_mov_b32 s1, exec_lo                                      // 000000007F14: BE81007E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000007F18: BF870091
	v_cvt_f32_i32_e32 v0, v1                                   // 000000007F1C: 7E000B01
	v_sub_f32_e32 v2, v8, v0                                   // 000000007F20: 08040108
	s_delay_alu instid0(VALU_DEP_1)                            // 000000007F24: BF870001
	v_cmpx_ngt_f32_e32 0.5, v2                                 // 000000007F28: 7D3604F0
	s_cbranch_execz L35                                        // 000000007F2C: BFA50010
	s_mov_b32 s5, exec_lo                                      // 000000007F30: BE85007E
	v_cmpx_nlt_f32_e32 0.5, v2                                 // 000000007F34: 7D3C04F0
	s_wait_alu depctr_sa_sdst(0)                               // 000000007F38: BF88FF9E
	s_xor_b32 s5, exec_lo, s5                                  // 000000007F3C: 8D05057E
	v_dual_add_f32 v2, 1.0, v0 :: v_dual_and_b32 v1, 1, v1     // 000000007F40: C92400F2 02000281
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000007F48: BF870121
	v_cmp_eq_u32_e32 vcc_lo, 0, v1                             // 000000007F4C: 7C940280
	s_wait_alu depctr_va_vcc(0)                                // 000000007F50: BF88FF9D
	v_cndmask_b32_e32 v0, v2, v0, vcc_lo                       // 000000007F54: 02000102
	s_wait_alu depctr_sa_sdst(0)                               // 000000007F58: BF88FF9E
	s_and_not1_saveexec_b32 s5, s5                             // 000000007F5C: BE853005
	s_delay_alu instid0(VALU_DEP_1)                            // 000000007F60: BF870001
	v_add_f32_e32 v0, 1.0, v0                                  // 000000007F64: 060000F2
	s_wait_alu depctr_sa_sdst(0)                               // 000000007F68: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s5                              // 000000007F6C: 8C7E057E

0000000000007f70 <L35>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000007F70: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s1                              // 000000007F74: 8C7E017E
	v_mov_b32_e32 v8, v0                                       // 000000007F78: 7E100300

0000000000007f7c <L36>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000007F7C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s0                              // 000000007F80: 8C7E007E

0000000000007f84 <L37>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000007F84: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s4                              // 000000007F88: 8C7E047E
	v_cvt_i32_f32_e32 v0, v8                                   // 000000007F8C: 7E001108

0000000000007f90 <L38>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000007F90: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s3                              // 000000007F94: 8C7E037E

0000000000007f98 <L39>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000007F98: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s2                              // 000000007F9C: 8C7E027E
	v_mov_b32_e32 v1, 0xff00                                   // 000000007FA0: 7E0202FF 0000FF00
	s_mov_b32 s2, exec_lo                                      // 000000007FA8: BE82007E
	v_cmpx_nlt_f32_e32 0x437f0000, v7                          // 000000007FAC: 7D3C0EFF 437F0000
	s_cbranch_execz L44                                        // 000000007FB4: BFA50039
	v_mov_b32_e32 v1, 0                                        // 000000007FB8: 7E020280
	s_mov_b32 s3, exec_lo                                      // 000000007FBC: BE83007E
	v_cmpx_ngt_f32_e32 0, v7                                   // 000000007FC0: 7D360E80
	s_cbranch_execz L43                                        // 000000007FC4: BFA50033
	s_mov_b32 s4, exec_lo                                      // 000000007FC8: BE84007E
	v_cmpx_o_f32_e32 v7, v7                                    // 000000007FCC: 7D2E0F07
	s_cbranch_execz L42                                        // 000000007FD0: BFA5002A
	v_add_f32_e32 v1, v7, v7                                   // 000000007FD4: 06020F07
	v_cmp_nlg_f32_e32 vcc_lo, 0, v7                            // 000000007FD8: 7C340E80
	v_cmp_neq_f32_e64 s1, 0, v7                                // 000000007FDC: D41D0001 02020E80
	s_delay_alu instid0(VALU_DEP_3)                            // 000000007FE4: BF870003
	v_cmp_neq_f32_e64 s0, v1, v7                               // 000000007FE8: D41D0000 02020F01
	s_or_b32 s0, vcc_lo, s0                                    // 000000007FF0: 8C00006A
	s_wait_alu depctr_sa_sdst(0)                               // 000000007FF4: BF88FF9E
	s_and_b32 s1, s1, s0                                       // 000000007FF8: 8B010001
	s_wait_alu depctr_sa_sdst(0)                               // 000000007FFC: BF88FF9E
	s_and_saveexec_b32 s0, s1                                  // 000000008000: BE802001
	s_cbranch_execz L41                                        // 000000008004: BFA5001B
	v_cvt_i32_f32_e32 v2, v7                                   // 000000008008: 7E041107
	s_mov_b32 s1, exec_lo                                      // 00000000800C: BE81007E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000008010: BF870091
	v_cvt_f32_i32_e32 v1, v2                                   // 000000008014: 7E020B02
	v_sub_f32_e32 v4, v7, v1                                   // 000000008018: 08080307
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000801C: BF870001
	v_cmpx_ngt_f32_e32 0.5, v4                                 // 000000008020: 7D3608F0
	s_cbranch_execz L40                                        // 000000008024: BFA50010
	s_mov_b32 s5, exec_lo                                      // 000000008028: BE85007E
	v_cmpx_nlt_f32_e32 0.5, v4                                 // 00000000802C: 7D3C08F0
	s_wait_alu depctr_sa_sdst(0)                               // 000000008030: BF88FF9E
	s_xor_b32 s5, exec_lo, s5                                  // 000000008034: 8D05057E
	v_add_f32_e32 v4, 1.0, v1                                  // 000000008038: 060802F2
	v_and_b32_e32 v2, 1, v2                                    // 00000000803C: 36040481
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)// 000000008040: BF8701A1
	v_cmp_eq_u32_e32 vcc_lo, 0, v2                             // 000000008044: 7C940480
	s_wait_alu depctr_va_vcc(0)                                // 000000008048: BF88FF9D
	v_cndmask_b32_e32 v1, v4, v1, vcc_lo                       // 00000000804C: 02020304
	s_wait_alu depctr_sa_sdst(0)                               // 000000008050: BF88FF9E
	s_and_not1_saveexec_b32 s5, s5                             // 000000008054: BE853005
	s_delay_alu instid0(VALU_DEP_1)                            // 000000008058: BF870001
	v_add_f32_e32 v1, 1.0, v1                                  // 00000000805C: 060202F2
	s_wait_alu depctr_sa_sdst(0)                               // 000000008060: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s5                              // 000000008064: 8C7E057E

0000000000008068 <L40>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000008068: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000806C: 8C7E017E
	v_mov_b32_e32 v7, v1                                       // 000000008070: 7E0E0301

0000000000008074 <L41>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000008074: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s0                              // 000000008078: 8C7E007E

000000000000807c <L42>:
	s_wait_alu depctr_sa_sdst(0)                               // 00000000807C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s4                              // 000000008080: 8C7E047E
	v_cvt_i32_f32_e32 v1, v7                                   // 000000008084: 7E021107
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000008088: BF870091
	v_cvt_u32_u16_e32 v1, v1.l                                 // 00000000808C: 7E02D701
	v_lshlrev_b32_e32 v1, 8, v1                                // 000000008090: 30020288

0000000000008094 <L43>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000008094: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s3                              // 000000008098: 8C7E037E

000000000000809c <L44>:
	s_wait_alu depctr_sa_sdst(0)                               // 00000000809C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s2                              // 0000000080A0: 8C7E027E
	v_mov_b32_e32 v2, 0xff0000                                 // 0000000080A4: 7E0402FF 00FF0000
	s_mov_b32 s2, exec_lo                                      // 0000000080AC: BE82007E
	v_cmpx_nlt_f32_e32 0x437f0000, v6                          // 0000000080B0: 7D3C0CFF 437F0000
	s_cbranch_execz L49                                        // 0000000080B8: BFA50038
	v_mov_b32_e32 v2, 0                                        // 0000000080BC: 7E040280
	s_mov_b32 s3, exec_lo                                      // 0000000080C0: BE83007E
	v_cmpx_ngt_f32_e32 0, v6                                   // 0000000080C4: 7D360C80
	s_cbranch_execz L48                                        // 0000000080C8: BFA50032
	s_mov_b32 s4, exec_lo                                      // 0000000080CC: BE84007E
	v_cmpx_o_f32_e32 v6, v6                                    // 0000000080D0: 7D2E0D06
	s_cbranch_execz L47                                        // 0000000080D4: BFA5002A
	v_add_f32_e32 v2, v6, v6                                   // 0000000080D8: 06040D06
	v_cmp_nlg_f32_e32 vcc_lo, 0, v6                            // 0000000080DC: 7C340C80
	v_cmp_neq_f32_e64 s1, 0, v6                                // 0000000080E0: D41D0001 02020C80
	s_delay_alu instid0(VALU_DEP_3)                            // 0000000080E8: BF870003
	v_cmp_neq_f32_e64 s0, v2, v6                               // 0000000080EC: D41D0000 02020D02
	s_or_b32 s0, vcc_lo, s0                                    // 0000000080F4: 8C00006A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000080F8: BF88FF9E
	s_and_b32 s1, s1, s0                                       // 0000000080FC: 8B010001
	s_wait_alu depctr_sa_sdst(0)                               // 000000008100: BF88FF9E
	s_and_saveexec_b32 s0, s1                                  // 000000008104: BE802001
	s_cbranch_execz L46                                        // 000000008108: BFA5001B
	v_cvt_i32_f32_e32 v4, v6                                   // 00000000810C: 7E081106
	s_mov_b32 s1, exec_lo                                      // 000000008110: BE81007E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000008114: BF870091
	v_cvt_f32_i32_e32 v2, v4                                   // 000000008118: 7E040B04
	v_sub_f32_e32 v6, v6, v2                                   // 00000000811C: 080C0506
	s_delay_alu instid0(VALU_DEP_1)                            // 000000008120: BF870001
	v_cmpx_ngt_f32_e32 0.5, v6                                 // 000000008124: 7D360CF0
	s_cbranch_execz L45                                        // 000000008128: BFA50010
	s_mov_b32 s5, exec_lo                                      // 00000000812C: BE85007E
	v_cmpx_nlt_f32_e32 0.5, v6                                 // 000000008130: 7D3C0CF0
	s_wait_alu depctr_sa_sdst(0)                               // 000000008134: BF88FF9E
	s_xor_b32 s5, exec_lo, s5                                  // 000000008138: 8D05057E
	v_and_b32_e32 v4, 1, v4                                    // 00000000813C: 36080881
	v_add_f32_e32 v6, 1.0, v2                                  // 000000008140: 060C04F2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000008144: BF870122
	v_cmp_eq_u32_e32 vcc_lo, 0, v4                             // 000000008148: 7C940880
	s_wait_alu depctr_va_vcc(0)                                // 00000000814C: BF88FF9D
	v_cndmask_b32_e32 v2, v6, v2, vcc_lo                       // 000000008150: 02040506
	s_wait_alu depctr_sa_sdst(0)                               // 000000008154: BF88FF9E
	s_and_not1_saveexec_b32 s5, s5                             // 000000008158: BE853005
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000815C: BF870001
	v_add_f32_e32 v2, 1.0, v2                                  // 000000008160: 060404F2
	s_wait_alu depctr_sa_sdst(0)                               // 000000008164: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s5                              // 000000008168: 8C7E057E

000000000000816c <L45>:
	s_wait_alu depctr_sa_sdst(0)                               // 00000000816C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s1                              // 000000008170: 8C7E017E
	v_mov_b32_e32 v6, v2                                       // 000000008174: 7E0C0302

0000000000008178 <L46>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000008178: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s0                              // 00000000817C: 8C7E007E

0000000000008180 <L47>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000008180: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s4                              // 000000008184: 8C7E047E
	v_cvt_i32_f32_e32 v2, v6                                   // 000000008188: 7E041106
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000818C: BF870001
	v_lshlrev_b32_e32 v2, 16, v2                               // 000000008190: 30040490

0000000000008194 <L48>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000008194: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s3                              // 000000008198: 8C7E037E

000000000000819c <L49>:
	s_wait_alu depctr_sa_sdst(0)                               // 00000000819C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s2                              // 0000000081A0: 8C7E027E
	s_wait_kmcnt 0x0                                           // 0000000081A4: BFC70000
	v_mul_lo_u32 v4, s10, v5                                   // 0000000081A8: D72C0004 02020A0A
	v_and_or_b32 v0, 0xff00, v1, v0                            // 0000000081B0: D6570000 040202FF 0000FF00
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 0000000081BC: BF8700B2
	v_add_co_u32 v4, s0, s8, v4                                // 0000000081C0: D7000004 02020808
	s_wait_alu depctr_va_sdst(0)                               // 0000000081C8: BF88F19F
	v_add_co_ci_u32_e64 v5, null, s9, 0, s0                    // 0000000081CC: D5207C05 00010009
	v_mad_co_i64_i32 v[3:4], null, v3, 3, v[4:5]               // 0000000081D4: D6FF7C03 04110703
	s_clause 0x1                                               // 0000000081DC: BF850001
	global_store_d16_hi_b8 v[3:4], v2, off offset:2            // 0000000081E0: EE09007C 01000000 00000203
	global_store_b16 v[3:4], v0, off                           // 0000000081EC: EE06407C 00000000 00000003

00000000000081f8 <L50>:
	s_endpgm                                                   // 0000000081F8: BFB00000
	v_illegal                                                  // 0000000081FC: 00000000

0000000000008200 <_ZN2fk25launchTransformDPP_KernelILNS_7ParArchE3ELNS_2TFE0ELb1ENS_20TransformDPPDetails_IvLb0EJNS_29ReadBackInstantiableOperationINS_14ResizeCompleteILNS_11AspectRatioE1ENS_28TernaryInstantiableOperationINS_19InterpolateCompleteILNS_17InterpolationTypeE1ENS_25ReadInstantiableOperationINS_13PerThreadReadILNS_2NDE2ENS_6Uchar3EEEEEEEEELNS_14ResizeSamplingE0EEEEENS_26WriteInstantiableOperationINS_14PerThreadWriteILSC_2ENS_6Float3EEEEEEEEJSK_SP_EEEvT2_DpKT3_>:
	s_clause 0x1                                               // 000000008200: BF850001
	s_load_b32 s4, s[0:1], 0x5c                                // 000000008204: F4000100 F800005C
	s_load_b64 s[2:3], s[0:1], 0x8                             // 00000000820C: F4002080 F8000008
	v_and_b32_e32 v1, 0x3ff, v0                                // 000000008214: 360200FF 000003FF
	v_bfe_u32 v2, v0, 10, 10                                   // 00000000821C: D6100002 02291500
	s_wait_kmcnt 0x0                                           // 000000008224: BFC70000
	s_and_b32 s5, s4, 0xffff                                   // 000000008228: 8B05FF04 0000FFFF
	s_lshr_b32 s4, s4, 16                                      // 000000008230: 85049004
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000008234: BF870121
	v_mad_co_u64_u32 v[0:1], null, ttmp9, s5, v[1:2]           // 000000008238: D6FE7C00 04040A75
	v_mad_co_u64_u32 v[1:2], null, ttmp7, s4, v[2:3]           // 000000008240: D6FE7C01 04080873
	v_cmp_gt_u32_e32 vcc_lo, s2, v0                            // 000000008248: 7C980002
	s_delay_alu instid0(VALU_DEP_2)                            // 00000000824C: BF870002
	v_cmp_gt_u32_e64 s2, s3, v1                                // 000000008250: D44C0002 02020203
	s_and_b32 s2, vcc_lo, s2                                   // 000000008258: 8B02026A
	s_wait_alu depctr_sa_sdst(0)                               // 00000000825C: BF88FF9E
	s_and_saveexec_b32 s3, s2                                  // 000000008260: BE832002
	s_cbranch_execz L51                                        // 000000008264: BFA50091
	s_load_b64 s[2:3], s[0:1], 0x10                            // 000000008268: F4002080 F8000010
	v_cvt_f32_i32_e32 v10, v1                                  // 000000008270: 7E140B01
	s_clause 0x1                                               // 000000008274: BF850001
	s_load_b64 s[8:9], s[0:1], 0x20                            // 000000008278: F4002200 F8000020
	s_load_b96 s[4:6], s[0:1], 0x28                            // 000000008280: F400A100 F8000028
	v_cvt_f32_i32_e32 v11, v0                                  // 000000008288: 7E160B00
	s_clause 0x1                                               // 00000000828C: BF850001
	s_load_b64 s[10:11], s[0:1], 0x38                          // 000000008290: F4002280 F8000038
	s_load_b32 s0, s[0:1], 0x48                                // 000000008298: F4000000 F8000048
	s_wait_kmcnt 0x0                                           // 0000000082A0: BFC70000
	v_dual_mul_f32 v2, s3, v10 :: v_dual_mul_f32 v3, s2, v11   // 0000000082A4: C8C61403 02021602
	s_add_co_i32 s1, s5, -1                                    // 0000000082AC: 8101C105
	v_mul_lo_u32 v1, s0, v1                                    // 0000000082B0: D72C0001 02020200
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000082B8: BF870092
	v_floor_f32_e32 v2, v2                                     // 0000000082BC: 7E044902
	v_cvt_i32_f32_e32 v12, v2                                  // 0000000082C0: 7E181102
	v_floor_f32_e32 v2, v3                                     // 0000000082C4: 7E044903
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000082C8: BF870112
	v_add_nc_u32_e32 v13, 1, v12                               // 0000000082CC: 4A1A1881
	v_cvt_i32_f32_e32 v14, v2                                  // 0000000082D0: 7E1C1102
	v_mul_lo_u32 v3, s6, v12                                   // 0000000082D4: D72C0003 02021806
	s_wait_alu depctr_sa_sdst(0)                               // 0000000082DC: BF88FF9E
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 0000000082E0: BF870193
	v_min_i32_e32 v2, s1, v13                                  // 0000000082E4: 22041A01
	v_add_nc_u32_e32 v15, 1, v14                               // 0000000082E8: 4A1E1C81
	s_add_co_i32 s1, s4, -1                                    // 0000000082EC: 8101C104
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 0000000082F0: BF870122
	v_mul_lo_u32 v4, v2, s6                                    // 0000000082F4: D72C0004 02000D02
	s_wait_alu depctr_sa_sdst(0)                               // 0000000082FC: BF88FF9E
	v_min_i32_e32 v8, s1, v15                                  // 000000008300: 22101E01
	v_add_co_u32 v2, s1, s8, v3                                // 000000008304: D7000102 02020608
	s_wait_alu depctr_va_sdst(0)                               // 00000000830C: BF88F19F
	v_add_co_ci_u32_e64 v3, null, s9, 0, s1                    // 000000008310: D5207C03 00050009
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)// 000000008318: BF8701B4
	v_add_co_u32 v4, s1, s8, v4                                // 00000000831C: D7000104 02020808
	s_wait_alu depctr_va_sdst(0)                               // 000000008324: BF88F19F
	v_add_co_ci_u32_e64 v5, null, s9, 0, s1                    // 000000008328: D5207C05 00050009
	v_mad_co_i64_i32 v[6:7], null, v14, 3, v[2:3]              // 000000008330: D6FF7C06 0409070E
	v_mad_co_i64_i32 v[2:3], null, v8, 3, v[2:3]               // 000000008338: D6FF7C02 04090708
	s_delay_alu instid0(VALU_DEP_3)                            // 000000008340: BF870003
	v_mad_co_i64_i32 v[8:9], null, v8, 3, v[4:5]               // 000000008344: D6FF7C08 04110708
	v_mad_co_i64_i32 v[4:5], null, v14, 3, v[4:5]              // 00000000834C: D6FF7C04 0411070E
	s_clause 0x7                                               // 000000008354: BF850007
	global_load_u8 v16, v[6:7], off offset:2                   // 000000008358: EE04007C 00000010 00000206
	global_load_u8 v17, v[2:3], off offset:2                   // 000000008364: EE04007C 00000011 00000202
	global_load_u16 v18, v[8:9], off                           // 000000008370: EE04807C 00000012 00000008
	global_load_u16 v19, v[4:5], off                           // 00000000837C: EE04807C 00000013 00000004
	global_load_u16 v2, v[2:3], off                            // 000000008388: EE04807C 00000002 00000002
	global_load_u16 v6, v[6:7], off                            // 000000008394: EE04807C 00000006 00000006
	global_load_u8 v7, v[4:5], off offset:2                    // 0000000083A0: EE04007C 00000007 00000204
	global_load_u8 v8, v[8:9], off offset:2                    // 0000000083AC: EE04007C 00000008 00000208
	v_cvt_f32_i32_e32 v3, v14                                  // 0000000083B8: 7E060B0E
	v_cvt_f32_i32_e32 v5, v15                                  // 0000000083BC: 7E0A0B0F
	v_cvt_f32_i32_e32 v9, v13                                  // 0000000083C0: 7E120B0D
	v_cvt_f32_i32_e32 v4, v12                                  // 0000000083C4: 7E080B0C
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 0000000083C8: BF870214
	v_fma_f32 v3, s2, v11, -v3                                 // 0000000083CC: D6130003 840E1602
	v_fma_f32 v11, -s2, v11, v5                                // 0000000083D4: D613000B 24161602
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 0000000083DC: BF870214
	v_fma_f32 v9, -s3, v10, v9                                 // 0000000083E0: D6130009 24261403
	v_fma_f32 v12, s3, v10, -v4                                // 0000000083E8: D613000C 84121403
	v_add_co_u32 v4, s0, s10, v1                               // 0000000083F0: D7000004 0202020A
	s_wait_alu depctr_va_sdst(0)                               // 0000000083F8: BF88F19F
	v_add_co_ci_u32_e64 v5, null, s11, 0, s0                   // 0000000083FC: D5207C05 0001000B
	v_mul_f32_e32 v10, v9, v11                                 // 000000008404: 10141709
	v_mul_f32_e32 v13, v12, v3                                 // 000000008408: 101A070C
	v_mul_f32_e32 v9, v3, v9                                   // 00000000840C: 10121303
	v_mul_f32_e32 v11, v12, v11                                // 000000008410: 1016170C
	v_mad_co_i64_i32 v[4:5], null, v0, 12, v[4:5]              // 000000008414: D6FF7C04 04111900
	s_wait_loadcnt 0x7                                         // 00000000841C: BFC00007
	v_cvt_f32_ubyte0_e32 v1, v16                               // 000000008420: 7E022310
	s_wait_loadcnt 0x6                                         // 000000008424: BFC00006
	v_cvt_f32_ubyte0_e32 v12, v17                              // 000000008428: 7E182311
	s_wait_loadcnt 0x3                                         // 00000000842C: BFC00003
	v_cvt_f32_ubyte1_e32 v15, v2                               // 000000008430: 7E1E2502
	v_mul_f32_e32 v3, v10, v1                                  // 000000008434: 1006030A
	s_wait_loadcnt 0x2                                         // 000000008438: BFC00002
	v_cvt_f32_ubyte0_e32 v1, v6                                // 00000000843C: 7E022306
	v_cvt_f32_ubyte1_e32 v6, v6                                // 000000008440: 7E0C2506
	v_cvt_f32_ubyte0_e32 v14, v2                               // 000000008444: 7E1C2302
	s_wait_loadcnt 0x1                                         // 000000008448: BFC00001
	v_cvt_f32_ubyte0_e32 v7, v7                                // 00000000844C: 7E0E2307
	s_wait_loadcnt 0x0                                         // 000000008450: BFC00000
	v_cvt_f32_ubyte0_e32 v8, v8                                // 000000008454: 7E102308
	v_dual_mul_f32 v1, v10, v1 :: v_dual_mul_f32 v2, v10, v6   // 000000008458: C8C6030A 01020D0A
	v_cvt_f32_ubyte1_e32 v10, v19                              // 000000008460: 7E142513
	v_cvt_f32_ubyte0_e32 v6, v19                               // 000000008464: 7E0C2313
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000008468: BF870123
	v_dual_fmac_f32 v2, v9, v15 :: v_dual_fmac_f32 v3, v9, v12 // 00000000846C: C8001F09 02021909
	v_cvt_f32_ubyte1_e32 v12, v18                              // 000000008474: 7E182512
	v_dual_fmac_f32 v2, v11, v10 :: v_dual_fmac_f32 v3, v11, v7// 000000008478: C800150B 02020F0B
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_4)// 000000008480: BF870231
	v_fmac_f32_e32 v2, v13, v12                                // 000000008484: 5604190D
	v_fmac_f32_e32 v1, v9, v14                                 // 000000008488: 56021D09
	v_cvt_f32_ubyte0_e32 v9, v18                               // 00000000848C: 7E122312
	v_fmac_f32_e32 v3, v13, v8                                 // 000000008490: 5606110D
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000008494: BF870093
	v_fmac_f32_e32 v1, v11, v6                                 // 000000008498: 56020D0B
	v_fmac_f32_e32 v1, v13, v9                                 // 00000000849C: 5602130D
	global_store_b96 v[4:5], v[1:3], off                       // 0000000084A0: EE07007C 00800000 00000004

00000000000084ac <L51>:
	s_endpgm                                                   // 0000000084AC: BFB00000
		...

0000000000008500 <_ZN2fk21linearFilterDPPKernelILNS_7ParArchE3ENS_22LinearFilterDPPDetailsINS_6Float3ELi16ELi8ELi7ELi7EEENS_5TupleIJNS_29ReadBackInstantiableOperationINS_14ResizeCompleteILNS_11AspectRatioE1ENS_28TernaryInstantiableOperationINS_19InterpolateCompleteILNS_17InterpolationTypeE1ENS_25ReadInstantiableOperationINS_13PerThreadReadILNS_2NDE2ENS_6Uchar3EEEEEEEEELNS_14ResizeSamplingE0EEEEENSC_INSD_ILSE_2ES3_EEEEEEENS_26UnaryInstantiableOperationINS_3MulIS3_S3_S3_NS_9UnaryTypeEEEEENSQ_INS_3AddIS3_S3_S3_SS_EEEENS_26WriteInstantiableOperationINS_14PerThreadWriteILSE_2ES3_EEEEEEvT0_T1_T2_T3_T4_>:
	s_clause 0x3                                               // 000000008500: BF850003
	s_load_b32 s10, s[0:1], 0x8c                               // 000000008504: F4000280 F800008C
	s_load_b128 s[4:7], s[0:1], 0x0                            // 00000000850C: F4004100 F8000000
	s_load_b64 s[8:9], s[0:1], 0x68                            // 000000008514: F4002200 F8000068
	s_load_b64 s[2:3], s[0:1], 0x48                            // 00000000851C: F4002080 F8000048
	v_bfe_u32 v6, v0, 10, 10                                   // 000000008524: D6100006 02291500
	v_and_b32_e32 v0, 0x3ff, v0                                // 00000000852C: 360000FF 000003FF
	s_lshl_b32 s15, ttmp9, 4                                   // 000000008534: 840F8475
	s_lshl_b32 s16, ttmp7, 3                                   // 000000008538: 84108373
	s_mov_b32 s19, exec_lo                                     // 00000000853C: BE93007E
	s_wait_kmcnt 0x0                                           // 000000008540: BFC70000
	s_and_b32 s11, s10, 0xffff                                 // 000000008544: 8B0BFF0A 0000FFFF
	s_add_co_i32 s14, s6, 15                                   // 00000000854C: 810E8F06
	v_mad_u32_u24 v1, v6, s11, v0                              // 000000008550: D60B0001 04001706
	s_add_co_i32 s18, s7, 7                                    // 000000008558: 81128707
	s_lshr_b32 s17, s10, 16                                    // 00000000855C: 8511900A
	s_mul_i32 s18, s18, s14                                    // 000000008560: 96120E12
	s_mul_i32 s17, s17, s11                                    // 000000008564: 96110B11
	v_cmpx_gt_i32_e32 s18, v1                                  // 000000008568: 7D880212
	s_cbranch_execz L53                                        // 00000000856C: BFA500D6
	s_abs_i32 s20, s14                                         // 000000008570: BE94150E
	s_add_nc_u64 s[22:23], s[0:1], 24                          // 000000008574: A9969800
	s_cvt_f32_u32 s10, s20                                     // 000000008578: BE8A6514
	s_clause 0x1                                               // 00000000857C: BF850001
	s_load_b64 s[28:29], s[0:1], 0x10                          // 000000008580: F4002700 F8000010
	s_load_b64 s[30:31], s[0:1], 0x38                          // 000000008588: F4002780 F8000038
	v_mul_lo_u32 v2, v1, 12                                    // 000000008590: D72C0002 02011901
	v_s_rcp_f32 s26, s10                                       // 000000008598: D684001A 0201000A
	s_clause 0x2                                               // 0000000085A0: BF850002
	s_load_b64 s[10:11], s[22:23], 0x8                         // 0000000085A4: F400228B F8000008
	s_load_b64 s[12:13], s[22:23], 0x18                        // 0000000085AC: F400230B F8000018
	s_load_b32 s21, s[22:23], 0x28                             // 0000000085B4: F400054B F8000028
	s_sub_co_i32 s34, 0, s20                                   // 0000000085BC: 81A21480
	v_mov_b32_e32 v3, v1                                       // 0000000085C0: 7E060301
	s_add_co_i32 s22, s4, -1                                   // 0000000085C4: 8116C104
	s_add_co_i32 s23, s5, -1                                   // 0000000085C8: 8117C105
	s_ashr_i32 s24, s14, 31                                    // 0000000085CC: 86189F0E
	s_mov_b32 s25, 0                                           // 0000000085D0: BE990080
	s_mul_f32 s27, s26, 0x4f7ffffe                             // 0000000085D4: A21BFF1A 4F7FFFFE
	s_sub_co_i32 s26, 0, s14                                   // 0000000085DC: 819A0E80
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(SALU_CYCLE_3)// 0000000085E0: BF87059A
	s_cvt_u32_f32 s33, s27                                     // 0000000085E4: BEA1671B
	s_mul_i32 s34, s34, s33                                    // 0000000085E8: 96222122
	s_wait_kmcnt 0x0                                           // 0000000085EC: BFC70000
	s_sub_co_i32 s27, s15, s28                                 // 0000000085F0: 819B1C0F
	s_mul_hi_u32 s34, s33, s34                                 // 0000000085F4: 96A22221
	s_sub_co_i32 s28, s16, s29                                 // 0000000085F8: 819C1D10
	s_add_co_i32 s29, s30, -1                                  // 0000000085FC: 811DC11E
	s_add_co_i32 s30, s31, -1                                  // 000000008600: 811EC11F
	s_add_co_i32 s31, s33, s34                                 // 000000008604: 811F2221
	s_mul_i32 s33, s17, 12                                     // 000000008608: 96218C11

000000000000860c <L52>:
	v_sub_nc_u32_e32 v4, 0, v3                                 // 00000000860C: 4C080680
	v_ashrrev_i32_e32 v5, 31, v3                               // 000000008610: 340A069F
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000008614: BF870112
	v_max_i32_e32 v4, v4, v3                                   // 000000008618: 24080704
	v_xor_b32_e32 v5, s24, v5                                  // 00000000861C: 3A0A0A18
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000008620: BF870092
	v_mul_hi_u32 v7, v4, s31                                   // 000000008624: D72D0007 02003F04
	v_mul_lo_u32 v8, v7, s20                                   // 00000000862C: D72C0008 02002907
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000008634: BF870091
	v_sub_nc_u32_e32 v4, v4, v8                                // 000000008638: 4C081104
	v_subrev_nc_u32_e32 v8, s20, v4                            // 00000000863C: 4E100814
	v_cmp_le_u32_e32 vcc_lo, s20, v4                           // 000000008640: 7C960814
	s_wait_alu depctr_va_vcc(0)                                // 000000008644: BF88FF9D
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000008648: BF870092
	v_dual_cndmask_b32 v4, v4, v8 :: v_dual_add_nc_u32 v9, 1, v7// 00000000864C: CA601104 04080E81
	v_cndmask_b32_e32 v7, v7, v9, vcc_lo                       // 000000008654: 020E1307
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000008658: BF870112
	v_cmp_le_u32_e32 vcc_lo, s20, v4                           // 00000000865C: 7C960814
	v_add_nc_u32_e32 v8, 1, v7                                 // 000000008660: 4A100E81
	s_wait_alu depctr_va_vcc(0)                                // 000000008664: BF88FF9D
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000008668: BF870091
	v_cndmask_b32_e32 v4, v7, v8, vcc_lo                       // 00000000866C: 02081107
	v_xor_b32_e32 v4, v4, v5                                   // 000000008670: 3A080B04
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000008674: BF870091
	v_sub_nc_u32_e32 v4, v4, v5                                // 000000008678: 4C080B04
	v_mul_lo_u32 v5, s26, v4                                   // 00000000867C: D72C0005 0202081A
	v_add_nc_u32_e32 v4, s28, v4                               // 000000008684: 4A08081C
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 000000008688: BF870221
	v_min_i32_e32 v7, s23, v4                                  // 00000000868C: 220E0817
	v_cmp_lt_i32_e32 vcc_lo, -1, v4                            // 000000008690: 7C8208C1
	v_add3_u32 v4, s27, v3, v5                                 // 000000008694: D6550004 0416061B
	s_wait_alu depctr_va_vcc(0)                                // 00000000869C: BF88FF9D
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000086A0: BF870113
	v_cndmask_b32_e32 v5, 0, v7, vcc_lo                        // 0000000086A4: 020A0E80
	v_min_i32_e32 v7, s22, v4                                  // 0000000086A8: 220E0816
	v_cmp_lt_i32_e32 vcc_lo, -1, v4                            // 0000000086AC: 7C8208C1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000086B0: BF8700A3
	v_cvt_f32_i32_e32 v13, v5                                  // 0000000086B4: 7E1A0B05
	s_wait_alu depctr_va_vcc(0)                                // 0000000086B8: BF88FF9D
	v_dual_cndmask_b32 v4, 0, v7 :: v_dual_mul_f32 v5, s11, v13// 0000000086BC: CA460E80 04041A0B
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000086C4: BF870111
	v_cvt_f32_i32_e32 v14, v4                                  // 0000000086C8: 7E1C0B04
	v_floor_f32_e32 v4, v5                                     // 0000000086CC: 7E084905
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000086D0: BF870112
	v_mul_f32_e32 v5, s10, v14                                 // 0000000086D4: 100A1C0A
	v_cvt_i32_f32_e32 v15, v4                                  // 0000000086D8: 7E1E1104
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000086DC: BF870112
	v_floor_f32_e32 v4, v5                                     // 0000000086E0: 7E084905
	v_add_nc_u32_e32 v16, 1, v15                               // 0000000086E4: 4A201E81
	v_mul_lo_u32 v5, s21, v15                                  // 0000000086E8: D72C0005 02021E15
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 0000000086F0: BF870193
	v_cvt_i32_f32_e32 v17, v4                                  // 0000000086F4: 7E221104
	v_min_i32_e32 v7, s30, v16                                 // 0000000086F8: 220E201E
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)// 0000000086FC: BF870212
	v_add_nc_u32_e32 v18, 1, v17                               // 000000008700: 4A242281
	v_add_co_u32 v4, s34, s12, v5                              // 000000008704: D7002204 02020A0C
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_2)// 00000000870C: BF870143
	v_mul_lo_u32 v9, v7, s21                                   // 000000008710: D72C0009 02002B07
	s_wait_alu depctr_va_sdst(0)                               // 000000008718: BF88F19F
	v_add_co_ci_u32_e64 v5, null, s13, 0, s34                  // 00000000871C: D5207C05 0089000D
	v_min_i32_e32 v11, s29, v18                                // 000000008724: 2216241D
	v_mad_co_i64_i32 v[7:8], null, v17, 3, v[4:5]              // 000000008728: D6FF7C07 04110711
	s_delay_alu instid0(VALU_DEP_4)                            // 000000008730: BF870004
	v_add_co_u32 v9, s34, s12, v9                              // 000000008734: D7002209 0202120C
	s_wait_alu depctr_va_sdst(0)                               // 00000000873C: BF88F19F
	v_add_co_ci_u32_e64 v10, null, s13, 0, s34                 // 000000008740: D5207C0A 0089000D
	v_mad_co_i64_i32 v[4:5], null, v11, 3, v[4:5]              // 000000008748: D6FF7C04 0411070B
	global_load_u8 v19, v[7:8], off offset:2                   // 000000008750: EE04007C 00000013 00000207
	v_mad_co_i64_i32 v[11:12], null, v11, 3, v[9:10]           // 00000000875C: D6FF7C0B 0425070B
	v_mad_co_i64_i32 v[9:10], null, v17, 3, v[9:10]            // 000000008764: D6FF7C09 04250711
	s_clause 0x6                                               // 00000000876C: BF850006
	global_load_u8 v20, v[4:5], off offset:2                   // 000000008770: EE04007C 00000014 00000204
	global_load_u16 v21, v[11:12], off                         // 00000000877C: EE04807C 00000015 0000000B
	global_load_u16 v22, v[9:10], off                          // 000000008788: EE04807C 00000016 00000009
	global_load_u16 v4, v[4:5], off                            // 000000008794: EE04807C 00000004 00000004
	global_load_u16 v5, v[7:8], off                            // 0000000087A0: EE04807C 00000005 00000007
	global_load_u8 v7, v[9:10], off offset:2                   // 0000000087AC: EE04007C 00000007 00000209
	global_load_u8 v8, v[11:12], off offset:2                  // 0000000087B8: EE04007C 00000008 0000020B
	v_cvt_f32_i32_e32 v9, v15                                  // 0000000087C4: 7E120B0F
	v_cvt_f32_i32_e32 v10, v16                                 // 0000000087C8: 7E140B10
	v_cvt_f32_i32_e32 v11, v17                                 // 0000000087CC: 7E160B11
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 0000000087D0: BF870193
	v_fma_f32 v9, s11, v13, -v9                                // 0000000087D4: D6130009 84261A0B
	v_fma_f32 v10, -s11, v13, v10                              // 0000000087DC: D613000A 242A1A0B
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000087E4: BF870093
	v_fma_f32 v11, s10, v14, -v11                              // 0000000087E8: D613000B 842E1C0A
	v_mul_f32_e32 v13, v11, v10                                // 0000000087F0: 101A150B
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000087F4: BF8700A4
	v_mul_f32_e32 v11, v9, v11                                 // 0000000087F8: 10161709
	v_cvt_f32_i32_e32 v12, v18                                 // 0000000087FC: 7E180B12
	v_fma_f32 v12, -s10, v14, v12                              // 000000008800: D613000C 24321C0A
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000008808: BF870121
	v_dual_mul_f32 v10, v10, v12 :: v_dual_add_nc_u32 v3, s17, v3// 00000000880C: C8E0190A 0A020611
	v_mul_f32_e32 v9, v9, v12                                  // 000000008814: 10121909
	v_cmp_le_i32_e32 vcc_lo, s18, v3                           // 000000008818: 7C860612
	s_or_b32 s25, vcc_lo, s25                                  // 00000000881C: 8C19196A
	s_wait_loadcnt 0x7                                         // 000000008820: BFC00007
	v_cvt_f32_ubyte0_e32 v12, v19                              // 000000008824: 7E182313
	s_wait_loadcnt 0x6                                         // 000000008828: BFC00006
	v_cvt_f32_ubyte0_e32 v14, v20                              // 00000000882C: 7E1C2314
	s_wait_loadcnt 0x5                                         // 000000008830: BFC00005
	v_cvt_f32_ubyte0_e32 v18, v21                              // 000000008834: 7E242315
	s_wait_loadcnt 0x4                                         // 000000008838: BFC00004
	v_cvt_f32_ubyte1_e32 v17, v22                              // 00000000883C: 7E222516
	v_mul_f32_e32 v12, v10, v12                                // 000000008840: 1018190A
	s_wait_loadcnt 0x2                                         // 000000008844: BFC00002
	v_cvt_f32_ubyte0_e32 v15, v5                               // 000000008848: 7E1E2305
	v_cvt_f32_ubyte1_e32 v5, v5                                // 00000000884C: 7E0A2505
	v_cvt_f32_ubyte0_e32 v16, v4                               // 000000008850: 7E202304
	v_cvt_f32_ubyte1_e32 v4, v4                                // 000000008854: 7E082504
	s_delay_alu instid0(VALU_DEP_4)                            // 000000008858: BF870004
	v_dual_fmac_f32 v12, v13, v14 :: v_dual_mul_f32 v15, v10, v15// 00000000885C: C8061D0D 0C0E1F0A
	s_wait_loadcnt 0x1                                         // 000000008864: BFC00001
	v_cvt_f32_ubyte0_e32 v7, v7                                // 000000008868: 7E0E2307
	v_mul_f32_e32 v5, v10, v5                                  // 00000000886C: 100A0B0A
	v_cvt_f32_ubyte0_e32 v14, v22                              // 000000008870: 7E1C2316
	v_cvt_f32_ubyte1_e32 v19, v21                              // 000000008874: 7E262515
	v_fmac_f32_e32 v15, v13, v16                               // 000000008878: 561E210D
	v_fmac_f32_e32 v12, v9, v7                                 // 00000000887C: 56180F09
	v_fmac_f32_e32 v5, v13, v4                                 // 000000008880: 560A090D
	s_wait_loadcnt 0x0                                         // 000000008884: BFC00000
	v_cvt_f32_ubyte0_e32 v8, v8                                // 000000008888: 7E102308
	v_fmac_f32_e32 v15, v9, v14                                // 00000000888C: 561E1D09
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000008890: BF870112
	v_dual_fmac_f32 v5, v9, v17 :: v_dual_fmac_f32 v12, v11, v8// 000000008894: C8002309 050C110B
	v_fmac_f32_e32 v15, v11, v18                               // 00000000889C: 561E250B
	s_delay_alu instid0(VALU_DEP_2)                            // 0000000088A0: BF870002
	v_fmac_f32_e32 v5, v11, v19                                // 0000000088A4: 560A270B
	ds_store_2addr_b32 v2, v15, v5 offset1:1                   // 0000000088A8: D8380100 00050F02
	ds_store_b32 v2, v12 offset:8                              // 0000000088B0: D8340008 00000C02
	v_add_nc_u32_e32 v2, s33, v2                               // 0000000088B8: 4A040421
	s_wait_alu depctr_sa_sdst(0)                               // 0000000088BC: BF88FF9E
	s_and_not1_b32 exec_lo, exec_lo, s25                       // 0000000088C0: 917E197E
	s_cbranch_execnz L52                                       // 0000000088C4: BFA6FF51

00000000000088c8 <L53>:
	s_or_b32 exec_lo, exec_lo, s19                             // 0000000088C8: 8C7E137E
	s_clause 0x1                                               // 0000000088CC: BF850001
	s_load_b32 s11, s[0:1], 0x58                               // 0000000088D0: F40002C0 F8000058
	s_load_b32 s10, s[0:1], 0x78                               // 0000000088D8: F4000280 F8000078
	s_mul_i32 s0, s7, s6                                       // 0000000088E0: 96000607
	s_mov_b32 s1, exec_lo                                      // 0000000088E4: BE81007E
	v_cmpx_gt_i32_e32 s0, v1                                   // 0000000088E8: 7D880200
	s_cbranch_execz L55                                        // 0000000088EC: BFA50054
	s_abs_i32 s12, s6                                          // 0000000088F0: BE8C1506
	s_wait_alu depctr_sa_sdst(0)                               // 0000000088F4: BF88FF9E
	s_cvt_f32_u32 s13, s12                                     // 0000000088F8: BE8D650C
	s_wait_alu depctr_sa_sdst(0)                               // 0000000088FC: BF88FF9E
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)// 000000008900: BF87029A
	v_s_rcp_f32 s13, s13                                       // 000000008904: D684000D 0201000D
	s_mul_f32 s13, s13, 0x4f7ffffe                             // 00000000890C: A20DFF0D 4F7FFFFE
	s_wait_alu depctr_sa_sdst(0)                               // 000000008914: BF88FF9E
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000008918: BF8704BA
	s_cvt_u32_f32 s19, s13                                     // 00000000891C: BE93670D
	s_sub_co_i32 s13, 0, s12                                   // 000000008920: 818D0C80
	s_wait_alu depctr_sa_sdst(0)                               // 000000008924: BF88FF9E
	s_mul_i32 s18, s13, s19                                    // 000000008928: 9612130D
	s_ashr_i32 s13, s6, 31                                     // 00000000892C: 860D9F06
	s_wait_alu depctr_sa_sdst(0)                               // 000000008930: BF88FF9E
	s_mul_hi_u32 s20, s19, s18                                 // 000000008934: 96941213
	s_mov_b32 s18, 0                                           // 000000008938: BE920080
	s_wait_alu depctr_sa_sdst(0)                               // 00000000893C: BF88FF9E
	s_add_co_i32 s19, s19, s20                                 // 000000008940: 81131413

0000000000008944 <L54>:
	v_sub_nc_u32_e32 v2, 0, v1                                 // 000000008944: 4C040280
	v_ashrrev_i32_e32 v3, 31, v1                               // 000000008948: 3406029F
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000894C: BF870112
	v_max_i32_e32 v2, v2, v1                                   // 000000008950: 24040302
	v_xor_b32_e32 v3, s13, v3                                  // 000000008954: 3A06060D
	s_wait_alu depctr_sa_sdst(0)                               // 000000008958: BF88FF9E
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000895C: BF870092
	v_mul_hi_u32 v4, v2, s19                                   // 000000008960: D72D0004 02002702
	v_mul_lo_u32 v5, v4, s12                                   // 000000008968: D72C0005 02001904
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000008970: BF870091
	v_sub_nc_u32_e32 v2, v2, v5                                // 000000008974: 4C040B02
	v_subrev_nc_u32_e32 v5, s12, v2                            // 000000008978: 4E0A040C
	v_cmp_le_u32_e32 vcc_lo, s12, v2                           // 00000000897C: 7C96040C
	s_wait_alu depctr_va_vcc(0)                                // 000000008980: BF88FF9D
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000008984: BF870092
	v_dual_cndmask_b32 v2, v2, v5 :: v_dual_add_nc_u32 v7, 1, v4// 000000008988: CA600B02 02060881
	v_cndmask_b32_e32 v4, v4, v7, vcc_lo                       // 000000008990: 02080F04
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000008994: BF870112
	v_cmp_le_u32_e32 vcc_lo, s12, v2                           // 000000008998: 7C96040C
	v_add_nc_u32_e32 v5, 1, v4                                 // 00000000899C: 4A0A0881
	s_wait_alu depctr_va_vcc(0)                                // 0000000089A0: BF88FF9D
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000089A4: BF8700A1
	v_cndmask_b32_e32 v2, v4, v5, vcc_lo                       // 0000000089A8: 02040B04
	v_mul_lo_u32 v5, v1, 12                                    // 0000000089AC: D72C0005 02011901
	v_add_nc_u32_e32 v7, 0xe70, v5                             // 0000000089B4: 4A0E0AFF 00000E70
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000089BC: BF870093
	v_xor_b32_e32 v2, v2, v3                                   // 0000000089C0: 3A040702
	v_sub_nc_u32_e32 v2, v2, v3                                // 0000000089C4: 4C040702
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 0000000089C8: BF870131
	v_mul_lo_u32 v3, v2, s6                                    // 0000000089CC: D72C0003 02000D02
	s_wait_kmcnt 0x0                                           // 0000000089D4: BFC70000
	v_mul_lo_u32 v2, v2, s11                                   // 0000000089D8: D72C0002 02001702
	v_sub_nc_u32_e32 v4, v1, v3                                // 0000000089E0: 4C080701
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)// 0000000089E4: BF870142
	v_add_co_u32 v2, s20, s2, v2                               // 0000000089E8: D7001402 02020402
	s_wait_alu depctr_va_sdst(0)                               // 0000000089F0: BF88F19F
	v_add_co_ci_u32_e64 v3, null, s3, 0, s20                   // 0000000089F4: D5207C03 00510003
	v_add_nc_u32_e32 v1, s17, v1                               // 0000000089FC: 4A020211
	v_mad_co_u64_u32 v[2:3], null, v4, 12, v[2:3]              // 000000008A00: D6FE7C02 04091904
	s_delay_alu instid0(VALU_DEP_2)                            // 000000008A08: BF870002
	v_cmp_le_i32_e32 vcc_lo, s0, v1                            // 000000008A0C: 7C860200
	s_or_b32 s18, vcc_lo, s18                                  // 000000008A10: 8C12126A
	global_load_b96 v[2:4], v[2:3], off                        // 000000008A14: EE05807C 00000002 00000002
	s_wait_loadcnt 0x0                                         // 000000008A20: BFC00000
	ds_store_2addr_b32 v7, v2, v3 offset1:1                    // 000000008A24: D8380100 00030207
	ds_store_b32 v5, v4 offset:3704                            // 000000008A2C: D8340E78 00000405
	s_wait_alu depctr_sa_sdst(0)                               // 000000008A34: BF88FF9E
	s_and_not1_b32 exec_lo, exec_lo, s18                       // 000000008A38: 917E127E
	s_cbranch_execnz L54                                       // 000000008A3C: BFA6FFC1

0000000000008a40 <L55>:
	s_or_b32 exec_lo, exec_lo, s1                              // 000000008A40: 8C7E017E
	s_wait_dscnt 0x0                                           // 000000008A44: BFC60000
	s_barrier_signal -1                                        // 000000008A48: BE804EC1
	v_add_nc_u32_e32 v7, s15, v0                               // 000000008A4C: 4A0E000F
	v_add_nc_u32_e32 v8, s16, v6                               // 000000008A50: 4A100C10
	v_cmp_gt_u32_e32 vcc_lo, 16, v0                            // 000000008A54: 7C980090
	v_cmp_gt_u32_e64 s0, 8, v6                                 // 000000008A58: D44C0000 02020C88
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000008A60: BF870214
	v_cmp_gt_i32_e64 s1, s4, v7                                // 000000008A64: D4440001 02020E04
	v_cmp_gt_i32_e64 s2, s5, v8                                // 000000008A6C: D4440002 02021005
	s_and_b32 s0, vcc_lo, s0                                   // 000000008A74: 8B00006A
	s_wait_alu depctr_sa_sdst(0)                               // 000000008A78: BF88FF9E
	s_and_b32 s0, s0, s1                                       // 000000008A7C: 8B000100
	s_wait_alu depctr_sa_sdst(0)                               // 000000008A80: BF88FF9E
	s_and_b32 s0, s0, s2                                       // 000000008A84: 8B000200
	s_barrier_wait 0xffff                                      // 000000008A88: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000008A8C: EE0AC07C 00040000 00000000
	s_wait_alu depctr_sa_sdst(0)                               // 000000008A98: BF88FF9E
	s_and_saveexec_b32 s1, s0                                  // 000000008A9C: BE812000
	s_cbranch_execz L61                                        // 000000008AA0: BFA50049
	s_cmp_lt_i32 s7, 1                                         // 000000008AA4: BF048107
	s_cbranch_scc1 L59                                         // 000000008AA8: BFA20036
	v_mul_u32_u24_e32 v3, 12, v0                               // 000000008AAC: 1606008C
	v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v1, 0               // 000000008AB0: CA100080 00000080
	v_mov_b32_e32 v2, 0                                        // 000000008AB8: 7E040280
	s_cmp_gt_i32 s6, 0                                         // 000000008ABC: BF028006
	s_mov_b32 s0, 0                                            // 000000008AC0: BE800080
	s_cselect_b32 s1, -1, 0                                    // 000000008AC4: 980180C1
	s_branch L57                                               // 000000008AC8: BFA00004

0000000000008acc <L56>:
	s_add_co_i32 s0, s0, 1                                     // 000000008ACC: 81008100
	s_wait_alu depctr_sa_sdst(0)                               // 000000008AD0: BF88FF9E
	s_cmp_eq_u32 s0, s7                                        // 000000008AD4: BF060700
	s_cbranch_scc1 L60                                         // 000000008AD8: BFA2002D

0000000000008adc <L57>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000008ADC: BF88FF9E
	s_and_b32 s2, s1, exec_lo                                  // 000000008AE0: 8B027E01
	s_cselect_b32 s2, 1, 0                                     // 000000008AE4: 98028081
	s_wait_alu depctr_sa_sdst(0)                               // 000000008AE8: BF88FF9E
	s_cmp_lg_u32 s2, 1                                         // 000000008AEC: BF078102
	s_cbranch_scc1 L56                                         // 000000008AF0: BFA2FFF6
	v_add_nc_u32_e32 v4, s0, v6                                // 000000008AF4: 4A080C00
	s_mul_i32 s2, s0, s6                                       // 000000008AF8: 96020600
	s_mov_b32 s3, 0                                            // 000000008AFC: BE830080
	s_wait_alu depctr_sa_sdst(0)                               // 000000008B00: BF88FF9E
	s_mul_i32 s2, s2, 12                                       // 000000008B04: 96028C02
	s_wait_alu depctr_sa_sdst(0)                               // 000000008B08: BF88FF9E
	s_addk_co_i32 s2, 0xe70                                    // 000000008B0C: B7820E70
	v_mul_lo_u32 v4, v4, s14                                   // 000000008B10: D72C0004 02001D04
	s_delay_alu instid0(VALU_DEP_1)                            // 000000008B18: BF870001
	v_mad_co_u64_u32 v[4:5], null, v4, 12, v[3:4]              // 000000008B1C: D6FE7C04 040D1904

0000000000008b24 <L58>:
	s_mul_i32 s4, s3, 12                                       // 000000008B24: 96048C03
	s_add_co_i32 s3, s3, 1                                     // 000000008B28: 81038103
	s_wait_alu depctr_sa_sdst(0)                               // 000000008B2C: BF88FF9E
	s_delay_alu instid0(VALU_DEP_1)                            // 000000008B30: BF870001
	v_add_nc_u32_e32 v5, s4, v4                                // 000000008B34: 4A0A0804
	s_add_co_i32 s4, s2, s4                                    // 000000008B38: 81040402
	s_cmp_eq_u32 s3, s6                                        // 000000008B3C: BF060603
	s_wait_alu depctr_sa_sdst(0)                               // 000000008B40: BF88FF9E
	v_mov_b32_e32 v13, s4                                      // 000000008B44: 7E1A0204
	ds_load_2addr_b32 v[9:10], v5 offset1:1                    // 000000008B48: D8DC0100 09000005
	ds_load_b32 v5, v5 offset:8                                // 000000008B50: D8D80008 05000005
	ds_load_2addr_b32 v[11:12], v13 offset1:1                  // 000000008B58: D8DC0100 0B00000D
	ds_load_b32 v13, v13 offset:8                              // 000000008B60: D8D80008 0D00000D
	s_wait_dscnt 0x1                                           // 000000008B68: BFC60001
	v_dual_fmac_f32 v0, v9, v11 :: v_dual_fmac_f32 v1, v10, v12// 000000008B6C: C8001709 0000190A
	s_wait_dscnt 0x0                                           // 000000008B74: BFC60000
	v_fmac_f32_e32 v2, v5, v13                                 // 000000008B78: 56041B05
	s_cbranch_scc0 L58                                         // 000000008B7C: BFA1FFE9
	s_branch L56                                               // 000000008B80: BFA0FFD2

0000000000008b84 <L59>:
	v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v1, 0               // 000000008B84: CA100080 00000080
	v_mov_b32_e32 v2, 0                                        // 000000008B8C: 7E040280

0000000000008b90 <L60>:
	s_wait_kmcnt 0x0                                           // 000000008B90: BFC70000
	v_mul_lo_u32 v3, s10, v8                                   // 000000008B94: D72C0003 0202100A
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 000000008B9C: BF8700B1
	v_add_co_u32 v3, s0, s8, v3                                // 000000008BA0: D7000003 02020608
	s_wait_alu depctr_va_sdst(0)                               // 000000008BA8: BF88F19F
	v_add_co_ci_u32_e64 v4, null, s9, 0, s0                    // 000000008BAC: D5207C04 00010009
	v_mad_co_i64_i32 v[3:4], null, v7, 12, v[3:4]              // 000000008BB4: D6FF7C03 040D1907
	global_store_b96 v[3:4], v[0:2], off                       // 000000008BBC: EE07007C 00000000 00000003

0000000000008bc8 <L61>:
	s_endpgm                                                   // 000000008BC8: BFB00000
		...

0000000000008c00 <_ZN2fk25launchTransformDPP_KernelILNS_7ParArchE3ELNS_2TFE0ELb1ENS_20TransformDPPDetails_IvLb0EJNS_29ReadBackInstantiableOperationINS_14ResizeCompleteILNS_11AspectRatioE1ENS_28TernaryInstantiableOperationINS_19InterpolateCompleteILNS_17InterpolationTypeE1ENS_25ReadInstantiableOperationINS_13PerThreadReadILNS_2NDE2ENS_6Uchar3EEEEEEEEELNS_14ResizeSamplingE1EEEEENS_26WriteInstantiableOperationINS_14PerThreadWriteILSC_2ENS_6Float3EEEEEEEEJSK_SP_EEEvT2_DpKT3_>:
	s_clause 0x1                                               // 000000008C00: BF850001
	s_load_b32 s4, s[0:1], 0x5c                                // 000000008C04: F4000100 F800005C
	s_load_b64 s[2:3], s[0:1], 0x8                             // 000000008C0C: F4002080 F8000008
	v_and_b32_e32 v1, 0x3ff, v0                                // 000000008C14: 360200FF 000003FF
	v_bfe_u32 v2, v0, 10, 10                                   // 000000008C1C: D6100002 02291500
	s_wait_kmcnt 0x0                                           // 000000008C24: BFC70000
	s_and_b32 s5, s4, 0xffff                                   // 000000008C28: 8B05FF04 0000FFFF
	s_lshr_b32 s4, s4, 16                                      // 000000008C30: 85049004
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000008C34: BF870121
	v_mad_co_u64_u32 v[0:1], null, ttmp9, s5, v[1:2]           // 000000008C38: D6FE7C00 04040A75
	v_mad_co_u64_u32 v[1:2], null, ttmp7, s4, v[2:3]           // 000000008C40: D6FE7C01 04080873
	v_cmp_gt_u32_e32 vcc_lo, s2, v0                            // 000000008C48: 7C980002
	s_delay_alu instid0(VALU_DEP_2)                            // 000000008C4C: BF870002
	v_cmp_gt_u32_e64 s2, s3, v1                                // 000000008C50: D44C0002 02020203
	s_and_b32 s2, vcc_lo, s2                                   // 000000008C58: 8B02026A
	s_wait_alu depctr_sa_sdst(0)                               // 000000008C5C: BF88FF9E
	s_and_saveexec_b32 s3, s2                                  // 000000008C60: BE832002
	s_cbranch_execz L62                                        // 000000008C64: BFA50099
	s_clause 0x1                                               // 000000008C68: BF850001
	s_load_b64 s[2:3], s[0:1], 0x10                            // 000000008C6C: F4002080 F8000010
	s_load_b64 s[8:9], s[0:1], 0x20                            // 000000008C74: F4002200 F8000020
	v_cvt_f32_i32_e32 v2, v1                                   // 000000008C7C: 7E040B01
	v_cvt_f32_i32_e32 v3, v0                                   // 000000008C80: 7E060B00
	s_load_b96 s[4:6], s[0:1], 0x28                            // 000000008C84: F400A100 F8000028
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000008C8C: BF8700A1
	v_dual_add_f32 v2, 0.5, v2 :: v_dual_add_f32 v3, 0.5, v3   // 000000008C90: C90804F0 020206F0
	s_wait_kmcnt 0x0                                           // 000000008C98: BFC70000
	v_fma_f32 v2, v2, s3, -0.5                                 // 000000008C9C: D6130002 03C40702
	s_delay_alu instid0(VALU_DEP_1)                            // 000000008CA4: BF870001
	v_cmp_le_f32_e32 vcc_lo, 0, v2                             // 000000008CA8: 7C260480
	v_cndmask_b32_e32 v10, 0, v2, vcc_lo                       // 000000008CAC: 02140480
	v_fma_f32 v2, v3, s2, -0.5                                 // 000000008CB0: D6130002 03C40503
	s_clause 0x1                                               // 000000008CB8: BF850001
	s_load_b64 s[2:3], s[0:1], 0x38                            // 000000008CBC: F4002080 F8000038
	s_load_b32 s0, s[0:1], 0x48                                // 000000008CC4: F4000000 F8000048
	s_add_co_i32 s1, s5, -1                                    // 000000008CCC: 8101C105
	v_cmp_le_f32_e32 vcc_lo, 0, v2                             // 000000008CD0: 7C260480
	s_wait_alu depctr_va_vcc(0)                                // 000000008CD4: BF88FF9D
	v_cndmask_b32_e32 v11, 0, v2, vcc_lo                       // 000000008CD8: 02160480
	v_floor_f32_e32 v3, v10                                    // 000000008CDC: 7E06490A
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000008CE0: BF870112
	v_floor_f32_e32 v2, v11                                    // 000000008CE4: 7E04490B
	v_cvt_i32_f32_e32 v12, v3                                  // 000000008CE8: 7E181103
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000008CEC: BF870112
	v_cvt_i32_f32_e32 v14, v2                                  // 000000008CF0: 7E1C1102
	v_add_nc_u32_e32 v13, 1, v12                               // 000000008CF4: 4A1A1881
	v_mul_lo_u32 v3, s6, v12                                   // 000000008CF8: D72C0003 02021806
	s_wait_kmcnt 0x0                                           // 000000008D00: BFC70000
	v_mul_lo_u32 v1, s0, v1                                    // 000000008D04: D72C0001 02020200
	v_add_nc_u32_e32 v15, 1, v14                               // 000000008D0C: 4A1E1C81
	s_wait_alu depctr_sa_sdst(0)                               // 000000008D10: BF88FF9E
	v_min_i32_e32 v2, s1, v13                                  // 000000008D14: 22041A01
	s_add_co_i32 s1, s4, -1                                    // 000000008D18: 8101C104
	s_wait_alu depctr_sa_sdst(0)                               // 000000008D1C: BF88FF9E
	v_min_i32_e32 v8, s1, v15                                  // 000000008D20: 22101E01
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 000000008D24: BF8700C2
	v_mul_lo_u32 v4, v2, s6                                    // 000000008D28: D72C0004 02000D02
	v_add_co_u32 v2, s1, s8, v3                                // 000000008D30: D7000102 02020608
	s_wait_alu depctr_va_sdst(0)                               // 000000008D38: BF88F19F
	v_add_co_ci_u32_e64 v3, null, s9, 0, s1                    // 000000008D3C: D5207C03 00050009
	v_mad_co_i64_i32 v[6:7], null, v14, 3, v[2:3]              // 000000008D44: D6FF7C06 0409070E
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_2)// 000000008D4C: BF870144
	v_add_co_u32 v4, s1, s8, v4                                // 000000008D50: D7000104 02020808
	s_wait_alu depctr_va_sdst(0)                               // 000000008D58: BF88F19F
	v_add_co_ci_u32_e64 v5, null, s9, 0, s1                    // 000000008D5C: D5207C05 00050009
	v_mad_co_i64_i32 v[2:3], null, v8, 3, v[2:3]               // 000000008D64: D6FF7C02 04090708
	v_mad_co_i64_i32 v[8:9], null, v8, 3, v[4:5]               // 000000008D6C: D6FF7C08 04110708
	v_mad_co_i64_i32 v[4:5], null, v14, 3, v[4:5]              // 000000008D74: D6FF7C04 0411070E
	s_clause 0x7                                               // 000000008D7C: BF850007
	global_load_u8 v16, v[6:7], off offset:2                   // 000000008D80: EE04007C 00000010 00000206
	global_load_u8 v17, v[2:3], off offset:2                   // 000000008D8C: EE04007C 00000011 00000202
	global_load_u16 v18, v[8:9], off                           // 000000008D98: EE04807C 00000012 00000008
	global_load_u16 v19, v[4:5], off                           // 000000008DA4: EE04807C 00000013 00000004
	global_load_u16 v2, v[2:3], off                            // 000000008DB0: EE04807C 00000002 00000002
	global_load_u16 v6, v[6:7], off                            // 000000008DBC: EE04807C 00000006 00000006
	global_load_u8 v7, v[4:5], off offset:2                    // 000000008DC8: EE04007C 00000007 00000204
	global_load_u8 v8, v[8:9], off offset:2                    // 000000008DD4: EE04007C 00000008 00000208
	v_cvt_f32_i32_e32 v4, v12                                  // 000000008DE0: 7E080B0C
	v_cvt_f32_i32_e32 v3, v14                                  // 000000008DE4: 7E060B0E
	v_cvt_f32_i32_e32 v9, v13                                  // 000000008DE8: 7E120B0D
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 000000008DEC: BF870223
	v_sub_f32_e32 v12, v10, v4                                 // 000000008DF0: 0818090A
	v_cvt_f32_i32_e32 v5, v15                                  // 000000008DF4: 7E0A0B0F
	v_sub_f32_e32 v3, v11, v3                                  // 000000008DF8: 0806070B
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 000000008DFC: BF870224
	v_sub_f32_e32 v9, v9, v10                                  // 000000008E00: 08121509
	v_add_co_u32 v4, s0, s2, v1                                // 000000008E04: D7000004 02020202
	v_sub_f32_e32 v11, v5, v11                                 // 000000008E0C: 08161705
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)// 000000008E10: BF8701B4
	v_mul_f32_e32 v13, v12, v3                                 // 000000008E14: 101A070C
	s_wait_alu depctr_va_sdst(0)                               // 000000008E18: BF88F19F
	v_add_co_ci_u32_e64 v5, null, s3, 0, s0                    // 000000008E1C: D5207C05 00010003
	v_dual_mul_f32 v10, v9, v11 :: v_dual_mul_f32 v9, v3, v9   // 000000008E24: C8C61709 0A081303
	v_mul_f32_e32 v11, v12, v11                                // 000000008E2C: 1016170C
	s_delay_alu instid0(VALU_DEP_3)                            // 000000008E30: BF870003
	v_mad_co_i64_i32 v[4:5], null, v0, 12, v[4:5]              // 000000008E34: D6FF7C04 04111900
	s_wait_loadcnt 0x7                                         // 000000008E3C: BFC00007
	v_cvt_f32_ubyte0_e32 v1, v16                               // 000000008E40: 7E022310
	s_wait_loadcnt 0x6                                         // 000000008E44: BFC00006
	v_cvt_f32_ubyte0_e32 v12, v17                              // 000000008E48: 7E182311
	s_wait_loadcnt 0x3                                         // 000000008E4C: BFC00003
	v_cvt_f32_ubyte1_e32 v15, v2                               // 000000008E50: 7E1E2502
	v_mul_f32_e32 v3, v10, v1                                  // 000000008E54: 1006030A
	s_wait_loadcnt 0x2                                         // 000000008E58: BFC00002
	v_cvt_f32_ubyte0_e32 v1, v6                                // 000000008E5C: 7E022306
	v_cvt_f32_ubyte1_e32 v6, v6                                // 000000008E60: 7E0C2506
	v_cvt_f32_ubyte0_e32 v14, v2                               // 000000008E64: 7E1C2302
	s_wait_loadcnt 0x1                                         // 000000008E68: BFC00001
	v_cvt_f32_ubyte0_e32 v7, v7                                // 000000008E6C: 7E0E2307
	s_wait_loadcnt 0x0                                         // 000000008E70: BFC00000
	v_cvt_f32_ubyte0_e32 v8, v8                                // 000000008E74: 7E102308
	v_dual_mul_f32 v1, v10, v1 :: v_dual_mul_f32 v2, v10, v6   // 000000008E78: C8C6030A 01020D0A
	v_cvt_f32_ubyte1_e32 v10, v19                              // 000000008E80: 7E142513
	v_cvt_f32_ubyte0_e32 v6, v19                               // 000000008E84: 7E0C2313
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000008E88: BF870123
	v_dual_fmac_f32 v2, v9, v15 :: v_dual_fmac_f32 v3, v9, v12 // 000000008E8C: C8001F09 02021909
	v_cvt_f32_ubyte1_e32 v12, v18                              // 000000008E94: 7E182512
	v_dual_fmac_f32 v2, v11, v10 :: v_dual_fmac_f32 v3, v11, v7// 000000008E98: C800150B 02020F0B
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_4)// 000000008EA0: BF870231
	v_fmac_f32_e32 v2, v13, v12                                // 000000008EA4: 5604190D
	v_fmac_f32_e32 v1, v9, v14                                 // 000000008EA8: 56021D09
	v_cvt_f32_ubyte0_e32 v9, v18                               // 000000008EAC: 7E122312
	v_fmac_f32_e32 v3, v13, v8                                 // 000000008EB0: 5606110D
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000008EB4: BF870093
	v_fmac_f32_e32 v1, v11, v6                                 // 000000008EB8: 56020D0B
	v_fmac_f32_e32 v1, v13, v9                                 // 000000008EBC: 5602130D
	global_store_b96 v[4:5], v[1:3], off                       // 000000008EC0: EE07007C 00800000 00000004

0000000000008ecc <L62>:
	s_endpgm                                                   // 000000008ECC: BFB00000
		...

0000000000008f00 <_ZN2fk21linearFilterDPPKernelILNS_7ParArchE3ENS_22LinearFilterDPPDetailsINS_6Float3ELi16ELi8ELi7ELi7EEENS_5TupleIJNS_29ReadBackInstantiableOperationINS_14ResizeCompleteILNS_11AspectRatioE1ENS_28TernaryInstantiableOperationINS_19InterpolateCompleteILNS_17InterpolationTypeE1ENS_25ReadInstantiableOperationINS_13PerThreadReadILNS_2NDE2ENS_6Uchar3EEEEEEEEELNS_14ResizeSamplingE1EEEEENSC_INSD_ILSE_2ES3_EEEEEEENS_26UnaryInstantiableOperationINS_3MulIS3_S3_S3_NS_9UnaryTypeEEEEENSQ_INS_3AddIS3_S3_S3_SS_EEEENS_26WriteInstantiableOperationINS_14PerThreadWriteILSE_2ES3_EEEEEEvT0_T1_T2_T3_T4_>:
	s_clause 0x3                                               // 000000008F00: BF850003
	s_load_b32 s10, s[0:1], 0x8c                               // 000000008F04: F4000280 F800008C
	s_load_b128 s[4:7], s[0:1], 0x0                            // 000000008F0C: F4004100 F8000000
	s_load_b64 s[8:9], s[0:1], 0x68                            // 000000008F14: F4002200 F8000068
	s_load_b64 s[2:3], s[0:1], 0x48                            // 000000008F1C: F4002080 F8000048
	v_bfe_u32 v6, v0, 10, 10                                   // 000000008F24: D6100006 02291500
	v_and_b32_e32 v0, 0x3ff, v0                                // 000000008F2C: 360000FF 000003FF
	s_lshl_b32 s15, ttmp9, 4                                   // 000000008F34: 840F8475
	s_lshl_b32 s16, ttmp7, 3                                   // 000000008F38: 84108373
	s_mov_b32 s19, exec_lo                                     // 000000008F3C: BE93007E
	s_wait_kmcnt 0x0                                           // 000000008F40: BFC70000
	s_and_b32 s11, s10, 0xffff                                 // 000000008F44: 8B0BFF0A 0000FFFF
	s_add_co_i32 s14, s6, 15                                   // 000000008F4C: 810E8F06
	v_mad_u32_u24 v1, v6, s11, v0                              // 000000008F50: D60B0001 04001706
	s_add_co_i32 s18, s7, 7                                    // 000000008F58: 81128707
	s_lshr_b32 s17, s10, 16                                    // 000000008F5C: 8511900A
	s_mul_i32 s18, s18, s14                                    // 000000008F60: 96120E12
	s_mul_i32 s17, s17, s11                                    // 000000008F64: 96110B11
	v_cmpx_gt_i32_e32 s18, v1                                  // 000000008F68: 7D880212
	s_cbranch_execz L64                                        // 000000008F6C: BFA500DD
	s_abs_i32 s20, s14                                         // 000000008F70: BE94150E
	s_add_nc_u64 s[22:23], s[0:1], 24                          // 000000008F74: A9969800
	s_cvt_f32_u32 s10, s20                                     // 000000008F78: BE8A6514
	s_clause 0x1                                               // 000000008F7C: BF850001
	s_load_b64 s[28:29], s[0:1], 0x10                          // 000000008F80: F4002700 F8000010
	s_load_b64 s[30:31], s[0:1], 0x38                          // 000000008F88: F4002780 F8000038
	v_mul_lo_u32 v2, v1, 12                                    // 000000008F90: D72C0002 02011901
	v_s_rcp_f32 s26, s10                                       // 000000008F98: D684001A 0201000A
	s_clause 0x2                                               // 000000008FA0: BF850002
	s_load_b64 s[10:11], s[22:23], 0x8                         // 000000008FA4: F400228B F8000008
	s_load_b64 s[12:13], s[22:23], 0x18                        // 000000008FAC: F400230B F8000018
	s_load_b32 s21, s[22:23], 0x28                             // 000000008FB4: F400054B F8000028
	s_sub_co_i32 s34, 0, s20                                   // 000000008FBC: 81A21480
	v_mov_b32_e32 v3, v1                                       // 000000008FC0: 7E060301
	s_add_co_i32 s22, s4, -1                                   // 000000008FC4: 8116C104
	s_add_co_i32 s23, s5, -1                                   // 000000008FC8: 8117C105
	s_ashr_i32 s24, s14, 31                                    // 000000008FCC: 86189F0E
	s_mov_b32 s25, 0                                           // 000000008FD0: BE990080
	s_mul_f32 s27, s26, 0x4f7ffffe                             // 000000008FD4: A21BFF1A 4F7FFFFE
	s_sub_co_i32 s26, 0, s14                                   // 000000008FDC: 819A0E80
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(SALU_CYCLE_3)// 000000008FE0: BF87059A
	s_cvt_u32_f32 s33, s27                                     // 000000008FE4: BEA1671B
	s_mul_i32 s34, s34, s33                                    // 000000008FE8: 96222122
	s_wait_kmcnt 0x0                                           // 000000008FEC: BFC70000
	s_sub_co_i32 s27, s15, s28                                 // 000000008FF0: 819B1C0F
	s_mul_hi_u32 s34, s33, s34                                 // 000000008FF4: 96A22221
	s_sub_co_i32 s28, s16, s29                                 // 000000008FF8: 819C1D10
	s_add_co_i32 s29, s30, -1                                  // 000000008FFC: 811DC11E
	s_add_co_i32 s30, s31, -1                                  // 000000009000: 811EC11F
	s_add_co_i32 s31, s33, s34                                 // 000000009004: 811F2221
	s_mul_i32 s33, s17, 12                                     // 000000009008: 96218C11

000000000000900c <L63>:
	v_sub_nc_u32_e32 v4, 0, v3                                 // 00000000900C: 4C080680
	v_ashrrev_i32_e32 v5, 31, v3                               // 000000009010: 340A069F
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000009014: BF870112
	v_max_i32_e32 v4, v4, v3                                   // 000000009018: 24080704
	v_xor_b32_e32 v5, s24, v5                                  // 00000000901C: 3A0A0A18
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000009020: BF870092
	v_mul_hi_u32 v7, v4, s31                                   // 000000009024: D72D0007 02003F04
	v_mul_lo_u32 v8, v7, s20                                   // 00000000902C: D72C0008 02002907
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000009034: BF870091
	v_sub_nc_u32_e32 v4, v4, v8                                // 000000009038: 4C081104
	v_subrev_nc_u32_e32 v8, s20, v4                            // 00000000903C: 4E100814
	v_cmp_le_u32_e32 vcc_lo, s20, v4                           // 000000009040: 7C960814
	s_wait_alu depctr_va_vcc(0)                                // 000000009044: BF88FF9D
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000009048: BF870092
	v_dual_cndmask_b32 v4, v4, v8 :: v_dual_add_nc_u32 v9, 1, v7// 00000000904C: CA601104 04080E81
	v_cndmask_b32_e32 v7, v7, v9, vcc_lo                       // 000000009054: 020E1307
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000009058: BF870112
	v_cmp_le_u32_e32 vcc_lo, s20, v4                           // 00000000905C: 7C960814
	v_add_nc_u32_e32 v8, 1, v7                                 // 000000009060: 4A100E81
	s_wait_alu depctr_va_vcc(0)                                // 000000009064: BF88FF9D
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000009068: BF870091
	v_cndmask_b32_e32 v4, v7, v8, vcc_lo                       // 00000000906C: 02081107
	v_xor_b32_e32 v4, v4, v5                                   // 000000009070: 3A080B04
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000009074: BF870091
	v_sub_nc_u32_e32 v4, v4, v5                                // 000000009078: 4C080B04
	v_mul_lo_u32 v5, s26, v4                                   // 00000000907C: D72C0005 0202081A
	v_add_nc_u32_e32 v4, s28, v4                               // 000000009084: 4A08081C
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 000000009088: BF870221
	v_min_i32_e32 v7, s23, v4                                  // 00000000908C: 220E0817
	v_cmp_lt_i32_e32 vcc_lo, -1, v4                            // 000000009090: 7C8208C1
	v_add3_u32 v4, s27, v3, v5                                 // 000000009094: D6550004 0416061B
	s_wait_alu depctr_va_vcc(0)                                // 00000000909C: BF88FF9D
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000090A0: BF870113
	v_cndmask_b32_e32 v5, 0, v7, vcc_lo                        // 0000000090A4: 020A0E80
	v_min_i32_e32 v7, s22, v4                                  // 0000000090A8: 220E0816
	v_cmp_lt_i32_e32 vcc_lo, -1, v4                            // 0000000090AC: 7C8208C1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)// 0000000090B0: BF8701A3
	v_cvt_f32_i32_e32 v4, v5                                   // 0000000090B4: 7E080B05
	s_wait_alu depctr_va_vcc(0)                                // 0000000090B8: BF88FF9D
	v_cndmask_b32_e32 v5, 0, v7, vcc_lo                        // 0000000090BC: 020A0E80
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000090C0: BF870091
	v_cvt_f32_i32_e32 v5, v5                                   // 0000000090C4: 7E0A0B05
	v_dual_add_f32 v4, 0.5, v4 :: v_dual_add_f32 v5, 0.5, v5   // 0000000090C8: C90808F0 04040AF0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000090D0: BF870091
	v_fma_f32 v4, s11, v4, -0.5                                // 0000000090D4: D6130004 03C6080B
	v_cmp_le_f32_e32 vcc_lo, 0, v4                             // 0000000090DC: 7C260880
	s_wait_alu depctr_va_vcc(0)                                // 0000000090E0: BF88FF9D
	v_cndmask_b32_e32 v13, 0, v4, vcc_lo                       // 0000000090E4: 021A0880
	v_fma_f32 v4, s10, v5, -0.5                                // 0000000090E8: D6130004 03C60A0A
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)// 0000000090F0: BF870141
	v_cmp_le_f32_e32 vcc_lo, 0, v4                             // 0000000090F4: 7C260880
	s_wait_alu depctr_va_vcc(0)                                // 0000000090F8: BF88FF9D
	v_cndmask_b32_e32 v14, 0, v4, vcc_lo                       // 0000000090FC: 021C0880
	v_floor_f32_e32 v5, v13                                    // 000000009100: 7E0A490D
	v_floor_f32_e32 v4, v14                                    // 000000009104: 7E08490E
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000009108: BF870112
	v_cvt_i32_f32_e32 v15, v5                                  // 00000000910C: 7E1E1105
	v_cvt_i32_f32_e32 v17, v4                                  // 000000009110: 7E221104
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)// 000000009114: BF8701A2
	v_add_nc_u32_e32 v16, 1, v15                               // 000000009118: 4A201E81
	v_mul_lo_u32 v5, s21, v15                                  // 00000000911C: D72C0005 02021E15
	v_add_nc_u32_e32 v18, 1, v17                               // 000000009124: 4A242281
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000009128: BF870193
	v_min_i32_e32 v7, s30, v16                                 // 00000000912C: 220E201E
	v_add_co_u32 v4, s34, s12, v5                              // 000000009130: D7002204 02020A0C
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)// 000000009138: BF870142
	v_mul_lo_u32 v9, v7, s21                                   // 00000000913C: D72C0009 02002B07
	s_wait_alu depctr_va_sdst(0)                               // 000000009144: BF88F19F
	v_add_co_ci_u32_e64 v5, null, s13, 0, s34                  // 000000009148: D5207C05 0089000D
	v_min_i32_e32 v11, s29, v18                                // 000000009150: 2216241D
	v_mad_co_i64_i32 v[7:8], null, v17, 3, v[4:5]              // 000000009154: D6FF7C07 04110711
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000915C: BF870004
	v_add_co_u32 v9, s34, s12, v9                              // 000000009160: D7002209 0202120C
	s_wait_alu depctr_va_sdst(0)                               // 000000009168: BF88F19F
	v_add_co_ci_u32_e64 v10, null, s13, 0, s34                 // 00000000916C: D5207C0A 0089000D
	v_mad_co_i64_i32 v[4:5], null, v11, 3, v[4:5]              // 000000009174: D6FF7C04 0411070B
	global_load_u8 v19, v[7:8], off offset:2                   // 00000000917C: EE04007C 00000013 00000207
	v_mad_co_i64_i32 v[11:12], null, v11, 3, v[9:10]           // 000000009188: D6FF7C0B 0425070B
	v_mad_co_i64_i32 v[9:10], null, v17, 3, v[9:10]            // 000000009190: D6FF7C09 04250711
	s_clause 0x6                                               // 000000009198: BF850006
	global_load_u8 v20, v[4:5], off offset:2                   // 00000000919C: EE04007C 00000014 00000204
	global_load_u16 v21, v[11:12], off                         // 0000000091A8: EE04807C 00000015 0000000B
	global_load_u16 v22, v[9:10], off                          // 0000000091B4: EE04807C 00000016 00000009
	global_load_u16 v4, v[4:5], off                            // 0000000091C0: EE04807C 00000004 00000004
	global_load_u16 v5, v[7:8], off                            // 0000000091CC: EE04807C 00000005 00000007
	global_load_u8 v7, v[9:10], off offset:2                   // 0000000091D8: EE04007C 00000007 00000209
	global_load_u8 v8, v[11:12], off offset:2                  // 0000000091E4: EE04007C 00000008 0000020B
	v_cvt_f32_i32_e32 v9, v15                                  // 0000000091F0: 7E120B0F
	v_cvt_f32_i32_e32 v11, v17                                 // 0000000091F4: 7E160B11
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)// 0000000091F8: BF8701A2
	v_sub_f32_e32 v9, v13, v9                                  // 0000000091FC: 0812130D
	v_cvt_f32_i32_e32 v10, v16                                 // 000000009200: 7E140B10
	v_sub_f32_e32 v11, v14, v11                                // 000000009204: 0816170E
	v_cvt_f32_i32_e32 v12, v18                                 // 000000009208: 7E180B12
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000920C: BF870113
	v_dual_sub_f32 v10, v10, v13 :: v_dual_add_nc_u32 v3, s17, v3// 000000009210: C9601B0A 0A020611
	v_sub_f32_e32 v12, v12, v14                                // 000000009218: 08181D0C
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)// 00000000921C: BF870192
	v_cmp_le_i32_e32 vcc_lo, s18, v3                           // 000000009220: 7C860612
	v_mul_f32_e32 v13, v11, v10                                // 000000009224: 101A150B
	s_delay_alu instid0(VALU_DEP_3)                            // 000000009228: BF870003
	v_dual_mul_f32 v11, v9, v11 :: v_dual_mul_f32 v10, v10, v12// 00000000922C: C8C61709 0B0A190A
	v_mul_f32_e32 v9, v9, v12                                  // 000000009234: 10121909
	s_or_b32 s25, vcc_lo, s25                                  // 000000009238: 8C19196A
	s_wait_loadcnt 0x7                                         // 00000000923C: BFC00007
	v_cvt_f32_ubyte0_e32 v12, v19                              // 000000009240: 7E182313
	s_wait_loadcnt 0x6                                         // 000000009244: BFC00006
	v_cvt_f32_ubyte0_e32 v14, v20                              // 000000009248: 7E1C2314
	s_wait_loadcnt 0x5                                         // 00000000924C: BFC00005
	v_cvt_f32_ubyte0_e32 v18, v21                              // 000000009250: 7E242315
	s_wait_loadcnt 0x4                                         // 000000009254: BFC00004
	v_cvt_f32_ubyte1_e32 v17, v22                              // 000000009258: 7E222516
	v_mul_f32_e32 v12, v10, v12                                // 00000000925C: 1018190A
	s_wait_loadcnt 0x2                                         // 000000009260: BFC00002
	v_cvt_f32_ubyte0_e32 v15, v5                               // 000000009264: 7E1E2305
	v_cvt_f32_ubyte1_e32 v5, v5                                // 000000009268: 7E0A2505
	v_cvt_f32_ubyte0_e32 v16, v4                               // 00000000926C: 7E202304
	v_cvt_f32_ubyte1_e32 v4, v4                                // 000000009270: 7E082504
	s_delay_alu instid0(VALU_DEP_4)                            // 000000009274: BF870004
	v_dual_fmac_f32 v12, v13, v14 :: v_dual_mul_f32 v15, v10, v15// 000000009278: C8061D0D 0C0E1F0A
	s_wait_loadcnt 0x1                                         // 000000009280: BFC00001
	v_cvt_f32_ubyte0_e32 v7, v7                                // 000000009284: 7E0E2307
	v_mul_f32_e32 v5, v10, v5                                  // 000000009288: 100A0B0A
	v_cvt_f32_ubyte0_e32 v14, v22                              // 00000000928C: 7E1C2316
	v_cvt_f32_ubyte1_e32 v19, v21                              // 000000009290: 7E262515
	v_fmac_f32_e32 v15, v13, v16                               // 000000009294: 561E210D
	v_fmac_f32_e32 v12, v9, v7                                 // 000000009298: 56180F09
	v_fmac_f32_e32 v5, v13, v4                                 // 00000000929C: 560A090D
	s_wait_loadcnt 0x0                                         // 0000000092A0: BFC00000
	v_cvt_f32_ubyte0_e32 v8, v8                                // 0000000092A4: 7E102308
	v_fmac_f32_e32 v15, v9, v14                                // 0000000092A8: 561E1D09
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000092AC: BF870112
	v_dual_fmac_f32 v5, v9, v17 :: v_dual_fmac_f32 v12, v11, v8// 0000000092B0: C8002309 050C110B
	v_fmac_f32_e32 v15, v11, v18                               // 0000000092B8: 561E250B
	s_delay_alu instid0(VALU_DEP_2)                            // 0000000092BC: BF870002
	v_fmac_f32_e32 v5, v11, v19                                // 0000000092C0: 560A270B
	ds_store_2addr_b32 v2, v15, v5 offset1:1                   // 0000000092C4: D8380100 00050F02
	ds_store_b32 v2, v12 offset:8                              // 0000000092CC: D8340008 00000C02
	v_add_nc_u32_e32 v2, s33, v2                               // 0000000092D4: 4A040421
	s_wait_alu depctr_sa_sdst(0)                               // 0000000092D8: BF88FF9E
	s_and_not1_b32 exec_lo, exec_lo, s25                       // 0000000092DC: 917E197E
	s_cbranch_execnz L63                                       // 0000000092E0: BFA6FF4A

00000000000092e4 <L64>:
	s_or_b32 exec_lo, exec_lo, s19                             // 0000000092E4: 8C7E137E
	s_clause 0x1                                               // 0000000092E8: BF850001
	s_load_b32 s11, s[0:1], 0x58                               // 0000000092EC: F40002C0 F8000058
	s_load_b32 s10, s[0:1], 0x78                               // 0000000092F4: F4000280 F8000078
	s_mul_i32 s0, s7, s6                                       // 0000000092FC: 96000607
	s_mov_b32 s1, exec_lo                                      // 000000009300: BE81007E
	v_cmpx_gt_i32_e32 s0, v1                                   // 000000009304: 7D880200
	s_cbranch_execz L66                                        // 000000009308: BFA50054
	s_abs_i32 s12, s6                                          // 00000000930C: BE8C1506
	s_wait_alu depctr_sa_sdst(0)                               // 000000009310: BF88FF9E
	s_cvt_f32_u32 s13, s12                                     // 000000009314: BE8D650C
	s_wait_alu depctr_sa_sdst(0)                               // 000000009318: BF88FF9E
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)// 00000000931C: BF87029A
	v_s_rcp_f32 s13, s13                                       // 000000009320: D684000D 0201000D
	s_mul_f32 s13, s13, 0x4f7ffffe                             // 000000009328: A20DFF0D 4F7FFFFE
	s_wait_alu depctr_sa_sdst(0)                               // 000000009330: BF88FF9E
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000009334: BF8704BA
	s_cvt_u32_f32 s19, s13                                     // 000000009338: BE93670D
	s_sub_co_i32 s13, 0, s12                                   // 00000000933C: 818D0C80
	s_wait_alu depctr_sa_sdst(0)                               // 000000009340: BF88FF9E
	s_mul_i32 s18, s13, s19                                    // 000000009344: 9612130D
	s_ashr_i32 s13, s6, 31                                     // 000000009348: 860D9F06
	s_wait_alu depctr_sa_sdst(0)                               // 00000000934C: BF88FF9E
	s_mul_hi_u32 s20, s19, s18                                 // 000000009350: 96941213
	s_mov_b32 s18, 0                                           // 000000009354: BE920080
	s_wait_alu depctr_sa_sdst(0)                               // 000000009358: BF88FF9E
	s_add_co_i32 s19, s19, s20                                 // 00000000935C: 81131413

0000000000009360 <L65>:
	v_sub_nc_u32_e32 v2, 0, v1                                 // 000000009360: 4C040280
	v_ashrrev_i32_e32 v3, 31, v1                               // 000000009364: 3406029F
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000009368: BF870112
	v_max_i32_e32 v2, v2, v1                                   // 00000000936C: 24040302
	v_xor_b32_e32 v3, s13, v3                                  // 000000009370: 3A06060D
	s_wait_alu depctr_sa_sdst(0)                               // 000000009374: BF88FF9E
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000009378: BF870092
	v_mul_hi_u32 v4, v2, s19                                   // 00000000937C: D72D0004 02002702
	v_mul_lo_u32 v5, v4, s12                                   // 000000009384: D72C0005 02001904
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000938C: BF870091
	v_sub_nc_u32_e32 v2, v2, v5                                // 000000009390: 4C040B02
	v_subrev_nc_u32_e32 v5, s12, v2                            // 000000009394: 4E0A040C
	v_cmp_le_u32_e32 vcc_lo, s12, v2                           // 000000009398: 7C96040C
	s_wait_alu depctr_va_vcc(0)                                // 00000000939C: BF88FF9D
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000093A0: BF870092
	v_dual_cndmask_b32 v2, v2, v5 :: v_dual_add_nc_u32 v7, 1, v4// 0000000093A4: CA600B02 02060881
	v_cndmask_b32_e32 v4, v4, v7, vcc_lo                       // 0000000093AC: 02080F04
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000093B0: BF870112
	v_cmp_le_u32_e32 vcc_lo, s12, v2                           // 0000000093B4: 7C96040C
	v_add_nc_u32_e32 v5, 1, v4                                 // 0000000093B8: 4A0A0881
	s_wait_alu depctr_va_vcc(0)                                // 0000000093BC: BF88FF9D
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000093C0: BF8700A1
	v_cndmask_b32_e32 v2, v4, v5, vcc_lo                       // 0000000093C4: 02040B04
	v_mul_lo_u32 v5, v1, 12                                    // 0000000093C8: D72C0005 02011901
	v_add_nc_u32_e32 v7, 0xe70, v5                             // 0000000093D0: 4A0E0AFF 00000E70
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000093D8: BF870093
	v_xor_b32_e32 v2, v2, v3                                   // 0000000093DC: 3A040702
	v_sub_nc_u32_e32 v2, v2, v3                                // 0000000093E0: 4C040702
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 0000000093E4: BF870131
	v_mul_lo_u32 v3, v2, s6                                    // 0000000093E8: D72C0003 02000D02
	s_wait_kmcnt 0x0                                           // 0000000093F0: BFC70000
	v_mul_lo_u32 v2, v2, s11                                   // 0000000093F4: D72C0002 02001702
	v_sub_nc_u32_e32 v4, v1, v3                                // 0000000093FC: 4C080701
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)// 000000009400: BF870142
	v_add_co_u32 v2, s20, s2, v2                               // 000000009404: D7001402 02020402
	s_wait_alu depctr_va_sdst(0)                               // 00000000940C: BF88F19F
	v_add_co_ci_u32_e64 v3, null, s3, 0, s20                   // 000000009410: D5207C03 00510003
	v_add_nc_u32_e32 v1, s17, v1                               // 000000009418: 4A020211
	v_mad_co_u64_u32 v[2:3], null, v4, 12, v[2:3]              // 00000000941C: D6FE7C02 04091904
	s_delay_alu instid0(VALU_DEP_2)                            // 000000009424: BF870002
	v_cmp_le_i32_e32 vcc_lo, s0, v1                            // 000000009428: 7C860200
	s_or_b32 s18, vcc_lo, s18                                  // 00000000942C: 8C12126A
	global_load_b96 v[2:4], v[2:3], off                        // 000000009430: EE05807C 00000002 00000002
	s_wait_loadcnt 0x0                                         // 00000000943C: BFC00000
	ds_store_2addr_b32 v7, v2, v3 offset1:1                    // 000000009440: D8380100 00030207
	ds_store_b32 v5, v4 offset:3704                            // 000000009448: D8340E78 00000405
	s_wait_alu depctr_sa_sdst(0)                               // 000000009450: BF88FF9E
	s_and_not1_b32 exec_lo, exec_lo, s18                       // 000000009454: 917E127E
	s_cbranch_execnz L65                                       // 000000009458: BFA6FFC1

000000000000945c <L66>:
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000945C: 8C7E017E
	s_wait_dscnt 0x0                                           // 000000009460: BFC60000
	s_barrier_signal -1                                        // 000000009464: BE804EC1
	v_add_nc_u32_e32 v7, s15, v0                               // 000000009468: 4A0E000F
	v_add_nc_u32_e32 v8, s16, v6                               // 00000000946C: 4A100C10
	v_cmp_gt_u32_e32 vcc_lo, 16, v0                            // 000000009470: 7C980090
	v_cmp_gt_u32_e64 s0, 8, v6                                 // 000000009474: D44C0000 02020C88
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 00000000947C: BF870214
	v_cmp_gt_i32_e64 s1, s4, v7                                // 000000009480: D4440001 02020E04
	v_cmp_gt_i32_e64 s2, s5, v8                                // 000000009488: D4440002 02021005
	s_and_b32 s0, vcc_lo, s0                                   // 000000009490: 8B00006A
	s_wait_alu depctr_sa_sdst(0)                               // 000000009494: BF88FF9E
	s_and_b32 s0, s0, s1                                       // 000000009498: 8B000100
	s_wait_alu depctr_sa_sdst(0)                               // 00000000949C: BF88FF9E
	s_and_b32 s0, s0, s2                                       // 0000000094A0: 8B000200
	s_barrier_wait 0xffff                                      // 0000000094A4: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 0000000094A8: EE0AC07C 00040000 00000000
	s_wait_alu depctr_sa_sdst(0)                               // 0000000094B4: BF88FF9E
	s_and_saveexec_b32 s1, s0                                  // 0000000094B8: BE812000
	s_cbranch_execz L72                                        // 0000000094BC: BFA50049
	s_cmp_lt_i32 s7, 1                                         // 0000000094C0: BF048107
	s_cbranch_scc1 L70                                         // 0000000094C4: BFA20036
	v_mul_u32_u24_e32 v3, 12, v0                               // 0000000094C8: 1606008C
	v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v1, 0               // 0000000094CC: CA100080 00000080
	v_mov_b32_e32 v2, 0                                        // 0000000094D4: 7E040280
	s_cmp_gt_i32 s6, 0                                         // 0000000094D8: BF028006
	s_mov_b32 s0, 0                                            // 0000000094DC: BE800080
	s_cselect_b32 s1, -1, 0                                    // 0000000094E0: 980180C1
	s_branch L68                                               // 0000000094E4: BFA00004

00000000000094e8 <L67>:
	s_add_co_i32 s0, s0, 1                                     // 0000000094E8: 81008100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000094EC: BF88FF9E
	s_cmp_eq_u32 s0, s7                                        // 0000000094F0: BF060700
	s_cbranch_scc1 L71                                         // 0000000094F4: BFA2002D

00000000000094f8 <L68>:
	s_wait_alu depctr_sa_sdst(0)                               // 0000000094F8: BF88FF9E
	s_and_b32 s2, s1, exec_lo                                  // 0000000094FC: 8B027E01
	s_cselect_b32 s2, 1, 0                                     // 000000009500: 98028081
	s_wait_alu depctr_sa_sdst(0)                               // 000000009504: BF88FF9E
	s_cmp_lg_u32 s2, 1                                         // 000000009508: BF078102
	s_cbranch_scc1 L67                                         // 00000000950C: BFA2FFF6
	v_add_nc_u32_e32 v4, s0, v6                                // 000000009510: 4A080C00
	s_mul_i32 s2, s0, s6                                       // 000000009514: 96020600
	s_mov_b32 s3, 0                                            // 000000009518: BE830080
	s_wait_alu depctr_sa_sdst(0)                               // 00000000951C: BF88FF9E
	s_mul_i32 s2, s2, 12                                       // 000000009520: 96028C02
	s_wait_alu depctr_sa_sdst(0)                               // 000000009524: BF88FF9E
	s_addk_co_i32 s2, 0xe70                                    // 000000009528: B7820E70
	v_mul_lo_u32 v4, v4, s14                                   // 00000000952C: D72C0004 02001D04
	s_delay_alu instid0(VALU_DEP_1)                            // 000000009534: BF870001
	v_mad_co_u64_u32 v[4:5], null, v4, 12, v[3:4]              // 000000009538: D6FE7C04 040D1904

0000000000009540 <L69>:
	s_mul_i32 s4, s3, 12                                       // 000000009540: 96048C03
	s_add_co_i32 s3, s3, 1                                     // 000000009544: 81038103
	s_wait_alu depctr_sa_sdst(0)                               // 000000009548: BF88FF9E
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000954C: BF870001
	v_add_nc_u32_e32 v5, s4, v4                                // 000000009550: 4A0A0804
	s_add_co_i32 s4, s2, s4                                    // 000000009554: 81040402
	s_cmp_eq_u32 s3, s6                                        // 000000009558: BF060603
	s_wait_alu depctr_sa_sdst(0)                               // 00000000955C: BF88FF9E
	v_mov_b32_e32 v13, s4                                      // 000000009560: 7E1A0204
	ds_load_2addr_b32 v[9:10], v5 offset1:1                    // 000000009564: D8DC0100 09000005
	ds_load_b32 v5, v5 offset:8                                // 00000000956C: D8D80008 05000005
	ds_load_2addr_b32 v[11:12], v13 offset1:1                  // 000000009574: D8DC0100 0B00000D
	ds_load_b32 v13, v13 offset:8                              // 00000000957C: D8D80008 0D00000D
	s_wait_dscnt 0x1                                           // 000000009584: BFC60001
	v_dual_fmac_f32 v0, v9, v11 :: v_dual_fmac_f32 v1, v10, v12// 000000009588: C8001709 0000190A
	s_wait_dscnt 0x0                                           // 000000009590: BFC60000
	v_fmac_f32_e32 v2, v5, v13                                 // 000000009594: 56041B05
	s_cbranch_scc0 L69                                         // 000000009598: BFA1FFE9
	s_branch L67                                               // 00000000959C: BFA0FFD2

00000000000095a0 <L70>:
	v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v1, 0               // 0000000095A0: CA100080 00000080
	v_mov_b32_e32 v2, 0                                        // 0000000095A8: 7E040280

00000000000095ac <L71>:
	s_wait_kmcnt 0x0                                           // 0000000095AC: BFC70000
	v_mul_lo_u32 v3, s10, v8                                   // 0000000095B0: D72C0003 0202100A
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 0000000095B8: BF8700B1
	v_add_co_u32 v3, s0, s8, v3                                // 0000000095BC: D7000003 02020608
	s_wait_alu depctr_va_sdst(0)                               // 0000000095C4: BF88F19F
	v_add_co_ci_u32_e64 v4, null, s9, 0, s0                    // 0000000095C8: D5207C04 00010009
	v_mad_co_i64_i32 v[3:4], null, v7, 12, v[3:4]              // 0000000095D0: D6FF7C03 040D1907
	global_store_b96 v[3:4], v[0:2], off                       // 0000000095D8: EE07007C 00000000 00000003

00000000000095e4 <L72>:
	s_endpgm                                                   // 0000000095E4: BFB00000
