
e18_box_hip.0.hip-amdgcn-amd-amdhsa--gfx1200:	file format elf64-amdgpu
	.amdgcn_target "amdgpu-amd-amdhsa-unknown-gfx1200"

Disassembly of section .text:

0000000000004500 <__cxa_pure_virtual>:
	s_wait_loadcnt_dscnt 0x0                                   // 000000004500: BFC80000
	s_wait_expcnt 0x0                                          // 000000004504: BFC40000
	s_wait_samplecnt 0x0                                       // 000000004508: BFC20000
	s_wait_bvhcnt 0x0                                          // 00000000450C: BFC30000
	s_wait_kmcnt 0x0                                           // 000000004510: BFC70000
	s_trap 2                                                   // 000000004514: BF900002
		...

0000000000004580 <__cxa_deleted_virtual>:
	s_wait_loadcnt_dscnt 0x0                                   // 000000004580: BFC80000
	s_wait_expcnt 0x0                                          // 000000004584: BFC40000
	s_wait_samplecnt 0x0                                       // 000000004588: BFC20000
	s_wait_bvhcnt 0x0                                          // 00000000458C: BFC30000
	s_wait_kmcnt 0x0                                           // 000000004590: BFC70000
	s_trap 2                                                   // 000000004594: BF900002
		...

0000000000004600 <_ZN6e18box7gpuSpinEy>:
	s_load_b64 s[0:1], s[0:1], 0x0                             // 000000004600: F4002000 F8000000
	s_sendmsg_rtn_b64 s[2:3], sendmsg(MSG_RTN_GET_REALTIME)    // 000000004608: BE824D83

000000000000460c <L0>:
	s_sendmsg_rtn_b64 s[4:5], sendmsg(MSG_RTN_GET_REALTIME)    // 00000000460C: BE844D83
	s_wait_kmcnt 0x0                                           // 000000004610: BFC70000
	s_wait_alu depctr_sa_sdst(0)                               // 000000004614: BF88FF9E
	s_sub_nc_u64 s[4:5], s[4:5], s[2:3]                        // 000000004618: AA040204
	s_wait_alu depctr_sa_sdst(0)                               // 00000000461C: BF88FF9E
	v_cmp_lt_u64_e64 s4, s[4:5], s[0:1]                        // 000000004620: D4590004 02000004
	s_and_b32 vcc_lo, exec_lo, s4                              // 000000004628: 8B6A047E
	s_cbranch_vccnz L0                                         // 00000000462C: BFA4FFF7
	s_endpgm                                                   // 000000004630: BFB00000
		...

0000000000004700 <_ZN2fk21linearFilterDPPKernelILNS_7ParArchE3ENS_22LinearFilterDPPDetailsINS_6Float3ELi16ELi8ELi7ELi7EEENS_5TupleIJNS_25ReadInstantiableOperationINS_15FusedOperation_IvJNS_29ReadBackInstantiableOperationINS_14ResizeCompleteILNS_11AspectRatioE1ENS_28TernaryInstantiableOperationINS_19InterpolateCompleteILNS_17InterpolationTypeE1ENS6_INS_13PerThreadReadILNS_2NDE2ENS_6Uchar3EEEEEEEEELNS_14ResizeSamplingE1EEEEENS_26UnaryInstantiableOperationINS_12SaturateCastIS3_SG_EEEENSO_INS_4CastISG_S3_EEEEEEEEENS6_INS_18ConstantFilterReadIS3_EEEEEEENSO_INS_3MulIS3_S3_S3_NS_9UnaryTypeEEEEENSO_INS_3AddIS3_S3_S3_S12_EEEENS_26WriteInstantiableOperationINS7_IvJSR_NS18_INS_14PerThreadWriteILSF_2ESG_EEEEEEEEEEEvT0_T1_T2_T3_T4_>:
	s_clause 0x2                                               // 000000004700: BF850002
	s_load_b32 s2, s[0:1], 0x84                                // 000000004704: F4000080 F8000084
	s_load_b128 s[4:7], s[0:1], 0x0                            // 00000000470C: F4004100 F8000000
	s_load_b64 s[12:13], s[0:1], 0x60                          // 000000004714: F4002300 F8000060
	v_bfe_u32 v3, v0, 10, 10                                   // 00000000471C: D6100003 02291500
	v_and_b32_e32 v0, 0x3ff, v0                                // 000000004724: 360000FF 000003FF
	s_lshl_b32 s14, ttmp9, 4                                   // 00000000472C: 840E8475
	s_lshl_b32 s15, ttmp7, 3                                   // 000000004730: 840F8373
	s_mov_b32 s17, exec_lo                                     // 000000004734: BE91007E
	s_wait_kmcnt 0x0                                           // 000000004738: BFC70000
	s_and_b32 s3, s2, 0xffff                                   // 00000000473C: 8B03FF02 0000FFFF
	s_add_co_i32 s11, s6, 15                                   // 000000004744: 810B8F06
	v_mad_u32_u24 v1, v3, s3, v0                               // 000000004748: D60B0001 04000703
	s_add_co_i32 s10, s7, 7                                    // 000000004750: 810A8707
	s_lshr_b32 s16, s2, 16                                     // 000000004754: 85109002
	s_mul_i32 s10, s10, s11                                    // 000000004758: 960A0B0A
	s_mul_i32 s16, s16, s3                                     // 00000000475C: 96100310
	v_cmpx_gt_i32_e32 s10, v1                                  // 000000004760: 7D88020A
	s_cbranch_execz L18                                        // 000000004764: BFA501A5
	s_abs_i32 s18, s11                                         // 000000004768: BE92150B
	s_add_nc_u64 s[2:3], s[0:1], 24                            // 00000000476C: A9829800
	s_cvt_f32_u32 s8, s18                                      // 000000004770: BE886512
	s_clause 0x2                                               // 000000004774: BF850002
	s_load_b64 s[24:25], s[0:1], 0x10                          // 000000004778: F4002600 F8000010
	s_load_b64 s[26:27], s[0:1], 0x38                          // 000000004780: F4002680 F8000038
	s_load_b64 s[28:29], s[2:3], 0x8                           // 000000004788: F4002701 F8000008
	v_s_rcp_f32 s21, s8                                        // 000000004790: D6840015 02010008
	s_clause 0x1                                               // 000000004798: BF850001
	s_load_b64 s[8:9], s[2:3], 0x18                            // 00000000479C: F4002201 F8000018
	s_load_b32 s19, s[2:3], 0x28                               // 0000000047A4: F40004C1 F8000028
	s_sub_co_i32 s3, 0, s18                                    // 0000000047AC: 81831280
	v_mov_b32_e32 v2, v1                                       // 0000000047B0: 7E040301
	s_add_co_i32 s20, s4, -1                                   // 0000000047B4: 8114C104
	s_ashr_i32 s22, s11, 31                                    // 0000000047B8: 86169F0B
	s_mul_f32 s2, s21, 0x4f7ffffe                              // 0000000047BC: A202FF15 4F7FFFFE
	s_add_co_i32 s21, s5, -1                                   // 0000000047C4: 8115C105
	s_wait_alu depctr_sa_sdst(0)                               // 0000000047C8: BF88FF9E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 0000000047CC: BF870529
	s_cvt_u32_f32 s2, s2                                       // 0000000047D0: BE826702
	s_wait_alu depctr_sa_sdst(0)                               // 0000000047D4: BF88FF9E
	s_mul_i32 s3, s3, s2                                       // 0000000047D8: 96030203
	s_wait_kmcnt 0x0                                           // 0000000047DC: BFC70000
	s_sub_co_i32 s23, s14, s24                                 // 0000000047E0: 8197180E
	s_wait_alu depctr_sa_sdst(0)                               // 0000000047E4: BF88FF9E
	s_mul_hi_u32 s3, s2, s3                                    // 0000000047E8: 96830302
	s_sub_co_i32 s24, s15, s25                                 // 0000000047EC: 8198190F
	s_add_co_i32 s25, s26, -1                                  // 0000000047F0: 8119C11A
	s_add_co_i32 s26, s27, -1                                  // 0000000047F4: 811AC11B
	s_mov_b32 s27, s29                                         // 0000000047F8: BE9B001D
	s_mov_b32 s29, 0                                           // 0000000047FC: BE9D0080
	s_wait_alu depctr_sa_sdst(0)                               // 000000004800: BF88FF9E
	s_add_co_i32 s30, s2, s3                                   // 000000004804: 811E0302
	s_branch L7                                                // 000000004808: BFA0001C

000000000000480c <L1>:
	s_or_b32 exec_lo, exec_lo, s35                             // 00000000480C: 8C7E237E

0000000000004810 <L2>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000004810: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s3                              // 000000004814: 8C7E037E
	v_mov_b32_e32 v5, v6                                       // 000000004818: 7E0A0306

000000000000481c <L3>:
	s_wait_alu depctr_sa_sdst(0)                               // 00000000481C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s2                              // 000000004820: 8C7E027E

0000000000004824 <L4>:
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000004824: BF870099
	s_or_b32 exec_lo, exec_lo, s34                             // 000000004828: 8C7E227E
	v_cvt_i32_f32_e32 v5, v5                                   // 00000000482C: 7E0A1105
	s_delay_alu instid0(VALU_DEP_1)                            // 000000004830: BF870001
	v_lshlrev_b32_e32 v6, 16, v5                               // 000000004834: 300C0A90

0000000000004838 <L5>:
	s_or_b32 exec_lo, exec_lo, s33                             // 000000004838: 8C7E217E

000000000000483c <L6>:
	s_wait_alu depctr_sa_sdst(0)                               // 00000000483C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s31                             // 000000004840: 8C7E1F7E
	v_mul_lo_u32 v5, v2, 12                                    // 000000004844: D72C0005 02011902
	v_add_nc_u32_e32 v2, s16, v2                               // 00000000484C: 4A040410
	v_cvt_f32_ubyte1_e32 v7, v9                                // 000000004850: 7E0E2509
	v_cvt_f32_ubyte2_e32 v6, v6                                // 000000004854: 7E0C2706
	ds_store_2addr_b32 v5, v4, v7 offset1:1                    // 000000004858: D8380100 00070405
	ds_store_b32 v5, v6 offset:8                               // 000000004860: D8340008 00000605
	v_cmp_le_i32_e32 vcc_lo, s10, v2                           // 000000004868: 7C86040A
	s_or_b32 s29, vcc_lo, s29                                  // 00000000486C: 8C1D1D6A
	s_wait_alu depctr_sa_sdst(0)                               // 000000004870: BF88FF9E
	s_and_not1_b32 exec_lo, exec_lo, s29                       // 000000004874: 917E1D7E
	s_cbranch_execz L18                                        // 000000004878: BFA50160

000000000000487c <L7>:
	v_sub_nc_u32_e32 v4, 0, v2                                 // 00000000487C: 4C080480
	s_mov_b32 s31, exec_lo                                     // 000000004880: BE9F007E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000004884: BF870091
	v_max_i32_e32 v4, v4, v2                                   // 000000004888: 24080504
	v_mul_hi_u32 v5, v4, s30                                   // 00000000488C: D72D0005 02003D04
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000004894: BF870091
	v_mul_lo_u32 v6, v5, s18                                   // 000000004898: D72C0006 02002505
	v_sub_nc_u32_e32 v4, v4, v6                                // 0000000048A0: 4C080D04
	v_add_nc_u32_e32 v6, 1, v5                                 // 0000000048A4: 4A0C0A81
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 0000000048A8: BF870132
	v_subrev_nc_u32_e32 v7, s18, v4                            // 0000000048AC: 4E0E0812
	v_cmp_le_u32_e32 vcc_lo, s18, v4                           // 0000000048B0: 7C960812
	s_wait_alu depctr_va_vcc(0)                                // 0000000048B4: BF88FF9D
	v_dual_cndmask_b32 v5, v5, v6 :: v_dual_cndmask_b32 v4, v4, v7// 0000000048B8: CA520D05 05040F04
	v_ashrrev_i32_e32 v6, 31, v2                               // 0000000048C0: 340C049F
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)// 0000000048C4: BF870192
	v_add_nc_u32_e32 v7, 1, v5                                 // 0000000048C8: 4A0E0A81
	v_cmp_le_u32_e32 vcc_lo, s18, v4                           // 0000000048CC: 7C960812
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)// 0000000048D0: BF8701A3
	v_xor_b32_e32 v6, s22, v6                                  // 0000000048D4: 3A0C0C16
	s_wait_alu depctr_va_vcc(0)                                // 0000000048D8: BF88FF9D
	v_cndmask_b32_e32 v4, v5, v7, vcc_lo                       // 0000000048DC: 02080F05
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000048E0: BF870091
	v_xor_b32_e32 v4, v4, v6                                   // 0000000048E4: 3A080D04
	v_sub_nc_u32_e32 v4, v4, v6                                // 0000000048E8: 4C080D04
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000048EC: BF8700A1
	v_mul_lo_u32 v5, v4, s11                                   // 0000000048F0: D72C0005 02001704
	v_add_nc_u32_e32 v4, s24, v4                               // 0000000048F8: 4A080818
	v_min_i32_e32 v6, s21, v4                                  // 0000000048FC: 220C0815
	v_cmp_lt_i32_e32 vcc_lo, -1, v4                            // 000000004900: 7C8208C1
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000004904: BF8700A4
	v_sub_nc_u32_e32 v5, v2, v5                                // 000000004908: 4C0A0B02
	s_wait_alu depctr_va_vcc(0)                                // 00000000490C: BF88FF9D
	v_dual_cndmask_b32 v4, 0, v6 :: v_dual_add_nc_u32 v5, s23, v5// 000000004910: CA600C80 04040A17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000004918: BF870111
	v_cvt_f32_i32_e32 v4, v4                                   // 00000000491C: 7E080B04
	v_min_i32_e32 v6, s20, v5                                  // 000000004920: 220C0A14
	v_cmp_lt_i32_e32 vcc_lo, -1, v5                            // 000000004924: 7C820AC1
	s_wait_alu depctr_va_vcc(0)                                // 000000004928: BF88FF9D
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000492C: BF870092
	v_dual_add_f32 v4, 0.5, v4 :: v_dual_cndmask_b32 v5, 0, v6 // 000000004930: C91208F0 04040C80
	v_fma_f32 v4, s27, v4, -0.5                                // 000000004938: D6130004 03C6081B
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000004940: BF870112
	v_cvt_f32_i32_e32 v5, v5                                   // 000000004944: 7E0A0B05
	v_cmp_le_f32_e32 vcc_lo, 0, v4                             // 000000004948: 7C260880
	s_wait_alu depctr_va_vcc(0)                                // 00000000494C: BF88FF9D
	v_cndmask_b32_e32 v8, 0, v4, vcc_lo                        // 000000004950: 02100880
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000004954: BF870113
	v_add_f32_e32 v4, 0.5, v5                                  // 000000004958: 06080AF0
	v_floor_f32_e32 v5, v8                                     // 00000000495C: 7E0A4908
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000004960: BF870112
	v_fma_f32 v4, s28, v4, -0.5                                // 000000004964: D6130004 03C6081C
	v_cvt_i32_f32_e32 v21, v5                                  // 00000000496C: 7E2A1105
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000004970: BF870122
	v_cmp_le_f32_e32 vcc_lo, 0, v4                             // 000000004974: 7C260880
	s_wait_alu depctr_va_vcc(0)                                // 000000004978: BF88FF9D
	v_dual_cndmask_b32 v22, 0, v4 :: v_dual_add_nc_u32 v23, 1, v21// 00000000497C: CA600880 16162A81
	v_mul_lo_u32 v4, s19, v21                                  // 000000004984: D72C0004 02022A13
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)// 00000000498C: BF870192
	v_min_i32_e32 v6, s26, v23                                 // 000000004990: 220C2E1A
	v_floor_f32_e32 v5, v22                                    // 000000004994: 7E0A4916
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000004998: BF870113
	v_add_co_u32 v4, s2, s8, v4                                // 00000000499C: D7000204 02020808
	v_cvt_i32_f32_e32 v24, v5                                  // 0000000049A4: 7E301105
	s_wait_alu depctr_va_sdst(0)                               // 0000000049A8: BF88F19F
	v_add_co_ci_u32_e64 v5, null, s9, 0, s2                    // 0000000049AC: D5207C05 00090009
	v_mul_lo_u32 v9, v6, s19                                   // 0000000049B4: D72C0009 02002706
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 0000000049BC: BF870193
	v_add_nc_u32_e32 v25, 1, v24                               // 0000000049C0: 4A323081
	v_mad_co_i64_i32 v[6:7], null, v24, 3, v[4:5]              // 0000000049C4: D6FF7C06 04110718
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)// 0000000049CC: BF870212
	v_min_i32_e32 v10, s25, v25                                // 0000000049D0: 22143219
	v_add_co_u32 v12, s2, s8, v9                               // 0000000049D4: D700020C 02021208
	s_wait_alu depctr_va_sdst(0)                               // 0000000049DC: BF88F19F
	v_add_co_ci_u32_e64 v13, null, s9, 0, s2                   // 0000000049E0: D5207C0D 00090009
	global_load_u16 v9, v[6:7], off                            // 0000000049E8: EE04807C 00000009 00000006
	v_mad_co_i64_i32 v[4:5], null, v10, 3, v[4:5]              // 0000000049F4: D6FF7C04 0411070A
	v_mad_co_i64_i32 v[17:18], null, v24, 3, v[12:13]          // 0000000049FC: D6FF7C11 04310718
	v_mad_co_i64_i32 v[19:20], null, v10, 3, v[12:13]          // 000000004A04: D6FF7C13 0431070A
	s_clause 0x6                                               // 000000004A0C: BF850006
	global_load_u16 v12, v[17:18], off                         // 000000004A10: EE04807C 0000000C 00000011
	global_load_u16 v10, v[19:20], off                         // 000000004A1C: EE04807C 0000000A 00000013
	global_load_u8 v15, v[6:7], off offset:2                   // 000000004A28: EE04007C 0000000F 00000206
	global_load_u8 v16, v[4:5], off offset:2                   // 000000004A34: EE04007C 00000010 00000204
	global_load_u8 v14, v[17:18], off offset:2                 // 000000004A40: EE04007C 0000000E 00000211
	global_load_u8 v13, v[19:20], off offset:2                 // 000000004A4C: EE04007C 0000000D 00000213
	global_load_u16 v11, v[4:5], off                           // 000000004A58: EE04807C 0000000B 00000004
	v_cvt_f32_i32_e32 v4, v23                                  // 000000004A64: 7E080B17
	v_cvt_f32_i32_e32 v5, v25                                  // 000000004A68: 7E0A0B19
	v_cvt_f32_i32_e32 v6, v24                                  // 000000004A6C: 7E0C0B18
	v_cvt_f32_i32_e32 v17, v21                                 // 000000004A70: 7E220B15
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 000000004A74: BF8700C3
	v_sub_f32_e32 v7, v5, v22                                  // 000000004A78: 080E2D05
	s_wait_loadcnt 0x7                                         // 000000004A7C: BFC00007
	v_cvt_f32_ubyte0_e32 v19, v9                               // 000000004A80: 7E262309
	v_sub_f32_e32 v4, v4, v8                                   // 000000004A84: 08081104
	v_mul_f32_e32 v5, v4, v7                                   // 000000004A88: 100A0F04
	v_sub_f32_e32 v8, v8, v17                                  // 000000004A8C: 08102308
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000004A90: BF870112
	v_dual_sub_f32 v18, v22, v6 :: v_dual_mul_f32 v17, v5, v19 // 000000004A94: C9460D16 12102705
	v_mul_f32_e32 v7, v7, v8                                   // 000000004A9C: 100E1107
	s_wait_loadcnt 0x6                                         // 000000004AA0: BFC00006
	v_cvt_f32_ubyte0_e32 v19, v12                              // 000000004AA4: 7E26230C
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 000000004AA8: BF8700B3
	v_mul_f32_e32 v6, v18, v4                                  // 000000004AAC: 100C0912
	s_wait_loadcnt 0x0                                         // 000000004AB0: BFC00000
	v_cvt_f32_ubyte0_e32 v4, v11                               // 000000004AB4: 7E08230B
	v_dual_fmac_f32 v17, v6, v4 :: v_dual_mul_f32 v8, v8, v18  // 000000004AB8: C8060906 11082508
	v_cvt_f32_ubyte0_e32 v4, v10                               // 000000004AC0: 7E08230A
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000004AC4: BF870092
	v_fmac_f32_e32 v17, v7, v19                                // 000000004AC8: 56222707
	v_dual_fmac_f32 v17, v8, v4 :: v_dual_mov_b32 v4, 0x437f0000// 000000004ACC: C8100908 110400FF 437F0000
	s_delay_alu instid0(VALU_DEP_1)                            // 000000004AD8: BF870001
	v_cmpx_nlt_f32_e32 0x437f0000, v17                         // 000000004ADC: 7D3C22FF 437F0000
	s_cbranch_execz L12                                        // 000000004AE4: BFA50035
	v_mov_b32_e32 v4, 0                                        // 000000004AE8: 7E080280
	s_mov_b32 s33, exec_lo                                     // 000000004AEC: BEA1007E
	v_cmpx_ngt_f32_e32 0, v17                                  // 000000004AF0: 7D362280
	s_cbranch_execz L11                                        // 000000004AF4: BFA50030
	s_mov_b32 s34, exec_lo                                     // 000000004AF8: BEA2007E
	v_cmpx_o_f32_e32 v17, v17                                  // 000000004AFC: 7D2E2311
	s_cbranch_execz L10                                        // 000000004B00: BFA50029
	v_add_f32_e32 v4, v17, v17                                 // 000000004B04: 06082311
	v_cmp_nlg_f32_e32 vcc_lo, 0, v17                           // 000000004B08: 7C342280
	v_cmp_neq_f32_e64 s3, 0, v17                               // 000000004B0C: D41D0003 02022280
	s_delay_alu instid0(VALU_DEP_3)                            // 000000004B14: BF870003
	v_cmp_neq_f32_e64 s2, v4, v17                              // 000000004B18: D41D0002 02022304
	s_or_b32 s2, vcc_lo, s2                                    // 000000004B20: 8C02026A
	s_wait_alu depctr_sa_sdst(0)                               // 000000004B24: BF88FF9E
	s_and_b32 s3, s3, s2                                       // 000000004B28: 8B030203
	s_wait_alu depctr_sa_sdst(0)                               // 000000004B2C: BF88FF9E
	s_and_saveexec_b32 s2, s3                                  // 000000004B30: BE822003
	s_cbranch_execz L9                                         // 000000004B34: BFA5001A
	v_cvt_i32_f32_e32 v18, v17                                 // 000000004B38: 7E241111
	s_mov_b32 s3, exec_lo                                      // 000000004B3C: BE83007E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000004B40: BF870091
	v_cvt_f32_i32_e32 v4, v18                                  // 000000004B44: 7E080B12
	v_sub_f32_e32 v17, v17, v4                                 // 000000004B48: 08220911
	s_delay_alu instid0(VALU_DEP_1)                            // 000000004B4C: BF870001
	v_cmpx_ngt_f32_e32 0.5, v17                                // 000000004B50: 7D3622F0
	s_cbranch_execz L8                                         // 000000004B54: BFA5000E
	v_cmp_nlt_f32_e32 vcc_lo, 0.5, v17                         // 000000004B58: 7C3C22F0
	v_add_f32_e32 v17, 1.0, v4                                 // 000000004B5C: 062208F2
	s_and_saveexec_b32 s35, vcc_lo                             // 000000004B60: BEA3206A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000004B64: BF870009
	s_xor_b32 s35, exec_lo, s35                                // 000000004B68: 8D23237E
	v_dual_add_f32 v18, 1.0, v4 :: v_dual_and_b32 v17, 1, v18  // 000000004B6C: C92408F2 12102481
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000004B74: BF870121
	v_cmp_eq_u32_e32 vcc_lo, 0, v17                            // 000000004B78: 7C942280
	s_wait_alu depctr_va_vcc(0)                                // 000000004B7C: BF88FF9D
	v_cndmask_b32_e32 v4, v18, v4, vcc_lo                      // 000000004B80: 02080912
	s_and_not1_saveexec_b32 s35, s35                           // 000000004B84: BEA33023
	v_mov_b32_e32 v4, v17                                      // 000000004B88: 7E080311
	s_or_b32 exec_lo, exec_lo, s35                             // 000000004B8C: 8C7E237E

0000000000004b90 <L8>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000004B90: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s3                              // 000000004B94: 8C7E037E
	s_delay_alu instid0(VALU_DEP_1)                            // 000000004B98: BF870001
	v_mov_b32_e32 v17, v4                                      // 000000004B9C: 7E220304

0000000000004ba0 <L9>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000004BA0: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s2                              // 000000004BA4: 8C7E027E

0000000000004ba8 <L10>:
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000004BA8: BF870099
	s_or_b32 exec_lo, exec_lo, s34                             // 000000004BAC: 8C7E227E
	v_trunc_f32_e64 v4, |v17|                                  // 000000004BB0: D5A10104 02010111

0000000000004bb8 <L11>:
	s_or_b32 exec_lo, exec_lo, s33                             // 000000004BB8: 8C7E217E

0000000000004bbc <L12>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000004BBC: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s31                             // 000000004BC0: 8C7E1F7E
	v_lshl_or_b32 v15, v15, 16, v9                             // 000000004BC4: D656000F 0425210F
	v_lshl_or_b32 v11, v16, 16, v11                            // 000000004BCC: D656000B 042D2110
	v_lshl_or_b32 v12, v14, 16, v12                            // 000000004BD4: D656000C 0431210E
	v_lshl_or_b32 v10, v13, 16, v10                            // 000000004BDC: D656000A 0429210D
	s_mov_b32 s31, exec_lo                                     // 000000004BE4: BE9F007E
	v_cvt_f32_ubyte1_e32 v9, v15                               // 000000004BE8: 7E12250F
	v_cvt_f32_ubyte1_e32 v16, v11                              // 000000004BEC: 7E20250B
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000004BF0: BF870193
	v_cvt_f32_ubyte1_e32 v13, v10                              // 000000004BF4: 7E1A250A
	v_mul_f32_e32 v14, v5, v9                                  // 000000004BF8: 101C1305
	v_cvt_f32_ubyte1_e32 v9, v12                               // 000000004BFC: 7E12250C
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000004C00: BF870092
	v_fmac_f32_e32 v14, v6, v16                                // 000000004C04: 561C2106
	v_dual_fmac_f32 v14, v7, v9 :: v_dual_mov_b32 v9, 0xff00   // 000000004C08: C8101307 0E0800FF 0000FF00
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000004C14: BF870091
	v_fmac_f32_e32 v14, v8, v13                                // 000000004C18: 561C1B08
	v_cmpx_nlt_f32_e32 0x437f0000, v14                         // 000000004C1C: 7D3C1CFF 437F0000
	s_cbranch_execz L17                                        // 000000004C24: BFA50037
	v_mov_b32_e32 v9, 0                                        // 000000004C28: 7E120280
	s_mov_b32 s33, exec_lo                                     // 000000004C2C: BEA1007E
	v_cmpx_ngt_f32_e32 0, v14                                  // 000000004C30: 7D361C80
	s_cbranch_execz L16                                        // 000000004C34: BFA50032
	s_mov_b32 s34, exec_lo                                     // 000000004C38: BEA2007E
	v_cmpx_o_f32_e32 v14, v14                                  // 000000004C3C: 7D2E1D0E
	s_cbranch_execz L15                                        // 000000004C40: BFA50029
	v_add_f32_e32 v9, v14, v14                                 // 000000004C44: 06121D0E
	v_cmp_nlg_f32_e32 vcc_lo, 0, v14                           // 000000004C48: 7C341C80
	v_cmp_neq_f32_e64 s3, 0, v14                               // 000000004C4C: D41D0003 02021C80
	s_delay_alu instid0(VALU_DEP_3)                            // 000000004C54: BF870003
	v_cmp_neq_f32_e64 s2, v9, v14                              // 000000004C58: D41D0002 02021D09
	s_or_b32 s2, vcc_lo, s2                                    // 000000004C60: 8C02026A
	s_wait_alu depctr_sa_sdst(0)                               // 000000004C64: BF88FF9E
	s_and_b32 s3, s3, s2                                       // 000000004C68: 8B030203
	s_wait_alu depctr_sa_sdst(0)                               // 000000004C6C: BF88FF9E
	s_and_saveexec_b32 s2, s3                                  // 000000004C70: BE822003
	s_cbranch_execz L14                                        // 000000004C74: BFA5001A
	v_cvt_i32_f32_e32 v13, v14                                 // 000000004C78: 7E1A110E
	s_mov_b32 s3, exec_lo                                      // 000000004C7C: BE83007E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000004C80: BF870091
	v_cvt_f32_i32_e32 v9, v13                                  // 000000004C84: 7E120B0D
	v_sub_f32_e32 v14, v14, v9                                 // 000000004C88: 081C130E
	s_delay_alu instid0(VALU_DEP_1)                            // 000000004C8C: BF870001
	v_cmpx_ngt_f32_e32 0.5, v14                                // 000000004C90: 7D361CF0
	s_cbranch_execz L13                                        // 000000004C94: BFA5000E
	v_cmp_nlt_f32_e32 vcc_lo, 0.5, v14                         // 000000004C98: 7C3C1CF0
	v_add_f32_e32 v14, 1.0, v9                                 // 000000004C9C: 061C12F2
	s_and_saveexec_b32 s35, vcc_lo                             // 000000004CA0: BEA3206A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000004CA4: BF870009
	s_xor_b32 s35, exec_lo, s35                                // 000000004CA8: 8D23237E
	v_and_b32_e32 v13, 1, v13                                  // 000000004CAC: 361A1A81
	v_add_f32_e32 v14, 1.0, v9                                 // 000000004CB0: 061C12F2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000004CB4: BF870122
	v_cmp_eq_u32_e32 vcc_lo, 0, v13                            // 000000004CB8: 7C941A80
	s_wait_alu depctr_va_vcc(0)                                // 000000004CBC: BF88FF9D
	v_cndmask_b32_e32 v9, v14, v9, vcc_lo                      // 000000004CC0: 0212130E
	s_and_not1_saveexec_b32 s35, s35                           // 000000004CC4: BEA33023
	v_mov_b32_e32 v9, v14                                      // 000000004CC8: 7E12030E
	s_or_b32 exec_lo, exec_lo, s35                             // 000000004CCC: 8C7E237E

0000000000004cd0 <L13>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000004CD0: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s3                              // 000000004CD4: 8C7E037E
	s_delay_alu instid0(VALU_DEP_1)                            // 000000004CD8: BF870001
	v_mov_b32_e32 v14, v9                                      // 000000004CDC: 7E1C0309

0000000000004ce0 <L14>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000004CE0: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s2                              // 000000004CE4: 8C7E027E

0000000000004ce8 <L15>:
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000004CE8: BF870099
	s_or_b32 exec_lo, exec_lo, s34                             // 000000004CEC: 8C7E227E
	v_cvt_i32_f32_e32 v9, v14                                  // 000000004CF0: 7E12110E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000004CF4: BF870091
	v_cvt_u32_u16_e32 v9, v9.l                                 // 000000004CF8: 7E12D709
	v_lshlrev_b32_e32 v9, 8, v9                                // 000000004CFC: 30121288

0000000000004d00 <L16>:
	s_or_b32 exec_lo, exec_lo, s33                             // 000000004D00: 8C7E217E

0000000000004d04 <L17>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000004D04: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s31                             // 000000004D08: 8C7E1F7E
	v_cvt_f32_ubyte2_e32 v13, v15                              // 000000004D0C: 7E1A270F
	v_cvt_f32_ubyte2_e32 v11, v11                              // 000000004D10: 7E16270B
	v_cvt_f32_ubyte2_e32 v12, v12                              // 000000004D14: 7E18270C
	s_mov_b32 s31, exec_lo                                     // 000000004D18: BE9F007E
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000004D1C: BF870093
	v_mul_f32_e32 v5, v5, v13                                  // 000000004D20: 100A1B05
	v_fmac_f32_e32 v5, v6, v11                                 // 000000004D24: 560A1706
	v_cvt_f32_ubyte2_e32 v6, v10                               // 000000004D28: 7E0C270A
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000004D2C: BF870092
	v_fmac_f32_e32 v5, v7, v12                                 // 000000004D30: 560A1907
	v_dual_fmac_f32 v5, v8, v6 :: v_dual_mov_b32 v6, 0xff0000  // 000000004D34: C8100D08 050600FF 00FF0000
	s_delay_alu instid0(VALU_DEP_1)                            // 000000004D40: BF870001
	v_cmpx_nlt_f32_e32 0x437f0000, v5                          // 000000004D44: 7D3C0AFF 437F0000
	s_cbranch_execz L6                                         // 000000004D4C: BFA5FEBB
	v_mov_b32_e32 v6, 0                                        // 000000004D50: 7E0C0280
	s_mov_b32 s33, exec_lo                                     // 000000004D54: BEA1007E
	v_cmpx_ngt_f32_e32 0, v5                                   // 000000004D58: 7D360A80
	s_cbranch_execz L5                                         // 000000004D5C: BFA5FEB6
	s_mov_b32 s34, exec_lo                                     // 000000004D60: BEA2007E
	v_cmpx_o_f32_e32 v5, v5                                    // 000000004D64: 7D2E0B05
	s_cbranch_execz L4                                         // 000000004D68: BFA5FEAE
	v_add_f32_e32 v6, v5, v5                                   // 000000004D6C: 060C0B05
	v_cmp_nlg_f32_e32 vcc_lo, 0, v5                            // 000000004D70: 7C340A80
	v_cmp_neq_f32_e64 s3, 0, v5                                // 000000004D74: D41D0003 02020A80
	s_delay_alu instid0(VALU_DEP_3)                            // 000000004D7C: BF870003
	v_cmp_neq_f32_e64 s2, v6, v5                               // 000000004D80: D41D0002 02020B06
	s_or_b32 s2, vcc_lo, s2                                    // 000000004D88: 8C02026A
	s_wait_alu depctr_sa_sdst(0)                               // 000000004D8C: BF88FF9E
	s_and_b32 s3, s3, s2                                       // 000000004D90: 8B030203
	s_wait_alu depctr_sa_sdst(0)                               // 000000004D94: BF88FF9E
	s_and_saveexec_b32 s2, s3                                  // 000000004D98: BE822003
	s_cbranch_execz L3                                         // 000000004D9C: BFA5FE9F
	v_cvt_i32_f32_e32 v7, v5                                   // 000000004DA0: 7E0E1105
	s_mov_b32 s3, exec_lo                                      // 000000004DA4: BE83007E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000004DA8: BF870091
	v_cvt_f32_i32_e32 v6, v7                                   // 000000004DAC: 7E0C0B07
	v_sub_f32_e32 v5, v5, v6                                   // 000000004DB0: 080A0D05
	s_delay_alu instid0(VALU_DEP_1)                            // 000000004DB4: BF870001
	v_cmpx_ngt_f32_e32 0.5, v5                                 // 000000004DB8: 7D360AF0
	s_cbranch_execz L2                                         // 000000004DBC: BFA5FE94
	v_cmp_nlt_f32_e32 vcc_lo, 0.5, v5                          // 000000004DC0: 7C3C0AF0
	v_add_f32_e32 v5, 1.0, v6                                  // 000000004DC4: 060A0CF2
	s_and_saveexec_b32 s35, vcc_lo                             // 000000004DC8: BEA3206A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000004DCC: BF870009
	s_xor_b32 s35, exec_lo, s35                                // 000000004DD0: 8D23237E
	v_and_b32_e32 v5, 1, v7                                    // 000000004DD4: 360A0E81
	v_add_f32_e32 v7, 1.0, v6                                  // 000000004DD8: 060E0CF2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000004DDC: BF870122
	v_cmp_eq_u32_e32 vcc_lo, 0, v5                             // 000000004DE0: 7C940A80
	s_wait_alu depctr_va_vcc(0)                                // 000000004DE4: BF88FF9D
	v_cndmask_b32_e32 v6, v7, v6, vcc_lo                       // 000000004DE8: 020C0D07
	s_and_not1_saveexec_b32 s35, s35                           // 000000004DEC: BEA33023
	s_cbranch_execz L1                                         // 000000004DF0: BFA5FE86
	v_mov_b32_e32 v6, v5                                       // 000000004DF4: 7E0C0305
	s_branch L1                                                // 000000004DF8: BFA0FE84

0000000000004dfc <L18>:
	s_or_b32 exec_lo, exec_lo, s17                             // 000000004DFC: 8C7E117E
	s_clause 0x1                                               // 000000004E00: BF850001
	s_load_b96 s[8:10], s[0:1], 0x48                           // 000000004E04: F400A200 F8000048
	s_load_b32 s3, s[0:1], 0x70                                // 000000004E0C: F40000C0 F8000070
	s_mul_i32 s0, s7, s6                                       // 000000004E14: 96000607
	s_mov_b32 s1, exec_lo                                      // 000000004E18: BE81007E
	v_cmpx_gt_i32_e32 s0, v1                                   // 000000004E1C: 7D880200
	s_cbranch_execz L20                                        // 000000004E20: BFA50015
	s_mov_b32 s2, 0                                            // 000000004E24: BE820080

0000000000004e28 <L19>:
	v_mul_lo_u32 v2, v1, 12                                    // 000000004E28: D72C0002 02011901
	s_wait_kmcnt 0x0                                           // 000000004E30: BFC70000
	v_dual_mov_b32 v4, s8 :: v_dual_add_nc_u32 v1, s16, v1     // 000000004E34: CA200008 04000210
	v_dual_mov_b32 v5, s9 :: v_dual_mov_b32 v6, s10            // 000000004E3C: CA100009 0506000A
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000004E44: BF870212
	v_cmp_le_i32_e32 vcc_lo, s0, v1                            // 000000004E48: 7C860200
	v_add_nc_u32_e32 v7, 0xe70, v2                             // 000000004E4C: 4A0E04FF 00000E70
	ds_store_2addr_b32 v7, v4, v5 offset1:1                    // 000000004E54: D8380100 00050407
	ds_store_b32 v2, v6 offset:3704                            // 000000004E5C: D8340E78 00000602
	s_wait_alu depctr_sa_sdst(0)                               // 000000004E64: BF88FF9E
	s_or_b32 s2, vcc_lo, s2                                    // 000000004E68: 8C02026A
	s_wait_alu depctr_sa_sdst(0)                               // 000000004E6C: BF88FF9E
	s_and_not1_b32 exec_lo, exec_lo, s2                        // 000000004E70: 917E027E
	s_cbranch_execnz L19                                       // 000000004E74: BFA6FFEC

0000000000004e78 <L20>:
	s_or_b32 exec_lo, exec_lo, s1                              // 000000004E78: 8C7E017E
	s_wait_dscnt 0x0                                           // 000000004E7C: BFC60000
	s_barrier_signal -1                                        // 000000004E80: BE804EC1
	v_add_nc_u32_e32 v4, s14, v0                               // 000000004E84: 4A08000E
	v_add_nc_u32_e32 v5, s15, v3                               // 000000004E88: 4A0A060F
	v_cmp_gt_u32_e32 vcc_lo, 16, v0                            // 000000004E8C: 7C980090
	v_cmp_gt_u32_e64 s0, 8, v3                                 // 000000004E90: D44C0000 02020688
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000004E98: BF870214
	v_cmp_gt_i32_e64 s1, s4, v4                                // 000000004E9C: D4440001 02020804
	v_cmp_gt_i32_e64 s2, s5, v5                                // 000000004EA4: D4440002 02020A05
	s_and_b32 s0, vcc_lo, s0                                   // 000000004EAC: 8B00006A
	s_wait_alu depctr_sa_sdst(0)                               // 000000004EB0: BF88FF9E
	s_and_b32 s0, s0, s1                                       // 000000004EB4: 8B000100
	s_wait_alu depctr_sa_sdst(0)                               // 000000004EB8: BF88FF9E
	s_and_b32 s0, s0, s2                                       // 000000004EBC: 8B000200
	s_barrier_wait 0xffff                                      // 000000004EC0: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000004EC4: EE0AC07C 00040000 00000000
	s_wait_alu depctr_sa_sdst(0)                               // 000000004ED0: BF88FF9E
	s_and_saveexec_b32 s1, s0                                  // 000000004ED4: BE812000
	s_cbranch_execz L41                                        // 000000004ED8: BFA5010A
	s_cmp_lt_i32 s7, 1                                         // 000000004EDC: BF048107
	s_cbranch_scc1 L24                                         // 000000004EE0: BFA20036
	v_mul_u32_u24_e32 v0, 12, v0                               // 000000004EE4: 1600008C
	v_dual_mov_b32 v8, 0 :: v_dual_mov_b32 v7, 0               // 000000004EE8: CA100080 08060080
	v_mov_b32_e32 v6, 0                                        // 000000004EF0: 7E0C0280
	s_cmp_gt_i32 s6, 0                                         // 000000004EF4: BF028006
	s_mov_b32 s0, 0                                            // 000000004EF8: BE800080
	s_cselect_b32 s1, -1, 0                                    // 000000004EFC: 980180C1
	s_branch L22                                               // 000000004F00: BFA00004

0000000000004f04 <L21>:
	s_add_co_i32 s0, s0, 1                                     // 000000004F04: 81008100
	s_wait_alu depctr_sa_sdst(0)                               // 000000004F08: BF88FF9E
	s_cmp_eq_u32 s0, s7                                        // 000000004F0C: BF060700
	s_cbranch_scc1 L25                                         // 000000004F10: BFA2002D

0000000000004f14 <L22>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000004F14: BF88FF9E
	s_and_b32 s2, s1, exec_lo                                  // 000000004F18: 8B027E01
	s_cselect_b32 s2, 1, 0                                     // 000000004F1C: 98028081
	s_wait_alu depctr_sa_sdst(0)                               // 000000004F20: BF88FF9E
	s_cmp_lg_u32 s2, 1                                         // 000000004F24: BF078102
	s_cbranch_scc1 L21                                         // 000000004F28: BFA2FFF6
	v_add_nc_u32_e32 v1, s0, v3                                // 000000004F2C: 4A020600
	s_mul_i32 s2, s0, s6                                       // 000000004F30: 96020600
	s_mov_b32 s4, 0                                            // 000000004F34: BE840080
	s_wait_alu depctr_sa_sdst(0)                               // 000000004F38: BF88FF9E
	s_mul_i32 s2, s2, 12                                       // 000000004F3C: 96028C02
	s_wait_alu depctr_sa_sdst(0)                               // 000000004F40: BF88FF9E
	s_addk_co_i32 s2, 0xe70                                    // 000000004F44: B7820E70
	v_mul_lo_u32 v1, v1, s11                                   // 000000004F48: D72C0001 02001701
	s_delay_alu instid0(VALU_DEP_1)                            // 000000004F50: BF870001
	v_mad_co_u64_u32 v[1:2], null, v1, 12, v[0:1]              // 000000004F54: D6FE7C01 04011901

0000000000004f5c <L23>:
	s_mul_i32 s5, s4, 12                                       // 000000004F5C: 96058C04
	s_add_co_i32 s4, s4, 1                                     // 000000004F60: 81048104
	s_wait_alu depctr_sa_sdst(0)                               // 000000004F64: BF88FF9E
	s_delay_alu instid0(VALU_DEP_1)                            // 000000004F68: BF870001
	v_add_nc_u32_e32 v2, s5, v1                                // 000000004F6C: 4A040205
	s_add_co_i32 s5, s2, s5                                    // 000000004F70: 81050502
	s_cmp_eq_u32 s4, s6                                        // 000000004F74: BF060604
	s_wait_alu depctr_sa_sdst(0)                               // 000000004F78: BF88FF9E
	v_mov_b32_e32 v13, s5                                      // 000000004F7C: 7E1A0205
	ds_load_2addr_b32 v[9:10], v2 offset1:1                    // 000000004F80: D8DC0100 09000002
	ds_load_b32 v2, v2 offset:8                                // 000000004F88: D8D80008 02000002
	ds_load_2addr_b32 v[11:12], v13 offset1:1                  // 000000004F90: D8DC0100 0B00000D
	ds_load_b32 v13, v13 offset:8                              // 000000004F98: D8D80008 0D00000D
	s_wait_dscnt 0x1                                           // 000000004FA0: BFC60001
	v_dual_fmac_f32 v8, v9, v11 :: v_dual_fmac_f32 v7, v10, v12// 000000004FA4: C8001709 0806190A
	s_wait_dscnt 0x0                                           // 000000004FAC: BFC60000
	v_fmac_f32_e32 v6, v2, v13                                 // 000000004FB0: 560C1B02
	s_cbranch_scc0 L23                                         // 000000004FB4: BFA1FFE9
	s_branch L21                                               // 000000004FB8: BFA0FFD2

0000000000004fbc <L24>:
	v_dual_mov_b32 v8, 0 :: v_dual_mov_b32 v7, 0               // 000000004FBC: CA100080 08060080
	v_mov_b32_e32 v6, 0                                        // 000000004FC4: 7E0C0280

0000000000004fc8 <L25>:
	v_mov_b16_e32 v0.l, 0xff                                   // 000000004FC8: 7E0038FF 000000FF
	s_mov_b32 s2, exec_lo                                      // 000000004FD0: BE82007E
	s_delay_alu instid0(VALU_DEP_3)                            // 000000004FD4: BF870003
	v_cmpx_nlt_f32_e32 0x437f0000, v8                          // 000000004FD8: 7D3C10FF 437F0000
	s_cbranch_execz L30                                        // 000000004FE0: BFA50034
	v_mov_b16_e32 v0.l, 0                                      // 000000004FE4: 7E003880
	s_mov_b32 s4, exec_lo                                      // 000000004FE8: BE84007E
	v_cmpx_ngt_f32_e32 0, v8                                   // 000000004FEC: 7D361080
	s_cbranch_execz L29                                        // 000000004FF0: BFA5002E
	s_mov_b32 s5, exec_lo                                      // 000000004FF4: BE85007E
	v_cmpx_o_f32_e32 v8, v8                                    // 000000004FF8: 7D2E1108
	s_cbranch_execz L28                                        // 000000004FFC: BFA50028
	v_add_f32_e32 v0, v8, v8                                   // 000000005000: 06001108
	v_cmp_nlg_f32_e32 vcc_lo, 0, v8                            // 000000005004: 7C341080
	v_cmp_neq_f32_e64 s1, 0, v8                                // 000000005008: D41D0001 02021080
	s_delay_alu instid0(VALU_DEP_3)                            // 000000005010: BF870003
	v_cmp_neq_f32_e64 s0, v0, v8                               // 000000005014: D41D0000 02021100
	s_or_b32 s0, vcc_lo, s0                                    // 00000000501C: 8C00006A
	s_wait_alu depctr_sa_sdst(0)                               // 000000005020: BF88FF9E
	s_and_b32 s1, s1, s0                                       // 000000005024: 8B010001
	s_wait_alu depctr_sa_sdst(0)                               // 000000005028: BF88FF9E
	s_and_saveexec_b32 s0, s1                                  // 00000000502C: BE802001
	s_cbranch_execz L27                                        // 000000005030: BFA50019
	v_cvt_i32_f32_e32 v1, v8                                   // 000000005034: 7E021108
	s_mov_b32 s1, exec_lo                                      // 000000005038: BE81007E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000503C: BF870091
	v_cvt_f32_i32_e32 v0, v1                                   // 000000005040: 7E000B01
	v_sub_f32_e32 v2, v8, v0                                   // 000000005044: 08040108
	s_delay_alu instid0(VALU_DEP_1)                            // 000000005048: BF870001
	v_cmpx_ngt_f32_e32 0.5, v2                                 // 00000000504C: 7D3604F0
	s_cbranch_execz L26                                        // 000000005050: BFA5000D
	s_mov_b32 s6, exec_lo                                      // 000000005054: BE86007E
	v_cmpx_nlt_f32_e32 0.5, v2                                 // 000000005058: 7D3C04F0
	s_xor_b32 s6, exec_lo, s6                                  // 00000000505C: 8D06067E
	v_dual_add_f32 v2, 1.0, v0 :: v_dual_and_b32 v1, 1, v1     // 000000005060: C92400F2 02000281
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000005068: BF870121
	v_cmp_eq_u32_e32 vcc_lo, 0, v1                             // 00000000506C: 7C940280
	s_wait_alu depctr_va_vcc(0)                                // 000000005070: BF88FF9D
	v_cndmask_b32_e32 v0, v2, v0, vcc_lo                       // 000000005074: 02000102
	s_and_not1_saveexec_b32 s6, s6                             // 000000005078: BE863006
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000507C: BF870001
	v_add_f32_e32 v0, 1.0, v0                                  // 000000005080: 060000F2
	s_or_b32 exec_lo, exec_lo, s6                              // 000000005084: 8C7E067E

0000000000005088 <L26>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000005088: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000508C: 8C7E017E
	s_delay_alu instid0(VALU_DEP_1)                            // 000000005090: BF870001
	v_mov_b32_e32 v8, v0                                       // 000000005094: 7E100300

0000000000005098 <L27>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000005098: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s0                              // 00000000509C: 8C7E007E

00000000000050a0 <L28>:
	s_wait_alu depctr_sa_sdst(0)                               // 0000000050A0: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s5                              // 0000000050A4: 8C7E057E
	v_cvt_i32_f32_e32 v0, v8                                   // 0000000050A8: 7E001108

00000000000050ac <L29>:
	s_wait_alu depctr_sa_sdst(0)                               // 0000000050AC: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s4                              // 0000000050B0: 8C7E047E

00000000000050b4 <L30>:
	s_wait_alu depctr_sa_sdst(0)                               // 0000000050B4: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s2                              // 0000000050B8: 8C7E027E
	v_mov_b32_e32 v1, 0xff00                                   // 0000000050BC: 7E0202FF 0000FF00
	s_mov_b32 s2, exec_lo                                      // 0000000050C4: BE82007E
	v_cmpx_nlt_f32_e32 0x437f0000, v7                          // 0000000050C8: 7D3C0EFF 437F0000
	s_cbranch_execz L35                                        // 0000000050D0: BFA50037
	v_mov_b32_e32 v1, 0                                        // 0000000050D4: 7E020280
	s_mov_b32 s4, exec_lo                                      // 0000000050D8: BE84007E
	v_cmpx_ngt_f32_e32 0, v7                                   // 0000000050DC: 7D360E80
	s_cbranch_execz L34                                        // 0000000050E0: BFA50031
	s_mov_b32 s5, exec_lo                                      // 0000000050E4: BE85007E
	v_cmpx_o_f32_e32 v7, v7                                    // 0000000050E8: 7D2E0F07
	s_cbranch_execz L33                                        // 0000000050EC: BFA50028
	v_add_f32_e32 v1, v7, v7                                   // 0000000050F0: 06020F07
	v_cmp_nlg_f32_e32 vcc_lo, 0, v7                            // 0000000050F4: 7C340E80
	v_cmp_neq_f32_e64 s1, 0, v7                                // 0000000050F8: D41D0001 02020E80
	s_delay_alu instid0(VALU_DEP_3)                            // 000000005100: BF870003
	v_cmp_neq_f32_e64 s0, v1, v7                               // 000000005104: D41D0000 02020F01
	s_or_b32 s0, vcc_lo, s0                                    // 00000000510C: 8C00006A
	s_wait_alu depctr_sa_sdst(0)                               // 000000005110: BF88FF9E
	s_and_b32 s1, s1, s0                                       // 000000005114: 8B010001
	s_wait_alu depctr_sa_sdst(0)                               // 000000005118: BF88FF9E
	s_and_saveexec_b32 s0, s1                                  // 00000000511C: BE802001
	s_cbranch_execz L32                                        // 000000005120: BFA50019
	v_cvt_i32_f32_e32 v2, v7                                   // 000000005124: 7E041107
	s_mov_b32 s1, exec_lo                                      // 000000005128: BE81007E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000512C: BF870091
	v_cvt_f32_i32_e32 v1, v2                                   // 000000005130: 7E020B02
	v_sub_f32_e32 v3, v7, v1                                   // 000000005134: 08060307
	s_delay_alu instid0(VALU_DEP_1)                            // 000000005138: BF870001
	v_cmpx_ngt_f32_e32 0.5, v3                                 // 00000000513C: 7D3606F0
	s_cbranch_execz L31                                        // 000000005140: BFA5000D
	s_mov_b32 s6, exec_lo                                      // 000000005144: BE86007E
	v_cmpx_nlt_f32_e32 0.5, v3                                 // 000000005148: 7D3C06F0
	s_xor_b32 s6, exec_lo, s6                                  // 00000000514C: 8D06067E
	v_dual_add_f32 v3, 1.0, v1 :: v_dual_and_b32 v2, 1, v2     // 000000005150: C92402F2 03020481
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000005158: BF870121
	v_cmp_eq_u32_e32 vcc_lo, 0, v2                             // 00000000515C: 7C940480
	s_wait_alu depctr_va_vcc(0)                                // 000000005160: BF88FF9D
	v_cndmask_b32_e32 v1, v3, v1, vcc_lo                       // 000000005164: 02020303
	s_and_not1_saveexec_b32 s6, s6                             // 000000005168: BE863006
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000516C: BF870001
	v_add_f32_e32 v1, 1.0, v1                                  // 000000005170: 060202F2
	s_or_b32 exec_lo, exec_lo, s6                              // 000000005174: 8C7E067E

0000000000005178 <L31>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000005178: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000517C: 8C7E017E
	s_delay_alu instid0(VALU_DEP_1)                            // 000000005180: BF870001
	v_mov_b32_e32 v7, v1                                       // 000000005184: 7E0E0301

0000000000005188 <L32>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000005188: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s0                              // 00000000518C: 8C7E007E

0000000000005190 <L33>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000005190: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s5                              // 000000005194: 8C7E057E
	v_cvt_i32_f32_e32 v1, v7                                   // 000000005198: 7E021107
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000519C: BF870091
	v_cvt_u32_u16_e32 v1, v1.l                                 // 0000000051A0: 7E02D701
	v_lshlrev_b32_e32 v1, 8, v1                                // 0000000051A4: 30020288

00000000000051a8 <L34>:
	s_wait_alu depctr_sa_sdst(0)                               // 0000000051A8: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s4                              // 0000000051AC: 8C7E047E

00000000000051b0 <L35>:
	s_wait_alu depctr_sa_sdst(0)                               // 0000000051B0: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s2                              // 0000000051B4: 8C7E027E
	v_mov_b32_e32 v2, 0xff0000                                 // 0000000051B8: 7E0402FF 00FF0000
	s_mov_b32 s2, exec_lo                                      // 0000000051C0: BE82007E
	v_cmpx_nlt_f32_e32 0x437f0000, v6                          // 0000000051C4: 7D3C0CFF 437F0000
	s_cbranch_execz L40                                        // 0000000051CC: BFA50036
	v_mov_b32_e32 v2, 0                                        // 0000000051D0: 7E040280
	s_mov_b32 s4, exec_lo                                      // 0000000051D4: BE84007E
	v_cmpx_ngt_f32_e32 0, v6                                   // 0000000051D8: 7D360C80
	s_cbranch_execz L39                                        // 0000000051DC: BFA50030
	s_mov_b32 s5, exec_lo                                      // 0000000051E0: BE85007E
	v_cmpx_o_f32_e32 v6, v6                                    // 0000000051E4: 7D2E0D06
	s_cbranch_execz L38                                        // 0000000051E8: BFA50028
	v_add_f32_e32 v2, v6, v6                                   // 0000000051EC: 06040D06
	v_cmp_nlg_f32_e32 vcc_lo, 0, v6                            // 0000000051F0: 7C340C80
	v_cmp_neq_f32_e64 s1, 0, v6                                // 0000000051F4: D41D0001 02020C80
	s_delay_alu instid0(VALU_DEP_3)                            // 0000000051FC: BF870003
	v_cmp_neq_f32_e64 s0, v2, v6                               // 000000005200: D41D0000 02020D02
	s_or_b32 s0, vcc_lo, s0                                    // 000000005208: 8C00006A
	s_wait_alu depctr_sa_sdst(0)                               // 00000000520C: BF88FF9E
	s_and_b32 s1, s1, s0                                       // 000000005210: 8B010001
	s_wait_alu depctr_sa_sdst(0)                               // 000000005214: BF88FF9E
	s_and_saveexec_b32 s0, s1                                  // 000000005218: BE802001
	s_cbranch_execz L37                                        // 00000000521C: BFA50019
	v_cvt_i32_f32_e32 v3, v6                                   // 000000005220: 7E061106
	s_mov_b32 s1, exec_lo                                      // 000000005224: BE81007E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000005228: BF870091
	v_cvt_f32_i32_e32 v2, v3                                   // 00000000522C: 7E040B03
	v_sub_f32_e32 v6, v6, v2                                   // 000000005230: 080C0506
	s_delay_alu instid0(VALU_DEP_1)                            // 000000005234: BF870001
	v_cmpx_ngt_f32_e32 0.5, v6                                 // 000000005238: 7D360CF0
	s_cbranch_execz L36                                        // 00000000523C: BFA5000D
	s_mov_b32 s6, exec_lo                                      // 000000005240: BE86007E
	v_cmpx_nlt_f32_e32 0.5, v6                                 // 000000005244: 7D3C0CF0
	s_xor_b32 s6, exec_lo, s6                                  // 000000005248: 8D06067E
	v_dual_add_f32 v6, 1.0, v2 :: v_dual_and_b32 v3, 1, v3     // 00000000524C: C92404F2 06020681
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000005254: BF870121
	v_cmp_eq_u32_e32 vcc_lo, 0, v3                             // 000000005258: 7C940680
	s_wait_alu depctr_va_vcc(0)                                // 00000000525C: BF88FF9D
	v_cndmask_b32_e32 v2, v6, v2, vcc_lo                       // 000000005260: 02040506
	s_and_not1_saveexec_b32 s6, s6                             // 000000005264: BE863006
	s_delay_alu instid0(VALU_DEP_1)                            // 000000005268: BF870001
	v_add_f32_e32 v2, 1.0, v2                                  // 00000000526C: 060404F2
	s_or_b32 exec_lo, exec_lo, s6                              // 000000005270: 8C7E067E

0000000000005274 <L36>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000005274: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005278: 8C7E017E
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000527C: BF870001
	v_mov_b32_e32 v6, v2                                       // 000000005280: 7E0C0302

0000000000005284 <L37>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000005284: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005288: 8C7E007E

000000000000528c <L38>:
	s_wait_alu depctr_sa_sdst(0)                               // 00000000528C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s5                              // 000000005290: 8C7E057E
	v_cvt_i32_f32_e32 v2, v6                                   // 000000005294: 7E041106
	s_delay_alu instid0(VALU_DEP_1)                            // 000000005298: BF870001
	v_lshlrev_b32_e32 v2, 16, v2                               // 00000000529C: 30040490

00000000000052a0 <L39>:
	s_wait_alu depctr_sa_sdst(0)                               // 0000000052A0: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s4                              // 0000000052A4: 8C7E047E

00000000000052a8 <L40>:
	s_wait_alu depctr_sa_sdst(0)                               // 0000000052A8: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s2                              // 0000000052AC: 8C7E027E
	s_wait_kmcnt 0x0                                           // 0000000052B0: BFC70000
	v_mul_lo_u32 v3, s3, v5                                    // 0000000052B4: D72C0003 02020A03
	v_and_or_b32 v0, 0xff00, v1, v0                            // 0000000052BC: D6570000 040202FF 0000FF00
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 0000000052C8: BF8700B2
	v_add_co_u32 v5, s0, s12, v3                               // 0000000052CC: D7000005 0202060C
	s_wait_alu depctr_va_sdst(0)                               // 0000000052D4: BF88F19F
	v_add_co_ci_u32_e64 v6, null, s13, 0, s0                   // 0000000052D8: D5207C06 0001000D
	v_mad_co_i64_i32 v[3:4], null, v4, 3, v[5:6]               // 0000000052E0: D6FF7C03 04150704
	s_clause 0x1                                               // 0000000052E8: BF850001
	global_store_d16_hi_b8 v[3:4], v2, off offset:2            // 0000000052EC: EE09007C 01000000 00000203
	global_store_b16 v[3:4], v0, off                           // 0000000052F8: EE06407C 00000000 00000003

0000000000005304 <L41>:
	s_endpgm                                                   // 000000005304: BFB00000
		...

0000000000005400 <_ZN2fk21linearFilterDPPKernelILNS_7ParArchE3ENS_22LinearFilterDPPDetailsINS_6Float3ELi16ELi8ELi7ELi7EEENS_5TupleIJNS_25ReadInstantiableOperationINS_15FusedOperation_IvJNS_29ReadBackInstantiableOperationINS_14ResizeCompleteILNS_11AspectRatioE1ENS_28TernaryInstantiableOperationINS_19InterpolateCompleteILNS_17InterpolationTypeE1ENS6_INS_13PerThreadReadILNS_2NDE2ENS_6Uchar3EEEEEEEEELNS_14ResizeSamplingE1EEEEENS_26UnaryInstantiableOperationINS_12SaturateCastIS3_SG_EEEENSO_INS_4CastISG_S3_EEEEEEEEENS6_INSE_ILSF_2ES3_EEEEEEENSO_INS_3MulIS3_S3_S3_NS_9UnaryTypeEEEEENSO_INS_3AddIS3_S3_S3_S11_EEEENS_26WriteInstantiableOperationINS7_IvJSR_NS17_INS_14PerThreadWriteILSF_2ESG_EEEEEEEEEEEvT0_T1_T2_T3_T4_>:
	s_clause 0x3                                               // 000000005400: BF850003
	s_load_b32 s2, s[0:1], 0x8c                                // 000000005404: F4000080 F800008C
	s_load_b128 s[4:7], s[0:1], 0x0                            // 00000000540C: F4004100 F8000000
	s_load_b64 s[8:9], s[0:1], 0x68                            // 000000005414: F4002200 F8000068
	s_load_b64 s[10:11], s[0:1], 0x48                          // 00000000541C: F4002280 F8000048
	v_bfe_u32 v3, v0, 10, 10                                   // 000000005424: D6100003 02291500
	v_and_b32_e32 v0, 0x3ff, v0                                // 00000000542C: 360000FF 000003FF
	s_lshl_b32 s15, ttmp9, 4                                   // 000000005434: 840F8475
	s_lshl_b32 s16, ttmp7, 3                                   // 000000005438: 84108373
	s_mov_b32 s19, exec_lo                                     // 00000000543C: BE93007E
	s_wait_kmcnt 0x0                                           // 000000005440: BFC70000
	s_and_b32 s3, s2, 0xffff                                   // 000000005444: 8B03FF02 0000FFFF
	s_add_co_i32 s14, s6, 15                                   // 00000000544C: 810E8F06
	v_mad_u32_u24 v1, v3, s3, v0                               // 000000005450: D60B0001 04000703
	s_add_co_i32 s18, s7, 7                                    // 000000005458: 81128707
	s_lshr_b32 s17, s2, 16                                     // 00000000545C: 85119002
	s_mul_i32 s18, s18, s14                                    // 000000005460: 96120E12
	s_mul_i32 s17, s17, s3                                     // 000000005464: 96110311
	v_cmpx_gt_i32_e32 s18, v1                                  // 000000005468: 7D880212
	s_cbranch_execz L59                                        // 00000000546C: BFA501A4
	s_abs_i32 s20, s14                                         // 000000005470: BE94150E
	s_add_nc_u64 s[2:3], s[0:1], 24                            // 000000005474: A9829800
	s_cvt_f32_u32 s12, s20                                     // 000000005478: BE8C6514
	s_clause 0x2                                               // 00000000547C: BF850002
	s_load_b64 s[26:27], s[0:1], 0x10                          // 000000005480: F4002680 F8000010
	s_load_b64 s[28:29], s[0:1], 0x38                          // 000000005488: F4002700 F8000038
	s_load_b64 s[30:31], s[2:3], 0x8                           // 000000005490: F4002781 F8000008
	v_s_rcp_f32 s23, s12                                       // 000000005498: D6840017 0201000C
	s_clause 0x1                                               // 0000000054A0: BF850001
	s_load_b64 s[12:13], s[2:3], 0x18                          // 0000000054A4: F4002301 F8000018
	s_load_b32 s21, s[2:3], 0x28                               // 0000000054AC: F4000541 F8000028
	s_sub_co_i32 s3, 0, s20                                    // 0000000054B4: 81831480
	v_mov_b32_e32 v2, v1                                       // 0000000054B8: 7E040301
	s_add_co_i32 s22, s4, -1                                   // 0000000054BC: 8116C104
	s_ashr_i32 s24, s14, 31                                    // 0000000054C0: 86189F0E
	s_mul_f32 s2, s23, 0x4f7ffffe                              // 0000000054C4: A202FF17 4F7FFFFE
	s_add_co_i32 s23, s5, -1                                   // 0000000054CC: 8117C105
	s_wait_alu depctr_sa_sdst(0)                               // 0000000054D0: BF88FF9E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 0000000054D4: BF870529
	s_cvt_u32_f32 s2, s2                                       // 0000000054D8: BE826702
	s_wait_alu depctr_sa_sdst(0)                               // 0000000054DC: BF88FF9E
	s_mul_i32 s3, s3, s2                                       // 0000000054E0: 96030203
	s_wait_kmcnt 0x0                                           // 0000000054E4: BFC70000
	s_sub_co_i32 s25, s15, s26                                 // 0000000054E8: 81991A0F
	s_wait_alu depctr_sa_sdst(0)                               // 0000000054EC: BF88FF9E
	s_mul_hi_u32 s3, s2, s3                                    // 0000000054F0: 96830302
	s_sub_co_i32 s26, s16, s27                                 // 0000000054F4: 819A1B10
	s_add_co_i32 s27, s28, -1                                  // 0000000054F8: 811BC11C
	s_add_co_i32 s28, s29, -1                                  // 0000000054FC: 811CC11D
	s_mov_b32 s29, s31                                         // 000000005500: BE9D001F
	s_mov_b32 s31, 0                                           // 000000005504: BE9F0080
	s_wait_alu depctr_sa_sdst(0)                               // 000000005508: BF88FF9E
	s_add_co_i32 s33, s2, s3                                   // 00000000550C: 81210302
	s_branch L48                                               // 000000005510: BFA0001C

0000000000005514 <L42>:
	s_or_b32 exec_lo, exec_lo, s37                             // 000000005514: 8C7E257E

0000000000005518 <L43>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000005518: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s3                              // 00000000551C: 8C7E037E
	v_mov_b32_e32 v5, v6                                       // 000000005520: 7E0A0306

0000000000005524 <L44>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000005524: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s2                              // 000000005528: 8C7E027E

000000000000552c <L45>:
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000552C: BF870099
	s_or_b32 exec_lo, exec_lo, s36                             // 000000005530: 8C7E247E
	v_cvt_i32_f32_e32 v5, v5                                   // 000000005534: 7E0A1105
	s_delay_alu instid0(VALU_DEP_1)                            // 000000005538: BF870001
	v_lshlrev_b32_e32 v6, 16, v5                               // 00000000553C: 300C0A90

0000000000005540 <L46>:
	s_or_b32 exec_lo, exec_lo, s35                             // 000000005540: 8C7E237E

0000000000005544 <L47>:
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000005544: BF870009
	s_or_b32 exec_lo, exec_lo, s34                             // 000000005548: 8C7E227E
	v_mul_lo_u32 v5, v2, 12                                    // 00000000554C: D72C0005 02011902
	v_add_nc_u32_e32 v2, s17, v2                               // 000000005554: 4A040411
	v_cvt_f32_ubyte1_e32 v7, v9                                // 000000005558: 7E0E2509
	v_cvt_f32_ubyte2_e32 v6, v6                                // 00000000555C: 7E0C2706
	ds_store_2addr_b32 v5, v4, v7 offset1:1                    // 000000005560: D8380100 00070405
	ds_store_b32 v5, v6 offset:8                               // 000000005568: D8340008 00000605
	v_cmp_le_i32_e32 vcc_lo, s18, v2                           // 000000005570: 7C860412
	s_or_b32 s31, vcc_lo, s31                                  // 000000005574: 8C1F1F6A
	s_wait_alu depctr_sa_sdst(0)                               // 000000005578: BF88FF9E
	s_and_not1_b32 exec_lo, exec_lo, s31                       // 00000000557C: 917E1F7E
	s_cbranch_execz L59                                        // 000000005580: BFA5015F

0000000000005584 <L48>:
	v_sub_nc_u32_e32 v4, 0, v2                                 // 000000005584: 4C080480
	s_mov_b32 s34, exec_lo                                     // 000000005588: BEA2007E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000558C: BF870091
	v_max_i32_e32 v4, v4, v2                                   // 000000005590: 24080504
	v_mul_hi_u32 v5, v4, s33                                   // 000000005594: D72D0005 02004304
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000559C: BF870091
	v_mul_lo_u32 v6, v5, s20                                   // 0000000055A0: D72C0006 02002905
	v_sub_nc_u32_e32 v4, v4, v6                                // 0000000055A8: 4C080D04
	v_add_nc_u32_e32 v6, 1, v5                                 // 0000000055AC: 4A0C0A81
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 0000000055B0: BF870132
	v_subrev_nc_u32_e32 v7, s20, v4                            // 0000000055B4: 4E0E0814
	v_cmp_le_u32_e32 vcc_lo, s20, v4                           // 0000000055B8: 7C960814
	s_wait_alu depctr_va_vcc(0)                                // 0000000055BC: BF88FF9D
	v_dual_cndmask_b32 v5, v5, v6 :: v_dual_cndmask_b32 v4, v4, v7// 0000000055C0: CA520D05 05040F04
	v_ashrrev_i32_e32 v6, 31, v2                               // 0000000055C8: 340C049F
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)// 0000000055CC: BF870192
	v_add_nc_u32_e32 v7, 1, v5                                 // 0000000055D0: 4A0E0A81
	v_cmp_le_u32_e32 vcc_lo, s20, v4                           // 0000000055D4: 7C960814
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)// 0000000055D8: BF8701A3
	v_xor_b32_e32 v6, s24, v6                                  // 0000000055DC: 3A0C0C18
	s_wait_alu depctr_va_vcc(0)                                // 0000000055E0: BF88FF9D
	v_cndmask_b32_e32 v4, v5, v7, vcc_lo                       // 0000000055E4: 02080F05
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000055E8: BF870091
	v_xor_b32_e32 v4, v4, v6                                   // 0000000055EC: 3A080D04
	v_sub_nc_u32_e32 v4, v4, v6                                // 0000000055F0: 4C080D04
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000055F4: BF8700A1
	v_mul_lo_u32 v5, v4, s14                                   // 0000000055F8: D72C0005 02001D04
	v_add_nc_u32_e32 v4, s26, v4                               // 000000005600: 4A08081A
	v_min_i32_e32 v6, s23, v4                                  // 000000005604: 220C0817
	v_cmp_lt_i32_e32 vcc_lo, -1, v4                            // 000000005608: 7C8208C1
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 00000000560C: BF8700A4
	v_sub_nc_u32_e32 v5, v2, v5                                // 000000005610: 4C0A0B02
	s_wait_alu depctr_va_vcc(0)                                // 000000005614: BF88FF9D
	v_dual_cndmask_b32 v4, 0, v6 :: v_dual_add_nc_u32 v5, s25, v5// 000000005618: CA600C80 04040A19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000005620: BF870111
	v_cvt_f32_i32_e32 v4, v4                                   // 000000005624: 7E080B04
	v_min_i32_e32 v6, s22, v5                                  // 000000005628: 220C0A16
	v_cmp_lt_i32_e32 vcc_lo, -1, v5                            // 00000000562C: 7C820AC1
	s_wait_alu depctr_va_vcc(0)                                // 000000005630: BF88FF9D
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000005634: BF870092
	v_dual_add_f32 v4, 0.5, v4 :: v_dual_cndmask_b32 v5, 0, v6 // 000000005638: C91208F0 04040C80
	v_fma_f32 v4, s29, v4, -0.5                                // 000000005640: D6130004 03C6081D
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000005648: BF870112
	v_cvt_f32_i32_e32 v5, v5                                   // 00000000564C: 7E0A0B05
	v_cmp_le_f32_e32 vcc_lo, 0, v4                             // 000000005650: 7C260880
	s_wait_alu depctr_va_vcc(0)                                // 000000005654: BF88FF9D
	v_cndmask_b32_e32 v8, 0, v4, vcc_lo                        // 000000005658: 02100880
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000565C: BF870113
	v_add_f32_e32 v4, 0.5, v5                                  // 000000005660: 06080AF0
	v_floor_f32_e32 v5, v8                                     // 000000005664: 7E0A4908
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000005668: BF870112
	v_fma_f32 v4, s30, v4, -0.5                                // 00000000566C: D6130004 03C6081E
	v_cvt_i32_f32_e32 v21, v5                                  // 000000005674: 7E2A1105
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000005678: BF870122
	v_cmp_le_f32_e32 vcc_lo, 0, v4                             // 00000000567C: 7C260880
	s_wait_alu depctr_va_vcc(0)                                // 000000005680: BF88FF9D
	v_dual_cndmask_b32 v22, 0, v4 :: v_dual_add_nc_u32 v23, 1, v21// 000000005684: CA600880 16162A81
	v_mul_lo_u32 v4, s21, v21                                  // 00000000568C: D72C0004 02022A15
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000005694: BF870192
	v_min_i32_e32 v6, s28, v23                                 // 000000005698: 220C2E1C
	v_floor_f32_e32 v5, v22                                    // 00000000569C: 7E0A4916
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000056A0: BF870113
	v_add_co_u32 v4, s2, s12, v4                               // 0000000056A4: D7000204 0202080C
	v_cvt_i32_f32_e32 v24, v5                                  // 0000000056AC: 7E301105
	s_wait_alu depctr_va_sdst(0)                               // 0000000056B0: BF88F19F
	v_add_co_ci_u32_e64 v5, null, s13, 0, s2                   // 0000000056B4: D5207C05 0009000D
	v_mul_lo_u32 v9, v6, s21                                   // 0000000056BC: D72C0009 02002B06
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 0000000056C4: BF870193
	v_add_nc_u32_e32 v25, 1, v24                               // 0000000056C8: 4A323081
	v_mad_co_i64_i32 v[6:7], null, v24, 3, v[4:5]              // 0000000056CC: D6FF7C06 04110718
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)// 0000000056D4: BF870212
	v_min_i32_e32 v10, s27, v25                                // 0000000056D8: 2214321B
	v_add_co_u32 v12, s2, s12, v9                              // 0000000056DC: D700020C 0202120C
	s_wait_alu depctr_va_sdst(0)                               // 0000000056E4: BF88F19F
	v_add_co_ci_u32_e64 v13, null, s13, 0, s2                  // 0000000056E8: D5207C0D 0009000D
	global_load_u16 v9, v[6:7], off                            // 0000000056F0: EE04807C 00000009 00000006
	v_mad_co_i64_i32 v[4:5], null, v10, 3, v[4:5]              // 0000000056FC: D6FF7C04 0411070A
	v_mad_co_i64_i32 v[17:18], null, v24, 3, v[12:13]          // 000000005704: D6FF7C11 04310718
	v_mad_co_i64_i32 v[19:20], null, v10, 3, v[12:13]          // 00000000570C: D6FF7C13 0431070A
	s_clause 0x6                                               // 000000005714: BF850006
	global_load_u16 v12, v[17:18], off                         // 000000005718: EE04807C 0000000C 00000011
	global_load_u16 v10, v[19:20], off                         // 000000005724: EE04807C 0000000A 00000013
	global_load_u8 v15, v[6:7], off offset:2                   // 000000005730: EE04007C 0000000F 00000206
	global_load_u8 v16, v[4:5], off offset:2                   // 00000000573C: EE04007C 00000010 00000204
	global_load_u8 v14, v[17:18], off offset:2                 // 000000005748: EE04007C 0000000E 00000211
	global_load_u8 v13, v[19:20], off offset:2                 // 000000005754: EE04007C 0000000D 00000213
	global_load_u16 v11, v[4:5], off                           // 000000005760: EE04807C 0000000B 00000004
	v_cvt_f32_i32_e32 v4, v23                                  // 00000000576C: 7E080B17
	v_cvt_f32_i32_e32 v5, v25                                  // 000000005770: 7E0A0B19
	v_cvt_f32_i32_e32 v6, v24                                  // 000000005774: 7E0C0B18
	v_cvt_f32_i32_e32 v17, v21                                 // 000000005778: 7E220B15
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 00000000577C: BF8700C3
	v_sub_f32_e32 v7, v5, v22                                  // 000000005780: 080E2D05
	s_wait_loadcnt 0x7                                         // 000000005784: BFC00007
	v_cvt_f32_ubyte0_e32 v19, v9                               // 000000005788: 7E262309
	v_sub_f32_e32 v4, v4, v8                                   // 00000000578C: 08081104
	v_mul_f32_e32 v5, v4, v7                                   // 000000005790: 100A0F04
	v_sub_f32_e32 v8, v8, v17                                  // 000000005794: 08102308
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000005798: BF870112
	v_dual_sub_f32 v18, v22, v6 :: v_dual_mul_f32 v17, v5, v19 // 00000000579C: C9460D16 12102705
	v_mul_f32_e32 v7, v7, v8                                   // 0000000057A4: 100E1107
	s_wait_loadcnt 0x6                                         // 0000000057A8: BFC00006
	v_cvt_f32_ubyte0_e32 v19, v12                              // 0000000057AC: 7E26230C
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 0000000057B0: BF8700B3
	v_mul_f32_e32 v6, v18, v4                                  // 0000000057B4: 100C0912
	s_wait_loadcnt 0x0                                         // 0000000057B8: BFC00000
	v_cvt_f32_ubyte0_e32 v4, v11                               // 0000000057BC: 7E08230B
	v_dual_fmac_f32 v17, v6, v4 :: v_dual_mul_f32 v8, v8, v18  // 0000000057C0: C8060906 11082508
	v_cvt_f32_ubyte0_e32 v4, v10                               // 0000000057C8: 7E08230A
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000057CC: BF870092
	v_fmac_f32_e32 v17, v7, v19                                // 0000000057D0: 56222707
	v_dual_fmac_f32 v17, v8, v4 :: v_dual_mov_b32 v4, 0x437f0000// 0000000057D4: C8100908 110400FF 437F0000
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000057E0: BF870001
	v_cmpx_nlt_f32_e32 0x437f0000, v17                         // 0000000057E4: 7D3C22FF 437F0000
	s_cbranch_execz L53                                        // 0000000057EC: BFA50035
	v_mov_b32_e32 v4, 0                                        // 0000000057F0: 7E080280
	s_mov_b32 s35, exec_lo                                     // 0000000057F4: BEA3007E
	v_cmpx_ngt_f32_e32 0, v17                                  // 0000000057F8: 7D362280
	s_cbranch_execz L52                                        // 0000000057FC: BFA50030
	s_mov_b32 s36, exec_lo                                     // 000000005800: BEA4007E
	v_cmpx_o_f32_e32 v17, v17                                  // 000000005804: 7D2E2311
	s_cbranch_execz L51                                        // 000000005808: BFA50029
	v_add_f32_e32 v4, v17, v17                                 // 00000000580C: 06082311
	v_cmp_nlg_f32_e32 vcc_lo, 0, v17                           // 000000005810: 7C342280
	v_cmp_neq_f32_e64 s3, 0, v17                               // 000000005814: D41D0003 02022280
	s_delay_alu instid0(VALU_DEP_3)                            // 00000000581C: BF870003
	v_cmp_neq_f32_e64 s2, v4, v17                              // 000000005820: D41D0002 02022304
	s_or_b32 s2, vcc_lo, s2                                    // 000000005828: 8C02026A
	s_wait_alu depctr_sa_sdst(0)                               // 00000000582C: BF88FF9E
	s_and_b32 s3, s3, s2                                       // 000000005830: 8B030203
	s_wait_alu depctr_sa_sdst(0)                               // 000000005834: BF88FF9E
	s_and_saveexec_b32 s2, s3                                  // 000000005838: BE822003
	s_cbranch_execz L50                                        // 00000000583C: BFA5001A
	v_cvt_i32_f32_e32 v18, v17                                 // 000000005840: 7E241111
	s_mov_b32 s3, exec_lo                                      // 000000005844: BE83007E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000005848: BF870091
	v_cvt_f32_i32_e32 v4, v18                                  // 00000000584C: 7E080B12
	v_sub_f32_e32 v17, v17, v4                                 // 000000005850: 08220911
	s_delay_alu instid0(VALU_DEP_1)                            // 000000005854: BF870001
	v_cmpx_ngt_f32_e32 0.5, v17                                // 000000005858: 7D3622F0
	s_cbranch_execz L49                                        // 00000000585C: BFA5000E
	v_cmp_nlt_f32_e32 vcc_lo, 0.5, v17                         // 000000005860: 7C3C22F0
	v_add_f32_e32 v17, 1.0, v4                                 // 000000005864: 062208F2
	s_and_saveexec_b32 s37, vcc_lo                             // 000000005868: BEA5206A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000586C: BF870009
	s_xor_b32 s37, exec_lo, s37                                // 000000005870: 8D25257E
	v_dual_add_f32 v18, 1.0, v4 :: v_dual_and_b32 v17, 1, v18  // 000000005874: C92408F2 12102481
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 00000000587C: BF870121
	v_cmp_eq_u32_e32 vcc_lo, 0, v17                            // 000000005880: 7C942280
	s_wait_alu depctr_va_vcc(0)                                // 000000005884: BF88FF9D
	v_cndmask_b32_e32 v4, v18, v4, vcc_lo                      // 000000005888: 02080912
	s_and_not1_saveexec_b32 s37, s37                           // 00000000588C: BEA53025
	v_mov_b32_e32 v4, v17                                      // 000000005890: 7E080311
	s_or_b32 exec_lo, exec_lo, s37                             // 000000005894: 8C7E257E

0000000000005898 <L49>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000005898: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s3                              // 00000000589C: 8C7E037E
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000058A0: BF870001
	v_mov_b32_e32 v17, v4                                      // 0000000058A4: 7E220304

00000000000058a8 <L50>:
	s_wait_alu depctr_sa_sdst(0)                               // 0000000058A8: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s2                              // 0000000058AC: 8C7E027E

00000000000058b0 <L51>:
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000058B0: BF870099
	s_or_b32 exec_lo, exec_lo, s36                             // 0000000058B4: 8C7E247E
	v_trunc_f32_e64 v4, |v17|                                  // 0000000058B8: D5A10104 02010111

00000000000058c0 <L52>:
	s_or_b32 exec_lo, exec_lo, s35                             // 0000000058C0: 8C7E237E

00000000000058c4 <L53>:
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000058C4: BF870009
	s_or_b32 exec_lo, exec_lo, s34                             // 0000000058C8: 8C7E227E
	v_lshl_or_b32 v15, v15, 16, v9                             // 0000000058CC: D656000F 0425210F
	v_lshl_or_b32 v11, v16, 16, v11                            // 0000000058D4: D656000B 042D2110
	v_lshl_or_b32 v12, v14, 16, v12                            // 0000000058DC: D656000C 0431210E
	v_lshl_or_b32 v10, v13, 16, v10                            // 0000000058E4: D656000A 0429210D
	s_mov_b32 s34, exec_lo                                     // 0000000058EC: BEA2007E
	v_cvt_f32_ubyte1_e32 v9, v15                               // 0000000058F0: 7E12250F
	v_cvt_f32_ubyte1_e32 v16, v11                              // 0000000058F4: 7E20250B
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 0000000058F8: BF870193
	v_cvt_f32_ubyte1_e32 v13, v10                              // 0000000058FC: 7E1A250A
	v_mul_f32_e32 v14, v5, v9                                  // 000000005900: 101C1305
	v_cvt_f32_ubyte1_e32 v9, v12                               // 000000005904: 7E12250C
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000005908: BF870092
	v_fmac_f32_e32 v14, v6, v16                                // 00000000590C: 561C2106
	v_dual_fmac_f32 v14, v7, v9 :: v_dual_mov_b32 v9, 0xff00   // 000000005910: C8101307 0E0800FF 0000FF00
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000591C: BF870091
	v_fmac_f32_e32 v14, v8, v13                                // 000000005920: 561C1B08
	v_cmpx_nlt_f32_e32 0x437f0000, v14                         // 000000005924: 7D3C1CFF 437F0000
	s_cbranch_execz L58                                        // 00000000592C: BFA50037
	v_mov_b32_e32 v9, 0                                        // 000000005930: 7E120280
	s_mov_b32 s35, exec_lo                                     // 000000005934: BEA3007E
	v_cmpx_ngt_f32_e32 0, v14                                  // 000000005938: 7D361C80
	s_cbranch_execz L57                                        // 00000000593C: BFA50032
	s_mov_b32 s36, exec_lo                                     // 000000005940: BEA4007E
	v_cmpx_o_f32_e32 v14, v14                                  // 000000005944: 7D2E1D0E
	s_cbranch_execz L56                                        // 000000005948: BFA50029
	v_add_f32_e32 v9, v14, v14                                 // 00000000594C: 06121D0E
	v_cmp_nlg_f32_e32 vcc_lo, 0, v14                           // 000000005950: 7C341C80
	v_cmp_neq_f32_e64 s3, 0, v14                               // 000000005954: D41D0003 02021C80
	s_delay_alu instid0(VALU_DEP_3)                            // 00000000595C: BF870003
	v_cmp_neq_f32_e64 s2, v9, v14                              // 000000005960: D41D0002 02021D09
	s_or_b32 s2, vcc_lo, s2                                    // 000000005968: 8C02026A
	s_wait_alu depctr_sa_sdst(0)                               // 00000000596C: BF88FF9E
	s_and_b32 s3, s3, s2                                       // 000000005970: 8B030203
	s_wait_alu depctr_sa_sdst(0)                               // 000000005974: BF88FF9E
	s_and_saveexec_b32 s2, s3                                  // 000000005978: BE822003
	s_cbranch_execz L55                                        // 00000000597C: BFA5001A
	v_cvt_i32_f32_e32 v13, v14                                 // 000000005980: 7E1A110E
	s_mov_b32 s3, exec_lo                                      // 000000005984: BE83007E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000005988: BF870091
	v_cvt_f32_i32_e32 v9, v13                                  // 00000000598C: 7E120B0D
	v_sub_f32_e32 v14, v14, v9                                 // 000000005990: 081C130E
	s_delay_alu instid0(VALU_DEP_1)                            // 000000005994: BF870001
	v_cmpx_ngt_f32_e32 0.5, v14                                // 000000005998: 7D361CF0
	s_cbranch_execz L54                                        // 00000000599C: BFA5000E
	v_cmp_nlt_f32_e32 vcc_lo, 0.5, v14                         // 0000000059A0: 7C3C1CF0
	v_add_f32_e32 v14, 1.0, v9                                 // 0000000059A4: 061C12F2
	s_and_saveexec_b32 s37, vcc_lo                             // 0000000059A8: BEA5206A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000059AC: BF870009
	s_xor_b32 s37, exec_lo, s37                                // 0000000059B0: 8D25257E
	v_and_b32_e32 v13, 1, v13                                  // 0000000059B4: 361A1A81
	v_add_f32_e32 v14, 1.0, v9                                 // 0000000059B8: 061C12F2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 0000000059BC: BF870122
	v_cmp_eq_u32_e32 vcc_lo, 0, v13                            // 0000000059C0: 7C941A80
	s_wait_alu depctr_va_vcc(0)                                // 0000000059C4: BF88FF9D
	v_cndmask_b32_e32 v9, v14, v9, vcc_lo                      // 0000000059C8: 0212130E
	s_and_not1_saveexec_b32 s37, s37                           // 0000000059CC: BEA53025
	v_mov_b32_e32 v9, v14                                      // 0000000059D0: 7E12030E
	s_or_b32 exec_lo, exec_lo, s37                             // 0000000059D4: 8C7E257E

00000000000059d8 <L54>:
	s_wait_alu depctr_sa_sdst(0)                               // 0000000059D8: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s3                              // 0000000059DC: 8C7E037E
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000059E0: BF870001
	v_mov_b32_e32 v14, v9                                      // 0000000059E4: 7E1C0309

00000000000059e8 <L55>:
	s_wait_alu depctr_sa_sdst(0)                               // 0000000059E8: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s2                              // 0000000059EC: 8C7E027E

00000000000059f0 <L56>:
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000059F0: BF870099
	s_or_b32 exec_lo, exec_lo, s36                             // 0000000059F4: 8C7E247E
	v_cvt_i32_f32_e32 v9, v14                                  // 0000000059F8: 7E12110E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000059FC: BF870091
	v_cvt_u32_u16_e32 v9, v9.l                                 // 000000005A00: 7E12D709
	v_lshlrev_b32_e32 v9, 8, v9                                // 000000005A04: 30121288

0000000000005a08 <L57>:
	s_or_b32 exec_lo, exec_lo, s35                             // 000000005A08: 8C7E237E

0000000000005a0c <L58>:
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(VALU_DEP_3)// 000000005A0C: BF8701D9
	s_or_b32 exec_lo, exec_lo, s34                             // 000000005A10: 8C7E227E
	v_cvt_f32_ubyte2_e32 v13, v15                              // 000000005A14: 7E1A270F
	v_cvt_f32_ubyte2_e32 v11, v11                              // 000000005A18: 7E16270B
	v_cvt_f32_ubyte2_e32 v12, v12                              // 000000005A1C: 7E18270C
	s_mov_b32 s34, exec_lo                                     // 000000005A20: BEA2007E
	v_mul_f32_e32 v5, v5, v13                                  // 000000005A24: 100A1B05
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000005A28: BF870121
	v_fmac_f32_e32 v5, v6, v11                                 // 000000005A2C: 560A1706
	v_cvt_f32_ubyte2_e32 v6, v10                               // 000000005A30: 7E0C270A
	v_fmac_f32_e32 v5, v7, v12                                 // 000000005A34: 560A1907
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000005A38: BF870091
	v_dual_fmac_f32 v5, v8, v6 :: v_dual_mov_b32 v6, 0xff0000  // 000000005A3C: C8100D08 050600FF 00FF0000
	v_cmpx_nlt_f32_e32 0x437f0000, v5                          // 000000005A48: 7D3C0AFF 437F0000
	s_cbranch_execz L47                                        // 000000005A50: BFA5FEBC
	v_mov_b32_e32 v6, 0                                        // 000000005A54: 7E0C0280
	s_mov_b32 s35, exec_lo                                     // 000000005A58: BEA3007E
	v_cmpx_ngt_f32_e32 0, v5                                   // 000000005A5C: 7D360A80
	s_cbranch_execz L46                                        // 000000005A60: BFA5FEB7
	s_mov_b32 s36, exec_lo                                     // 000000005A64: BEA4007E
	v_cmpx_o_f32_e32 v5, v5                                    // 000000005A68: 7D2E0B05
	s_cbranch_execz L45                                        // 000000005A6C: BFA5FEAF
	v_add_f32_e32 v6, v5, v5                                   // 000000005A70: 060C0B05
	v_cmp_nlg_f32_e32 vcc_lo, 0, v5                            // 000000005A74: 7C340A80
	v_cmp_neq_f32_e64 s3, 0, v5                                // 000000005A78: D41D0003 02020A80
	s_delay_alu instid0(VALU_DEP_3)                            // 000000005A80: BF870003
	v_cmp_neq_f32_e64 s2, v6, v5                               // 000000005A84: D41D0002 02020B06
	s_or_b32 s2, vcc_lo, s2                                    // 000000005A8C: 8C02026A
	s_wait_alu depctr_sa_sdst(0)                               // 000000005A90: BF88FF9E
	s_and_b32 s3, s3, s2                                       // 000000005A94: 8B030203
	s_wait_alu depctr_sa_sdst(0)                               // 000000005A98: BF88FF9E
	s_and_saveexec_b32 s2, s3                                  // 000000005A9C: BE822003
	s_cbranch_execz L44                                        // 000000005AA0: BFA5FEA0
	v_cvt_i32_f32_e32 v7, v5                                   // 000000005AA4: 7E0E1105
	s_mov_b32 s3, exec_lo                                      // 000000005AA8: BE83007E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000005AAC: BF870091
	v_cvt_f32_i32_e32 v6, v7                                   // 000000005AB0: 7E0C0B07
	v_sub_f32_e32 v5, v5, v6                                   // 000000005AB4: 080A0D05
	s_delay_alu instid0(VALU_DEP_1)                            // 000000005AB8: BF870001
	v_cmpx_ngt_f32_e32 0.5, v5                                 // 000000005ABC: 7D360AF0
	s_cbranch_execz L43                                        // 000000005AC0: BFA5FE95
	v_cmp_nlt_f32_e32 vcc_lo, 0.5, v5                          // 000000005AC4: 7C3C0AF0
	v_add_f32_e32 v5, 1.0, v6                                  // 000000005AC8: 060A0CF2
	s_and_saveexec_b32 s37, vcc_lo                             // 000000005ACC: BEA5206A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000005AD0: BF870009
	s_xor_b32 s37, exec_lo, s37                                // 000000005AD4: 8D25257E
	v_and_b32_e32 v5, 1, v7                                    // 000000005AD8: 360A0E81
	v_add_f32_e32 v7, 1.0, v6                                  // 000000005ADC: 060E0CF2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000005AE0: BF870122
	v_cmp_eq_u32_e32 vcc_lo, 0, v5                             // 000000005AE4: 7C940A80
	s_wait_alu depctr_va_vcc(0)                                // 000000005AE8: BF88FF9D
	v_cndmask_b32_e32 v6, v7, v6, vcc_lo                       // 000000005AEC: 020C0D07
	s_and_not1_saveexec_b32 s37, s37                           // 000000005AF0: BEA53025
	s_cbranch_execz L42                                        // 000000005AF4: BFA5FE87
	v_mov_b32_e32 v6, v5                                       // 000000005AF8: 7E0C0305
	s_branch L42                                               // 000000005AFC: BFA0FE85

0000000000005b00 <L59>:
	s_or_b32 exec_lo, exec_lo, s19                             // 000000005B00: 8C7E137E
	s_clause 0x1                                               // 000000005B04: BF850001
	s_load_b32 s2, s[0:1], 0x58                                // 000000005B08: F4000080 F8000058
	s_load_b32 s3, s[0:1], 0x78                                // 000000005B10: F40000C0 F8000078
	s_mul_i32 s0, s7, s6                                       // 000000005B18: 96000607
	s_mov_b32 s1, exec_lo                                      // 000000005B1C: BE81007E
	v_cmpx_gt_i32_e32 s0, v1                                   // 000000005B20: 7D880200
	s_cbranch_execz L61                                        // 000000005B24: BFA50054
	s_abs_i32 s12, s6                                          // 000000005B28: BE8C1506
	s_wait_alu depctr_sa_sdst(0)                               // 000000005B2C: BF88FF9E
	s_cvt_f32_u32 s13, s12                                     // 000000005B30: BE8D650C
	s_wait_alu depctr_sa_sdst(0)                               // 000000005B34: BF88FF9E
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)// 000000005B38: BF87029A
	v_s_rcp_f32 s13, s13                                       // 000000005B3C: D684000D 0201000D
	s_mul_f32 s13, s13, 0x4f7ffffe                             // 000000005B44: A20DFF0D 4F7FFFFE
	s_wait_alu depctr_sa_sdst(0)                               // 000000005B4C: BF88FF9E
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000005B50: BF8704BA
	s_cvt_u32_f32 s19, s13                                     // 000000005B54: BE93670D
	s_sub_co_i32 s13, 0, s12                                   // 000000005B58: 818D0C80
	s_wait_alu depctr_sa_sdst(0)                               // 000000005B5C: BF88FF9E
	s_mul_i32 s18, s13, s19                                    // 000000005B60: 9612130D
	s_ashr_i32 s13, s6, 31                                     // 000000005B64: 860D9F06
	s_wait_alu depctr_sa_sdst(0)                               // 000000005B68: BF88FF9E
	s_mul_hi_u32 s20, s19, s18                                 // 000000005B6C: 96941213
	s_mov_b32 s18, 0                                           // 000000005B70: BE920080
	s_wait_alu depctr_sa_sdst(0)                               // 000000005B74: BF88FF9E
	s_add_co_i32 s19, s19, s20                                 // 000000005B78: 81131413

0000000000005b7c <L60>:
	v_sub_nc_u32_e32 v2, 0, v1                                 // 000000005B7C: 4C040280
	v_ashrrev_i32_e32 v4, 31, v1                               // 000000005B80: 3408029F
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000005B84: BF870112
	v_max_i32_e32 v2, v2, v1                                   // 000000005B88: 24040302
	v_xor_b32_e32 v4, s13, v4                                  // 000000005B8C: 3A08080D
	s_wait_alu depctr_sa_sdst(0)                               // 000000005B90: BF88FF9E
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000005B94: BF870092
	v_mul_hi_u32 v5, v2, s19                                   // 000000005B98: D72D0005 02002702
	v_mul_lo_u32 v6, v5, s12                                   // 000000005BA0: D72C0006 02001905
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000005BA8: BF870091
	v_sub_nc_u32_e32 v2, v2, v6                                // 000000005BAC: 4C040D02
	v_subrev_nc_u32_e32 v6, s12, v2                            // 000000005BB0: 4E0C040C
	v_cmp_le_u32_e32 vcc_lo, s12, v2                           // 000000005BB4: 7C96040C
	s_wait_alu depctr_va_vcc(0)                                // 000000005BB8: BF88FF9D
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000005BBC: BF870092
	v_dual_cndmask_b32 v2, v2, v6 :: v_dual_add_nc_u32 v7, 1, v5// 000000005BC0: CA600D02 02060A81
	v_cndmask_b32_e32 v5, v5, v7, vcc_lo                       // 000000005BC8: 020A0F05
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000005BCC: BF870112
	v_cmp_le_u32_e32 vcc_lo, s12, v2                           // 000000005BD0: 7C96040C
	v_add_nc_u32_e32 v6, 1, v5                                 // 000000005BD4: 4A0C0A81
	s_wait_alu depctr_va_vcc(0)                                // 000000005BD8: BF88FF9D
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000005BDC: BF870091
	v_cndmask_b32_e32 v2, v5, v6, vcc_lo                       // 000000005BE0: 02040D05
	v_xor_b32_e32 v2, v2, v4                                   // 000000005BE4: 3A040902
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000005BE8: BF870091
	v_sub_nc_u32_e32 v2, v2, v4                                // 000000005BEC: 4C040902
	v_mul_lo_u32 v4, v2, s6                                    // 000000005BF0: D72C0004 02000D02
	s_wait_kmcnt 0x0                                           // 000000005BF8: BFC70000
	v_mul_lo_u32 v2, v2, s2                                    // 000000005BFC: D72C0002 02000502
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000005C04: BF870112
	v_sub_nc_u32_e32 v6, v1, v4                                // 000000005C08: 4C0C0901
	v_add_co_u32 v4, s20, s10, v2                              // 000000005C0C: D7001404 0202040A
	s_wait_alu depctr_va_sdst(0)                               // 000000005C14: BF88F19F
	v_add_co_ci_u32_e64 v5, null, s11, 0, s20                  // 000000005C18: D5207C05 0051000B
	v_mul_lo_u32 v2, v1, 12                                    // 000000005C20: D72C0002 02011901
	v_add_nc_u32_e32 v1, s17, v1                               // 000000005C28: 4A020211
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000005C2C: BF870113
	v_mad_co_u64_u32 v[4:5], null, v6, 12, v[4:5]              // 000000005C30: D6FE7C04 04111906
	v_cmp_le_i32_e32 vcc_lo, s0, v1                            // 000000005C38: 7C860200
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005C3C: BF870004
	v_add_nc_u32_e32 v7, 0xe70, v2                             // 000000005C40: 4A0E04FF 00000E70
	global_load_b96 v[4:6], v[4:5], off                        // 000000005C48: EE05807C 00000004 00000004
	s_or_b32 s18, vcc_lo, s18                                  // 000000005C54: 8C12126A
	s_wait_loadcnt 0x0                                         // 000000005C58: BFC00000
	ds_store_2addr_b32 v7, v4, v5 offset1:1                    // 000000005C5C: D8380100 00050407
	ds_store_b32 v2, v6 offset:3704                            // 000000005C64: D8340E78 00000602
	s_wait_alu depctr_sa_sdst(0)                               // 000000005C6C: BF88FF9E
	s_and_not1_b32 exec_lo, exec_lo, s18                       // 000000005C70: 917E127E
	s_cbranch_execnz L60                                       // 000000005C74: BFA6FFC1

0000000000005c78 <L61>:
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005C78: 8C7E017E
	s_wait_dscnt 0x0                                           // 000000005C7C: BFC60000
	s_barrier_signal -1                                        // 000000005C80: BE804EC1
	v_add_nc_u32_e32 v4, s15, v0                               // 000000005C84: 4A08000F
	v_add_nc_u32_e32 v5, s16, v3                               // 000000005C88: 4A0A0610
	v_cmp_gt_u32_e32 vcc_lo, 16, v0                            // 000000005C8C: 7C980090
	v_cmp_gt_u32_e64 s0, 8, v3                                 // 000000005C90: D44C0000 02020688
	s_delay_alu instid0(VALU_DEP_4)                            // 000000005C98: BF870004
	v_cmp_gt_i32_e64 s1, s4, v4                                // 000000005C9C: D4440001 02020804
	s_wait_kmcnt 0x0                                           // 000000005CA4: BFC70000
	v_cmp_gt_i32_e64 s2, s5, v5                                // 000000005CA8: D4440002 02020A05
	s_and_b32 s0, vcc_lo, s0                                   // 000000005CB0: 8B00006A
	s_wait_alu depctr_sa_sdst(0)                               // 000000005CB4: BF88FF9E
	s_and_b32 s0, s0, s1                                       // 000000005CB8: 8B000100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005CBC: BF88FF9E
	s_and_b32 s0, s0, s2                                       // 000000005CC0: 8B000200
	s_barrier_wait 0xffff                                      // 000000005CC4: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000005CC8: EE0AC07C 00040000 00000000
	s_wait_alu depctr_sa_sdst(0)                               // 000000005CD4: BF88FF9E
	s_and_saveexec_b32 s1, s0                                  // 000000005CD8: BE812000
	s_cbranch_execz L82                                        // 000000005CDC: BFA5010F
	s_cmp_lt_i32 s7, 1                                         // 000000005CE0: BF048107
	s_cbranch_scc1 L65                                         // 000000005CE4: BFA20036
	v_mul_u32_u24_e32 v0, 12, v0                               // 000000005CE8: 1600008C
	v_dual_mov_b32 v8, 0 :: v_dual_mov_b32 v7, 0               // 000000005CEC: CA100080 08060080
	v_mov_b32_e32 v6, 0                                        // 000000005CF4: 7E0C0280
	s_cmp_gt_i32 s6, 0                                         // 000000005CF8: BF028006
	s_mov_b32 s0, 0                                            // 000000005CFC: BE800080
	s_cselect_b32 s1, -1, 0                                    // 000000005D00: 980180C1
	s_branch L63                                               // 000000005D04: BFA00004

0000000000005d08 <L62>:
	s_add_co_i32 s0, s0, 1                                     // 000000005D08: 81008100
	s_wait_alu depctr_sa_sdst(0)                               // 000000005D0C: BF88FF9E
	s_cmp_eq_u32 s0, s7                                        // 000000005D10: BF060700
	s_cbranch_scc1 L66                                         // 000000005D14: BFA2002D

0000000000005d18 <L63>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000005D18: BF88FF9E
	s_and_b32 s2, s1, exec_lo                                  // 000000005D1C: 8B027E01
	s_cselect_b32 s2, 1, 0                                     // 000000005D20: 98028081
	s_wait_alu depctr_sa_sdst(0)                               // 000000005D24: BF88FF9E
	s_cmp_lg_u32 s2, 1                                         // 000000005D28: BF078102
	s_cbranch_scc1 L62                                         // 000000005D2C: BFA2FFF6
	v_add_nc_u32_e32 v1, s0, v3                                // 000000005D30: 4A020600
	s_mul_i32 s2, s0, s6                                       // 000000005D34: 96020600
	s_mov_b32 s4, 0                                            // 000000005D38: BE840080
	s_wait_alu depctr_sa_sdst(0)                               // 000000005D3C: BF88FF9E
	s_mul_i32 s2, s2, 12                                       // 000000005D40: 96028C02
	s_wait_alu depctr_sa_sdst(0)                               // 000000005D44: BF88FF9E
	s_addk_co_i32 s2, 0xe70                                    // 000000005D48: B7820E70
	v_mul_lo_u32 v1, v1, s14                                   // 000000005D4C: D72C0001 02001D01
	s_delay_alu instid0(VALU_DEP_1)                            // 000000005D54: BF870001
	v_mad_co_u64_u32 v[1:2], null, v1, 12, v[0:1]              // 000000005D58: D6FE7C01 04011901

0000000000005d60 <L64>:
	s_mul_i32 s5, s4, 12                                       // 000000005D60: 96058C04
	s_add_co_i32 s4, s4, 1                                     // 000000005D64: 81048104
	s_wait_alu depctr_sa_sdst(0)                               // 000000005D68: BF88FF9E
	s_delay_alu instid0(VALU_DEP_1)                            // 000000005D6C: BF870001
	v_add_nc_u32_e32 v2, s5, v1                                // 000000005D70: 4A040205
	s_add_co_i32 s5, s2, s5                                    // 000000005D74: 81050502
	s_cmp_eq_u32 s4, s6                                        // 000000005D78: BF060604
	s_wait_alu depctr_sa_sdst(0)                               // 000000005D7C: BF88FF9E
	v_mov_b32_e32 v13, s5                                      // 000000005D80: 7E1A0205
	ds_load_2addr_b32 v[9:10], v2 offset1:1                    // 000000005D84: D8DC0100 09000002
	ds_load_b32 v2, v2 offset:8                                // 000000005D8C: D8D80008 02000002
	ds_load_2addr_b32 v[11:12], v13 offset1:1                  // 000000005D94: D8DC0100 0B00000D
	ds_load_b32 v13, v13 offset:8                              // 000000005D9C: D8D80008 0D00000D
	s_wait_dscnt 0x1                                           // 000000005DA4: BFC60001
	v_dual_fmac_f32 v8, v9, v11 :: v_dual_fmac_f32 v7, v10, v12// 000000005DA8: C8001709 0806190A
	s_wait_dscnt 0x0                                           // 000000005DB0: BFC60000
	v_fmac_f32_e32 v6, v2, v13                                 // 000000005DB4: 560C1B02
	s_cbranch_scc0 L64                                         // 000000005DB8: BFA1FFE9
	s_branch L62                                               // 000000005DBC: BFA0FFD2

0000000000005dc0 <L65>:
	v_dual_mov_b32 v8, 0 :: v_dual_mov_b32 v7, 0               // 000000005DC0: CA100080 08060080
	v_mov_b32_e32 v6, 0                                        // 000000005DC8: 7E0C0280

0000000000005dcc <L66>:
	v_mov_b16_e32 v0.l, 0xff                                   // 000000005DCC: 7E0038FF 000000FF
	s_mov_b32 s2, exec_lo                                      // 000000005DD4: BE82007E
	s_delay_alu instid0(VALU_DEP_3)                            // 000000005DD8: BF870003
	v_cmpx_nlt_f32_e32 0x437f0000, v8                          // 000000005DDC: 7D3C10FF 437F0000
	s_cbranch_execz L71                                        // 000000005DE4: BFA50036
	v_mov_b16_e32 v0.l, 0                                      // 000000005DE8: 7E003880
	s_mov_b32 s4, exec_lo                                      // 000000005DEC: BE84007E
	v_cmpx_ngt_f32_e32 0, v8                                   // 000000005DF0: 7D361080
	s_cbranch_execz L70                                        // 000000005DF4: BFA50030
	s_mov_b32 s5, exec_lo                                      // 000000005DF8: BE85007E
	v_cmpx_o_f32_e32 v8, v8                                    // 000000005DFC: 7D2E1108
	s_cbranch_execz L69                                        // 000000005E00: BFA5002A
	v_add_f32_e32 v0, v8, v8                                   // 000000005E04: 06001108
	v_cmp_nlg_f32_e32 vcc_lo, 0, v8                            // 000000005E08: 7C341080
	v_cmp_neq_f32_e64 s1, 0, v8                                // 000000005E0C: D41D0001 02021080
	s_delay_alu instid0(VALU_DEP_3)                            // 000000005E14: BF870003
	v_cmp_neq_f32_e64 s0, v0, v8                               // 000000005E18: D41D0000 02021100
	s_or_b32 s0, vcc_lo, s0                                    // 000000005E20: 8C00006A
	s_wait_alu depctr_sa_sdst(0)                               // 000000005E24: BF88FF9E
	s_and_b32 s1, s1, s0                                       // 000000005E28: 8B010001
	s_wait_alu depctr_sa_sdst(0)                               // 000000005E2C: BF88FF9E
	s_and_saveexec_b32 s0, s1                                  // 000000005E30: BE802001
	s_cbranch_execz L68                                        // 000000005E34: BFA5001B
	v_cvt_i32_f32_e32 v1, v8                                   // 000000005E38: 7E021108
	s_mov_b32 s1, exec_lo                                      // 000000005E3C: BE81007E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000005E40: BF870091
	v_cvt_f32_i32_e32 v0, v1                                   // 000000005E44: 7E000B01
	v_sub_f32_e32 v2, v8, v0                                   // 000000005E48: 08040108
	s_delay_alu instid0(VALU_DEP_1)                            // 000000005E4C: BF870001
	v_cmpx_ngt_f32_e32 0.5, v2                                 // 000000005E50: 7D3604F0
	s_cbranch_execz L67                                        // 000000005E54: BFA50010
	s_mov_b32 s6, exec_lo                                      // 000000005E58: BE86007E
	v_cmpx_nlt_f32_e32 0.5, v2                                 // 000000005E5C: 7D3C04F0
	s_wait_alu depctr_sa_sdst(0)                               // 000000005E60: BF88FF9E
	s_xor_b32 s6, exec_lo, s6                                  // 000000005E64: 8D06067E
	v_dual_add_f32 v2, 1.0, v0 :: v_dual_and_b32 v1, 1, v1     // 000000005E68: C92400F2 02000281
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000005E70: BF870121
	v_cmp_eq_u32_e32 vcc_lo, 0, v1                             // 000000005E74: 7C940280
	s_wait_alu depctr_va_vcc(0)                                // 000000005E78: BF88FF9D
	v_cndmask_b32_e32 v0, v2, v0, vcc_lo                       // 000000005E7C: 02000102
	s_wait_alu depctr_sa_sdst(0)                               // 000000005E80: BF88FF9E
	s_and_not1_saveexec_b32 s6, s6                             // 000000005E84: BE863006
	s_delay_alu instid0(VALU_DEP_1)                            // 000000005E88: BF870001
	v_add_f32_e32 v0, 1.0, v0                                  // 000000005E8C: 060000F2
	s_wait_alu depctr_sa_sdst(0)                               // 000000005E90: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s6                              // 000000005E94: 8C7E067E

0000000000005e98 <L67>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000005E98: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005E9C: 8C7E017E
	v_mov_b32_e32 v8, v0                                       // 000000005EA0: 7E100300

0000000000005ea4 <L68>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000005EA4: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005EA8: 8C7E007E

0000000000005eac <L69>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000005EAC: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s5                              // 000000005EB0: 8C7E057E
	v_cvt_i32_f32_e32 v0, v8                                   // 000000005EB4: 7E001108

0000000000005eb8 <L70>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000005EB8: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s4                              // 000000005EBC: 8C7E047E

0000000000005ec0 <L71>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000005EC0: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s2                              // 000000005EC4: 8C7E027E
	v_mov_b32_e32 v1, 0xff00                                   // 000000005EC8: 7E0202FF 0000FF00
	s_mov_b32 s2, exec_lo                                      // 000000005ED0: BE82007E
	v_cmpx_nlt_f32_e32 0x437f0000, v7                          // 000000005ED4: 7D3C0EFF 437F0000
	s_cbranch_execz L76                                        // 000000005EDC: BFA50039
	v_mov_b32_e32 v1, 0                                        // 000000005EE0: 7E020280
	s_mov_b32 s4, exec_lo                                      // 000000005EE4: BE84007E
	v_cmpx_ngt_f32_e32 0, v7                                   // 000000005EE8: 7D360E80
	s_cbranch_execz L75                                        // 000000005EEC: BFA50033
	s_mov_b32 s5, exec_lo                                      // 000000005EF0: BE85007E
	v_cmpx_o_f32_e32 v7, v7                                    // 000000005EF4: 7D2E0F07
	s_cbranch_execz L74                                        // 000000005EF8: BFA5002A
	v_add_f32_e32 v1, v7, v7                                   // 000000005EFC: 06020F07
	v_cmp_nlg_f32_e32 vcc_lo, 0, v7                            // 000000005F00: 7C340E80
	v_cmp_neq_f32_e64 s1, 0, v7                                // 000000005F04: D41D0001 02020E80
	s_delay_alu instid0(VALU_DEP_3)                            // 000000005F0C: BF870003
	v_cmp_neq_f32_e64 s0, v1, v7                               // 000000005F10: D41D0000 02020F01
	s_or_b32 s0, vcc_lo, s0                                    // 000000005F18: 8C00006A
	s_wait_alu depctr_sa_sdst(0)                               // 000000005F1C: BF88FF9E
	s_and_b32 s1, s1, s0                                       // 000000005F20: 8B010001
	s_wait_alu depctr_sa_sdst(0)                               // 000000005F24: BF88FF9E
	s_and_saveexec_b32 s0, s1                                  // 000000005F28: BE802001
	s_cbranch_execz L73                                        // 000000005F2C: BFA5001B
	v_cvt_i32_f32_e32 v2, v7                                   // 000000005F30: 7E041107
	s_mov_b32 s1, exec_lo                                      // 000000005F34: BE81007E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000005F38: BF870091
	v_cvt_f32_i32_e32 v1, v2                                   // 000000005F3C: 7E020B02
	v_sub_f32_e32 v3, v7, v1                                   // 000000005F40: 08060307
	s_delay_alu instid0(VALU_DEP_1)                            // 000000005F44: BF870001
	v_cmpx_ngt_f32_e32 0.5, v3                                 // 000000005F48: 7D3606F0
	s_cbranch_execz L72                                        // 000000005F4C: BFA50010
	s_mov_b32 s6, exec_lo                                      // 000000005F50: BE86007E
	v_cmpx_nlt_f32_e32 0.5, v3                                 // 000000005F54: 7D3C06F0
	s_wait_alu depctr_sa_sdst(0)                               // 000000005F58: BF88FF9E
	s_xor_b32 s6, exec_lo, s6                                  // 000000005F5C: 8D06067E
	v_dual_add_f32 v3, 1.0, v1 :: v_dual_and_b32 v2, 1, v2     // 000000005F60: C92402F2 03020481
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000005F68: BF870121
	v_cmp_eq_u32_e32 vcc_lo, 0, v2                             // 000000005F6C: 7C940480
	s_wait_alu depctr_va_vcc(0)                                // 000000005F70: BF88FF9D
	v_cndmask_b32_e32 v1, v3, v1, vcc_lo                       // 000000005F74: 02020303
	s_wait_alu depctr_sa_sdst(0)                               // 000000005F78: BF88FF9E
	s_and_not1_saveexec_b32 s6, s6                             // 000000005F7C: BE863006
	s_delay_alu instid0(VALU_DEP_1)                            // 000000005F80: BF870001
	v_add_f32_e32 v1, 1.0, v1                                  // 000000005F84: 060202F2
	s_wait_alu depctr_sa_sdst(0)                               // 000000005F88: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s6                              // 000000005F8C: 8C7E067E

0000000000005f90 <L72>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000005F90: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005F94: 8C7E017E
	v_mov_b32_e32 v7, v1                                       // 000000005F98: 7E0E0301

0000000000005f9c <L73>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000005F9C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s0                              // 000000005FA0: 8C7E007E

0000000000005fa4 <L74>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000005FA4: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s5                              // 000000005FA8: 8C7E057E
	v_cvt_i32_f32_e32 v1, v7                                   // 000000005FAC: 7E021107
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000005FB0: BF870091
	v_cvt_u32_u16_e32 v1, v1.l                                 // 000000005FB4: 7E02D701
	v_lshlrev_b32_e32 v1, 8, v1                                // 000000005FB8: 30020288

0000000000005fbc <L75>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000005FBC: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s4                              // 000000005FC0: 8C7E047E

0000000000005fc4 <L76>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000005FC4: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s2                              // 000000005FC8: 8C7E027E
	v_mov_b32_e32 v2, 0xff0000                                 // 000000005FCC: 7E0402FF 00FF0000
	s_mov_b32 s2, exec_lo                                      // 000000005FD4: BE82007E
	v_cmpx_nlt_f32_e32 0x437f0000, v6                          // 000000005FD8: 7D3C0CFF 437F0000
	s_cbranch_execz L81                                        // 000000005FE0: BFA50038
	v_mov_b32_e32 v2, 0                                        // 000000005FE4: 7E040280
	s_mov_b32 s4, exec_lo                                      // 000000005FE8: BE84007E
	v_cmpx_ngt_f32_e32 0, v6                                   // 000000005FEC: 7D360C80
	s_cbranch_execz L80                                        // 000000005FF0: BFA50032
	s_mov_b32 s5, exec_lo                                      // 000000005FF4: BE85007E
	v_cmpx_o_f32_e32 v6, v6                                    // 000000005FF8: 7D2E0D06
	s_cbranch_execz L79                                        // 000000005FFC: BFA5002A
	v_add_f32_e32 v2, v6, v6                                   // 000000006000: 06040D06
	v_cmp_nlg_f32_e32 vcc_lo, 0, v6                            // 000000006004: 7C340C80
	v_cmp_neq_f32_e64 s1, 0, v6                                // 000000006008: D41D0001 02020C80
	s_delay_alu instid0(VALU_DEP_3)                            // 000000006010: BF870003
	v_cmp_neq_f32_e64 s0, v2, v6                               // 000000006014: D41D0000 02020D02
	s_or_b32 s0, vcc_lo, s0                                    // 00000000601C: 8C00006A
	s_wait_alu depctr_sa_sdst(0)                               // 000000006020: BF88FF9E
	s_and_b32 s1, s1, s0                                       // 000000006024: 8B010001
	s_wait_alu depctr_sa_sdst(0)                               // 000000006028: BF88FF9E
	s_and_saveexec_b32 s0, s1                                  // 00000000602C: BE802001
	s_cbranch_execz L78                                        // 000000006030: BFA5001B
	v_cvt_i32_f32_e32 v3, v6                                   // 000000006034: 7E061106
	s_mov_b32 s1, exec_lo                                      // 000000006038: BE81007E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000603C: BF870091
	v_cvt_f32_i32_e32 v2, v3                                   // 000000006040: 7E040B03
	v_sub_f32_e32 v6, v6, v2                                   // 000000006044: 080C0506
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006048: BF870001
	v_cmpx_ngt_f32_e32 0.5, v6                                 // 00000000604C: 7D360CF0
	s_cbranch_execz L77                                        // 000000006050: BFA50010
	s_mov_b32 s6, exec_lo                                      // 000000006054: BE86007E
	v_cmpx_nlt_f32_e32 0.5, v6                                 // 000000006058: 7D3C0CF0
	s_wait_alu depctr_sa_sdst(0)                               // 00000000605C: BF88FF9E
	s_xor_b32 s6, exec_lo, s6                                  // 000000006060: 8D06067E
	v_dual_add_f32 v6, 1.0, v2 :: v_dual_and_b32 v3, 1, v3     // 000000006064: C92404F2 06020681
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 00000000606C: BF870121
	v_cmp_eq_u32_e32 vcc_lo, 0, v3                             // 000000006070: 7C940680
	s_wait_alu depctr_va_vcc(0)                                // 000000006074: BF88FF9D
	v_cndmask_b32_e32 v2, v6, v2, vcc_lo                       // 000000006078: 02040506
	s_wait_alu depctr_sa_sdst(0)                               // 00000000607C: BF88FF9E
	s_and_not1_saveexec_b32 s6, s6                             // 000000006080: BE863006
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006084: BF870001
	v_add_f32_e32 v2, 1.0, v2                                  // 000000006088: 060404F2
	s_wait_alu depctr_sa_sdst(0)                               // 00000000608C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s6                              // 000000006090: 8C7E067E

0000000000006094 <L77>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000006094: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s1                              // 000000006098: 8C7E017E
	v_mov_b32_e32 v6, v2                                       // 00000000609C: 7E0C0302

00000000000060a0 <L78>:
	s_wait_alu depctr_sa_sdst(0)                               // 0000000060A0: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s0                              // 0000000060A4: 8C7E007E

00000000000060a8 <L79>:
	s_wait_alu depctr_sa_sdst(0)                               // 0000000060A8: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s5                              // 0000000060AC: 8C7E057E
	v_cvt_i32_f32_e32 v2, v6                                   // 0000000060B0: 7E041106
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000060B4: BF870001
	v_lshlrev_b32_e32 v2, 16, v2                               // 0000000060B8: 30040490

00000000000060bc <L80>:
	s_wait_alu depctr_sa_sdst(0)                               // 0000000060BC: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s4                              // 0000000060C0: 8C7E047E

00000000000060c4 <L81>:
	s_wait_alu depctr_sa_sdst(0)                               // 0000000060C4: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s2                              // 0000000060C8: 8C7E027E
	v_mul_lo_u32 v3, s3, v5                                    // 0000000060CC: D72C0003 02020A03
	v_and_or_b32 v0, 0xff00, v1, v0                            // 0000000060D4: D6570000 040202FF 0000FF00
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 0000000060E0: BF8700B2
	v_add_co_u32 v5, s0, s8, v3                                // 0000000060E4: D7000005 02020608
	s_wait_alu depctr_va_sdst(0)                               // 0000000060EC: BF88F19F
	v_add_co_ci_u32_e64 v6, null, s9, 0, s0                    // 0000000060F0: D5207C06 00010009
	v_mad_co_i64_i32 v[3:4], null, v4, 3, v[5:6]               // 0000000060F8: D6FF7C03 04150704
	s_clause 0x1                                               // 000000006100: BF850001
	global_store_d16_hi_b8 v[3:4], v2, off offset:2            // 000000006104: EE09007C 01000000 00000203
	global_store_b16 v[3:4], v0, off                           // 000000006110: EE06407C 00000000 00000003

000000000000611c <L82>:
	s_endpgm                                                   // 00000000611C: BFB00000
		...

0000000000006200 <_ZN2fk21linearFilterDPPKernelILNS_7ParArchE3ENS_22LinearFilterDPPDetailsINS_6Float3ELi16ELi8ELi7ELi7EEENS_5TupleIJNS_25ReadInstantiableOperationINS_15FusedOperation_IvJNS6_INS_13PerThreadReadILNS_2NDE2ENS_6Uchar3EEEEENS_26UnaryInstantiableOperationINS_4CastISA_S3_EEEEEEEEENS6_INS_18ConstantFilterReadIS3_EEEEEEENSD_INS_3MulIS3_S3_S3_NS_9UnaryTypeEEEEENSD_INS_3AddIS3_S3_S3_SO_EEEENS_26WriteInstantiableOperationINS7_IvJNSD_INS_12SaturateCastIS3_SA_EEEENSU_INS_14PerThreadWriteILS9_2ESA_EEEEEEEEEEEvT0_T1_T2_T3_T4_>:
	s_clause 0x2                                               // 000000006200: BF850002
	s_load_b32 s2, s[0:1], 0x6c                                // 000000006204: F4000080 F800006C
	s_load_b128 s[4:7], s[0:1], 0x0                            // 00000000620C: F4004100 F8000000
	s_load_b64 s[12:13], s[0:1], 0x48                          // 000000006214: F4002300 F8000048
	v_bfe_u32 v6, v0, 10, 10                                   // 00000000621C: D6100006 02291500
	v_and_b32_e32 v0, 0x3ff, v0                                // 000000006224: 360000FF 000003FF
	s_lshl_b32 s14, ttmp9, 4                                   // 00000000622C: 840E8475
	s_lshl_b32 s15, ttmp7, 3                                   // 000000006230: 840F8373
	s_mov_b32 s9, exec_lo                                      // 000000006234: BE89007E
	s_wait_kmcnt 0x0                                           // 000000006238: BFC70000
	s_and_b32 s3, s2, 0xffff                                   // 00000000623C: 8B03FF02 0000FFFF
	s_add_co_i32 s11, s6, 15                                   // 000000006244: 810B8F06
	v_mad_u32_u24 v1, v6, s3, v0                               // 000000006248: D60B0001 04000706
	s_add_co_i32 s8, s7, 7                                     // 000000006250: 81088707
	s_lshr_b32 s16, s2, 16                                     // 000000006254: 85109002
	s_mul_i32 s8, s8, s11                                      // 000000006258: 96080B08
	s_mul_i32 s16, s16, s3                                     // 00000000625C: 96100310
	v_cmpx_gt_i32_e32 s8, v1                                   // 000000006260: 7D880208
	s_cbranch_execz L84                                        // 000000006264: BFA5006D
	s_abs_i32 s10, s11                                         // 000000006268: BE8A150B
	s_load_b64 s[24:25], s[0:1], 0x10                          // 00000000626C: F4002600 F8000010
	s_cvt_f32_u32 s2, s10                                      // 000000006274: BE82650A
	s_add_nc_u64 s[18:19], s[0:1], 24                          // 000000006278: A9929800
	v_mul_lo_u32 v2, v1, 12                                    // 00000000627C: D72C0002 02011901
	v_mov_b32_e32 v3, v1                                       // 000000006284: 7E060301
	s_wait_alu depctr_sa_sdst(0)                               // 000000006288: BF88FF9E
	v_s_rcp_f32 s20, s2                                        // 00000000628C: D6840014 02010002
	s_clause 0x1                                               // 000000006294: BF850001
	s_load_b64 s[2:3], s[18:19], 0x0                           // 000000006298: F4002089 F8000000
	s_load_b32 s17, s[18:19], 0x10                             // 0000000062A0: F4000449 F8000010
	s_add_co_i32 s18, s4, -1                                   // 0000000062A8: 8112C104
	s_add_co_i32 s19, s5, -1                                   // 0000000062AC: 8113C105
	s_mov_b32 s21, 0                                           // 0000000062B0: BE950080
	s_sub_co_i32 s22, 0, s11                                   // 0000000062B4: 81960B80
	s_mul_f32 s23, s20, 0x4f7ffffe                             // 0000000062B8: A217FF14 4F7FFFFE
	s_ashr_i32 s20, s11, 31                                    // 0000000062C0: 86149F0B
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 0000000062C4: BF87052A
	s_cvt_u32_f32 s26, s23                                     // 0000000062C8: BE9A6717
	s_sub_co_i32 s23, 0, s10                                   // 0000000062CC: 81970A80
	s_mul_i32 s27, s23, s26                                    // 0000000062D0: 961B1A17
	s_wait_kmcnt 0x0                                           // 0000000062D4: BFC70000
	s_sub_co_i32 s23, s14, s24                                 // 0000000062D8: 8197180E
	s_mul_hi_u32 s27, s26, s27                                 // 0000000062DC: 969B1B1A
	s_sub_co_i32 s24, s15, s25                                 // 0000000062E0: 8198190F
	s_add_co_i32 s25, s26, s27                                 // 0000000062E4: 81191B1A
	s_mul_i32 s26, s16, 12                                     // 0000000062E8: 961A8C10

00000000000062ec <L83>:
	v_sub_nc_u32_e32 v4, 0, v3                                 // 0000000062EC: 4C080680
	v_ashrrev_i32_e32 v5, 31, v3                               // 0000000062F0: 340A069F
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000062F4: BF870112
	v_max_i32_e32 v4, v4, v3                                   // 0000000062F8: 24080704
	v_xor_b32_e32 v5, s20, v5                                  // 0000000062FC: 3A0A0A14
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000006300: BF870092
	v_mul_hi_u32 v7, v4, s25                                   // 000000006304: D72D0007 02003304
	v_mul_lo_u32 v8, v7, s10                                   // 00000000630C: D72C0008 02001507
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000006314: BF870091
	v_sub_nc_u32_e32 v4, v4, v8                                // 000000006318: 4C081104
	v_subrev_nc_u32_e32 v8, s10, v4                            // 00000000631C: 4E10080A
	v_cmp_le_u32_e32 vcc_lo, s10, v4                           // 000000006320: 7C96080A
	s_wait_alu depctr_va_vcc(0)                                // 000000006324: BF88FF9D
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000006328: BF870092
	v_dual_cndmask_b32 v4, v4, v8 :: v_dual_add_nc_u32 v9, 1, v7// 00000000632C: CA601104 04080E81
	v_cndmask_b32_e32 v7, v7, v9, vcc_lo                       // 000000006334: 020E1307
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000006338: BF870112
	v_cmp_le_u32_e32 vcc_lo, s10, v4                           // 00000000633C: 7C96080A
	v_add_nc_u32_e32 v8, 1, v7                                 // 000000006340: 4A100E81
	s_wait_alu depctr_va_vcc(0)                                // 000000006344: BF88FF9D
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000006348: BF870091
	v_cndmask_b32_e32 v4, v7, v8, vcc_lo                       // 00000000634C: 02081107
	v_xor_b32_e32 v4, v4, v5                                   // 000000006350: 3A080B04
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000006354: BF870091
	v_sub_nc_u32_e32 v4, v4, v5                                // 000000006358: 4C080B04
	v_mul_lo_u32 v5, s22, v4                                   // 00000000635C: D72C0005 02020816
	v_add_nc_u32_e32 v4, s24, v4                               // 000000006364: 4A080818
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 000000006368: BF870221
	v_min_i32_e32 v7, s19, v4                                  // 00000000636C: 220E0813
	v_cmp_lt_i32_e32 vcc_lo, -1, v4                            // 000000006370: 7C8208C1
	v_add3_u32 v4, s23, v3, v5                                 // 000000006374: D6550004 04160617
	v_add_nc_u32_e32 v3, s16, v3                               // 00000000637C: 4A060610
	s_wait_alu depctr_va_vcc(0)                                // 000000006380: BF88FF9D
	v_cndmask_b32_e32 v5, 0, v7, vcc_lo                        // 000000006384: 020A0E80
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)// 000000006388: BF8701A3
	v_min_i32_e32 v7, s18, v4                                  // 00000000638C: 220E0812
	v_cmp_lt_i32_e32 vcc_lo, -1, v4                            // 000000006390: 7C8208C1
	v_mul_lo_u32 v4, v5, s17                                   // 000000006394: D72C0004 02002305
	s_wait_alu depctr_va_vcc(0)                                // 00000000639C: BF88FF9D
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)// 0000000063A0: BF8701A3
	v_cndmask_b32_e32 v7, 0, v7, vcc_lo                        // 0000000063A4: 020E0E80
	v_cmp_le_i32_e32 vcc_lo, s8, v3                            // 0000000063A8: 7C860608
	v_add_co_u32 v4, s27, s2, v4                               // 0000000063AC: D7001B04 02020802
	s_wait_alu depctr_va_sdst(0)                               // 0000000063B4: BF88F19F
	v_add_co_ci_u32_e64 v5, null, s3, 0, s27                   // 0000000063B8: D5207C05 006D0003
	s_or_b32 s21, vcc_lo, s21                                  // 0000000063C0: 8C15156A
	v_mad_co_i64_i32 v[4:5], null, v7, 3, v[4:5]               // 0000000063C4: D6FF7C04 04110707
	s_clause 0x1                                               // 0000000063CC: BF850001
	global_load_u16 v7, v[4:5], off                            // 0000000063D0: EE04807C 00000007 00000004
	global_load_u8 v4, v[4:5], off offset:2                    // 0000000063DC: EE04007C 00000004 00000204
	s_wait_loadcnt 0x1                                         // 0000000063E8: BFC00001
	v_cvt_f32_ubyte0_e32 v5, v7                                // 0000000063EC: 7E0A2307
	v_cvt_f32_ubyte1_e32 v7, v7                                // 0000000063F0: 7E0E2507
	s_wait_loadcnt 0x0                                         // 0000000063F4: BFC00000
	v_cvt_f32_ubyte0_e32 v4, v4                                // 0000000063F8: 7E082304
	ds_store_2addr_b32 v2, v5, v7 offset1:1                    // 0000000063FC: D8380100 00070502
	ds_store_b32 v2, v4 offset:8                               // 000000006404: D8340008 00000402
	v_add_nc_u32_e32 v2, s26, v2                               // 00000000640C: 4A04041A
	s_wait_alu depctr_sa_sdst(0)                               // 000000006410: BF88FF9E
	s_and_not1_b32 exec_lo, exec_lo, s21                       // 000000006414: 917E157E
	s_cbranch_execnz L83                                       // 000000006418: BFA6FFB4

000000000000641c <L84>:
	s_or_b32 exec_lo, exec_lo, s9                              // 00000000641C: 8C7E097E
	s_clause 0x1                                               // 000000006420: BF850001
	s_load_b96 s[8:10], s[0:1], 0x30                           // 000000006424: F400A200 F8000030
	s_load_b32 s3, s[0:1], 0x58                                // 00000000642C: F40000C0 F8000058
	s_mul_i32 s0, s7, s6                                       // 000000006434: 96000607
	s_mov_b32 s1, exec_lo                                      // 000000006438: BE81007E
	v_cmpx_gt_i32_e32 s0, v1                                   // 00000000643C: 7D880200
	s_cbranch_execz L86                                        // 000000006440: BFA50015
	s_mov_b32 s2, 0                                            // 000000006444: BE820080

0000000000006448 <L85>:
	v_mul_lo_u32 v2, v1, 12                                    // 000000006448: D72C0002 02011901
	s_wait_kmcnt 0x0                                           // 000000006450: BFC70000
	v_dual_mov_b32 v4, s9 :: v_dual_add_nc_u32 v1, s16, v1     // 000000006454: CA200009 04000210
	v_mov_b32_e32 v3, s8                                       // 00000000645C: 7E060208
	v_mov_b32_e32 v5, s10                                      // 000000006460: 7E0A020A
	s_delay_alu instid0(VALU_DEP_3)                            // 000000006464: BF870003
	v_cmp_le_i32_e32 vcc_lo, s0, v1                            // 000000006468: 7C860200
	v_add_nc_u32_e32 v7, 0xe70, v2                             // 00000000646C: 4A0E04FF 00000E70
	ds_store_2addr_b32 v7, v3, v4 offset1:1                    // 000000006474: D8380100 00040307
	ds_store_b32 v2, v5 offset:3704                            // 00000000647C: D8340E78 00000502
	s_wait_alu depctr_sa_sdst(0)                               // 000000006484: BF88FF9E
	s_or_b32 s2, vcc_lo, s2                                    // 000000006488: 8C02026A
	s_wait_alu depctr_sa_sdst(0)                               // 00000000648C: BF88FF9E
	s_and_not1_b32 exec_lo, exec_lo, s2                        // 000000006490: 917E027E
	s_cbranch_execnz L85                                       // 000000006494: BFA6FFEC

0000000000006498 <L86>:
	s_or_b32 exec_lo, exec_lo, s1                              // 000000006498: 8C7E017E
	s_wait_dscnt 0x0                                           // 00000000649C: BFC60000
	s_barrier_signal -1                                        // 0000000064A0: BE804EC1
	v_add_nc_u32_e32 v3, s14, v0                               // 0000000064A4: 4A06000E
	v_add_nc_u32_e32 v4, s15, v6                               // 0000000064A8: 4A080C0F
	v_cmp_gt_u32_e32 vcc_lo, 16, v0                            // 0000000064AC: 7C980090
	v_cmp_gt_u32_e64 s0, 8, v6                                 // 0000000064B0: D44C0000 02020C88
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 0000000064B8: BF870214
	v_cmp_gt_i32_e64 s1, s4, v3                                // 0000000064BC: D4440001 02020604
	v_cmp_gt_i32_e64 s2, s5, v4                                // 0000000064C4: D4440002 02020805
	s_and_b32 s0, vcc_lo, s0                                   // 0000000064CC: 8B00006A
	s_wait_alu depctr_sa_sdst(0)                               // 0000000064D0: BF88FF9E
	s_and_b32 s0, s0, s1                                       // 0000000064D4: 8B000100
	s_wait_alu depctr_sa_sdst(0)                               // 0000000064D8: BF88FF9E
	s_and_b32 s0, s0, s2                                       // 0000000064DC: 8B000200
	s_barrier_wait 0xffff                                      // 0000000064E0: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 0000000064E4: EE0AC07C 00040000 00000000
	s_wait_alu depctr_sa_sdst(0)                               // 0000000064F0: BF88FF9E
	s_and_saveexec_b32 s1, s0                                  // 0000000064F4: BE812000
	s_cbranch_execz L107                                       // 0000000064F8: BFA5010A
	s_cmp_lt_i32 s7, 1                                         // 0000000064FC: BF048107
	s_cbranch_scc1 L90                                         // 000000006500: BFA20036
	v_mul_u32_u24_e32 v0, 12, v0                               // 000000006504: 1600008C
	v_dual_mov_b32 v8, 0 :: v_dual_mov_b32 v7, 0               // 000000006508: CA100080 08060080
	v_mov_b32_e32 v5, 0                                        // 000000006510: 7E0A0280
	s_cmp_gt_i32 s6, 0                                         // 000000006514: BF028006
	s_mov_b32 s0, 0                                            // 000000006518: BE800080
	s_cselect_b32 s1, -1, 0                                    // 00000000651C: 980180C1
	s_branch L88                                               // 000000006520: BFA00004

0000000000006524 <L87>:
	s_add_co_i32 s0, s0, 1                                     // 000000006524: 81008100
	s_wait_alu depctr_sa_sdst(0)                               // 000000006528: BF88FF9E
	s_cmp_eq_u32 s0, s7                                        // 00000000652C: BF060700
	s_cbranch_scc1 L91                                         // 000000006530: BFA2002D

0000000000006534 <L88>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000006534: BF88FF9E
	s_and_b32 s2, s1, exec_lo                                  // 000000006538: 8B027E01
	s_cselect_b32 s2, 1, 0                                     // 00000000653C: 98028081
	s_wait_alu depctr_sa_sdst(0)                               // 000000006540: BF88FF9E
	s_cmp_lg_u32 s2, 1                                         // 000000006544: BF078102
	s_cbranch_scc1 L87                                         // 000000006548: BFA2FFF6
	v_add_nc_u32_e32 v1, s0, v6                                // 00000000654C: 4A020C00
	s_mul_i32 s2, s0, s6                                       // 000000006550: 96020600
	s_mov_b32 s4, 0                                            // 000000006554: BE840080
	s_wait_alu depctr_sa_sdst(0)                               // 000000006558: BF88FF9E
	s_mul_i32 s2, s2, 12                                       // 00000000655C: 96028C02
	s_wait_alu depctr_sa_sdst(0)                               // 000000006560: BF88FF9E
	s_addk_co_i32 s2, 0xe70                                    // 000000006564: B7820E70
	v_mul_lo_u32 v1, v1, s11                                   // 000000006568: D72C0001 02001701
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006570: BF870001
	v_mad_co_u64_u32 v[1:2], null, v1, 12, v[0:1]              // 000000006574: D6FE7C01 04011901

000000000000657c <L89>:
	s_mul_i32 s5, s4, 12                                       // 00000000657C: 96058C04
	s_add_co_i32 s4, s4, 1                                     // 000000006580: 81048104
	s_wait_alu depctr_sa_sdst(0)                               // 000000006584: BF88FF9E
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006588: BF870001
	v_add_nc_u32_e32 v2, s5, v1                                // 00000000658C: 4A040205
	s_add_co_i32 s5, s2, s5                                    // 000000006590: 81050502
	s_cmp_eq_u32 s4, s6                                        // 000000006594: BF060604
	s_wait_alu depctr_sa_sdst(0)                               // 000000006598: BF88FF9E
	v_mov_b32_e32 v13, s5                                      // 00000000659C: 7E1A0205
	ds_load_2addr_b32 v[9:10], v2 offset1:1                    // 0000000065A0: D8DC0100 09000002
	ds_load_b32 v2, v2 offset:8                                // 0000000065A8: D8D80008 02000002
	ds_load_2addr_b32 v[11:12], v13 offset1:1                  // 0000000065B0: D8DC0100 0B00000D
	ds_load_b32 v13, v13 offset:8                              // 0000000065B8: D8D80008 0D00000D
	s_wait_dscnt 0x1                                           // 0000000065C0: BFC60001
	v_dual_fmac_f32 v8, v9, v11 :: v_dual_fmac_f32 v7, v10, v12// 0000000065C4: C8001709 0806190A
	s_wait_dscnt 0x0                                           // 0000000065CC: BFC60000
	v_fmac_f32_e32 v5, v2, v13                                 // 0000000065D0: 560A1B02
	s_cbranch_scc0 L89                                         // 0000000065D4: BFA1FFE9
	s_branch L87                                               // 0000000065D8: BFA0FFD2

00000000000065dc <L90>:
	v_dual_mov_b32 v8, 0 :: v_dual_mov_b32 v7, 0               // 0000000065DC: CA100080 08060080
	v_mov_b32_e32 v5, 0                                        // 0000000065E4: 7E0A0280

00000000000065e8 <L91>:
	v_mov_b16_e32 v0.l, 0xff                                   // 0000000065E8: 7E0038FF 000000FF
	s_mov_b32 s2, exec_lo                                      // 0000000065F0: BE82007E
	s_delay_alu instid0(VALU_DEP_3)                            // 0000000065F4: BF870003
	v_cmpx_nlt_f32_e32 0x437f0000, v8                          // 0000000065F8: 7D3C10FF 437F0000
	s_cbranch_execz L96                                        // 000000006600: BFA50034
	v_mov_b16_e32 v0.l, 0                                      // 000000006604: 7E003880
	s_mov_b32 s4, exec_lo                                      // 000000006608: BE84007E
	v_cmpx_ngt_f32_e32 0, v8                                   // 00000000660C: 7D361080
	s_cbranch_execz L95                                        // 000000006610: BFA5002E
	s_mov_b32 s5, exec_lo                                      // 000000006614: BE85007E
	v_cmpx_o_f32_e32 v8, v8                                    // 000000006618: 7D2E1108
	s_cbranch_execz L94                                        // 00000000661C: BFA50028
	v_add_f32_e32 v0, v8, v8                                   // 000000006620: 06001108
	v_cmp_nlg_f32_e32 vcc_lo, 0, v8                            // 000000006624: 7C341080
	v_cmp_neq_f32_e64 s1, 0, v8                                // 000000006628: D41D0001 02021080
	s_delay_alu instid0(VALU_DEP_3)                            // 000000006630: BF870003
	v_cmp_neq_f32_e64 s0, v0, v8                               // 000000006634: D41D0000 02021100
	s_or_b32 s0, vcc_lo, s0                                    // 00000000663C: 8C00006A
	s_wait_alu depctr_sa_sdst(0)                               // 000000006640: BF88FF9E
	s_and_b32 s1, s1, s0                                       // 000000006644: 8B010001
	s_wait_alu depctr_sa_sdst(0)                               // 000000006648: BF88FF9E
	s_and_saveexec_b32 s0, s1                                  // 00000000664C: BE802001
	s_cbranch_execz L93                                        // 000000006650: BFA50019
	v_cvt_i32_f32_e32 v1, v8                                   // 000000006654: 7E021108
	s_mov_b32 s1, exec_lo                                      // 000000006658: BE81007E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000665C: BF870091
	v_cvt_f32_i32_e32 v0, v1                                   // 000000006660: 7E000B01
	v_sub_f32_e32 v2, v8, v0                                   // 000000006664: 08040108
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006668: BF870001
	v_cmpx_ngt_f32_e32 0.5, v2                                 // 00000000666C: 7D3604F0
	s_cbranch_execz L92                                        // 000000006670: BFA5000D
	s_mov_b32 s6, exec_lo                                      // 000000006674: BE86007E
	v_cmpx_nlt_f32_e32 0.5, v2                                 // 000000006678: 7D3C04F0
	s_xor_b32 s6, exec_lo, s6                                  // 00000000667C: 8D06067E
	v_dual_add_f32 v2, 1.0, v0 :: v_dual_and_b32 v1, 1, v1     // 000000006680: C92400F2 02000281
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000006688: BF870121
	v_cmp_eq_u32_e32 vcc_lo, 0, v1                             // 00000000668C: 7C940280
	s_wait_alu depctr_va_vcc(0)                                // 000000006690: BF88FF9D
	v_cndmask_b32_e32 v0, v2, v0, vcc_lo                       // 000000006694: 02000102
	s_and_not1_saveexec_b32 s6, s6                             // 000000006698: BE863006
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000669C: BF870001
	v_add_f32_e32 v0, 1.0, v0                                  // 0000000066A0: 060000F2
	s_or_b32 exec_lo, exec_lo, s6                              // 0000000066A4: 8C7E067E

00000000000066a8 <L92>:
	s_wait_alu depctr_sa_sdst(0)                               // 0000000066A8: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000066AC: 8C7E017E
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000066B0: BF870001
	v_mov_b32_e32 v8, v0                                       // 0000000066B4: 7E100300

00000000000066b8 <L93>:
	s_wait_alu depctr_sa_sdst(0)                               // 0000000066B8: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s0                              // 0000000066BC: 8C7E007E

00000000000066c0 <L94>:
	s_wait_alu depctr_sa_sdst(0)                               // 0000000066C0: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s5                              // 0000000066C4: 8C7E057E
	v_cvt_i32_f32_e32 v0, v8                                   // 0000000066C8: 7E001108

00000000000066cc <L95>:
	s_wait_alu depctr_sa_sdst(0)                               // 0000000066CC: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s4                              // 0000000066D0: 8C7E047E

00000000000066d4 <L96>:
	s_wait_alu depctr_sa_sdst(0)                               // 0000000066D4: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s2                              // 0000000066D8: 8C7E027E
	v_mov_b32_e32 v1, 0xff00                                   // 0000000066DC: 7E0202FF 0000FF00
	s_mov_b32 s2, exec_lo                                      // 0000000066E4: BE82007E
	v_cmpx_nlt_f32_e32 0x437f0000, v7                          // 0000000066E8: 7D3C0EFF 437F0000
	s_cbranch_execz L101                                       // 0000000066F0: BFA50037
	v_mov_b32_e32 v1, 0                                        // 0000000066F4: 7E020280
	s_mov_b32 s4, exec_lo                                      // 0000000066F8: BE84007E
	v_cmpx_ngt_f32_e32 0, v7                                   // 0000000066FC: 7D360E80
	s_cbranch_execz L100                                       // 000000006700: BFA50031
	s_mov_b32 s5, exec_lo                                      // 000000006704: BE85007E
	v_cmpx_o_f32_e32 v7, v7                                    // 000000006708: 7D2E0F07
	s_cbranch_execz L99                                        // 00000000670C: BFA50028
	v_add_f32_e32 v1, v7, v7                                   // 000000006710: 06020F07
	v_cmp_nlg_f32_e32 vcc_lo, 0, v7                            // 000000006714: 7C340E80
	v_cmp_neq_f32_e64 s1, 0, v7                                // 000000006718: D41D0001 02020E80
	s_delay_alu instid0(VALU_DEP_3)                            // 000000006720: BF870003
	v_cmp_neq_f32_e64 s0, v1, v7                               // 000000006724: D41D0000 02020F01
	s_or_b32 s0, vcc_lo, s0                                    // 00000000672C: 8C00006A
	s_wait_alu depctr_sa_sdst(0)                               // 000000006730: BF88FF9E
	s_and_b32 s1, s1, s0                                       // 000000006734: 8B010001
	s_wait_alu depctr_sa_sdst(0)                               // 000000006738: BF88FF9E
	s_and_saveexec_b32 s0, s1                                  // 00000000673C: BE802001
	s_cbranch_execz L98                                        // 000000006740: BFA50019
	v_cvt_i32_f32_e32 v2, v7                                   // 000000006744: 7E041107
	s_mov_b32 s1, exec_lo                                      // 000000006748: BE81007E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000674C: BF870091
	v_cvt_f32_i32_e32 v1, v2                                   // 000000006750: 7E020B02
	v_sub_f32_e32 v6, v7, v1                                   // 000000006754: 080C0307
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006758: BF870001
	v_cmpx_ngt_f32_e32 0.5, v6                                 // 00000000675C: 7D360CF0
	s_cbranch_execz L97                                        // 000000006760: BFA5000D
	s_mov_b32 s6, exec_lo                                      // 000000006764: BE86007E
	v_cmpx_nlt_f32_e32 0.5, v6                                 // 000000006768: 7D3C0CF0
	s_xor_b32 s6, exec_lo, s6                                  // 00000000676C: 8D06067E
	v_add_f32_e32 v6, 1.0, v1                                  // 000000006770: 060C02F2
	v_and_b32_e32 v2, 1, v2                                    // 000000006774: 36040481
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)// 000000006778: BF8701A1
	v_cmp_eq_u32_e32 vcc_lo, 0, v2                             // 00000000677C: 7C940480
	s_wait_alu depctr_va_vcc(0)                                // 000000006780: BF88FF9D
	v_cndmask_b32_e32 v1, v6, v1, vcc_lo                       // 000000006784: 02020306
	s_and_not1_saveexec_b32 s6, s6                             // 000000006788: BE863006
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000678C: BF870001
	v_add_f32_e32 v1, 1.0, v1                                  // 000000006790: 060202F2
	s_or_b32 exec_lo, exec_lo, s6                              // 000000006794: 8C7E067E

0000000000006798 <L97>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000006798: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000679C: 8C7E017E
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000067A0: BF870001
	v_mov_b32_e32 v7, v1                                       // 0000000067A4: 7E0E0301

00000000000067a8 <L98>:
	s_wait_alu depctr_sa_sdst(0)                               // 0000000067A8: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s0                              // 0000000067AC: 8C7E007E

00000000000067b0 <L99>:
	s_wait_alu depctr_sa_sdst(0)                               // 0000000067B0: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s5                              // 0000000067B4: 8C7E057E
	v_cvt_i32_f32_e32 v1, v7                                   // 0000000067B8: 7E021107
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000067BC: BF870091
	v_cvt_u32_u16_e32 v1, v1.l                                 // 0000000067C0: 7E02D701
	v_lshlrev_b32_e32 v1, 8, v1                                // 0000000067C4: 30020288

00000000000067c8 <L100>:
	s_wait_alu depctr_sa_sdst(0)                               // 0000000067C8: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s4                              // 0000000067CC: 8C7E047E

00000000000067d0 <L101>:
	s_wait_alu depctr_sa_sdst(0)                               // 0000000067D0: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s2                              // 0000000067D4: 8C7E027E
	v_mov_b32_e32 v2, 0xff0000                                 // 0000000067D8: 7E0402FF 00FF0000
	s_mov_b32 s2, exec_lo                                      // 0000000067E0: BE82007E
	v_cmpx_nlt_f32_e32 0x437f0000, v5                          // 0000000067E4: 7D3C0AFF 437F0000
	s_cbranch_execz L106                                       // 0000000067EC: BFA50036
	v_mov_b32_e32 v2, 0                                        // 0000000067F0: 7E040280
	s_mov_b32 s4, exec_lo                                      // 0000000067F4: BE84007E
	v_cmpx_ngt_f32_e32 0, v5                                   // 0000000067F8: 7D360A80
	s_cbranch_execz L105                                       // 0000000067FC: BFA50030
	s_mov_b32 s5, exec_lo                                      // 000000006800: BE85007E
	v_cmpx_o_f32_e32 v5, v5                                    // 000000006804: 7D2E0B05
	s_cbranch_execz L104                                       // 000000006808: BFA50028
	v_add_f32_e32 v2, v5, v5                                   // 00000000680C: 06040B05
	v_cmp_nlg_f32_e32 vcc_lo, 0, v5                            // 000000006810: 7C340A80
	v_cmp_neq_f32_e64 s1, 0, v5                                // 000000006814: D41D0001 02020A80
	s_delay_alu instid0(VALU_DEP_3)                            // 00000000681C: BF870003
	v_cmp_neq_f32_e64 s0, v2, v5                               // 000000006820: D41D0000 02020B02
	s_or_b32 s0, vcc_lo, s0                                    // 000000006828: 8C00006A
	s_wait_alu depctr_sa_sdst(0)                               // 00000000682C: BF88FF9E
	s_and_b32 s1, s1, s0                                       // 000000006830: 8B010001
	s_wait_alu depctr_sa_sdst(0)                               // 000000006834: BF88FF9E
	s_and_saveexec_b32 s0, s1                                  // 000000006838: BE802001
	s_cbranch_execz L103                                       // 00000000683C: BFA50019
	v_cvt_i32_f32_e32 v6, v5                                   // 000000006840: 7E0C1105
	s_mov_b32 s1, exec_lo                                      // 000000006844: BE81007E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000006848: BF870091
	v_cvt_f32_i32_e32 v2, v6                                   // 00000000684C: 7E040B06
	v_sub_f32_e32 v5, v5, v2                                   // 000000006850: 080A0505
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006854: BF870001
	v_cmpx_ngt_f32_e32 0.5, v5                                 // 000000006858: 7D360AF0
	s_cbranch_execz L102                                       // 00000000685C: BFA5000D
	s_mov_b32 s6, exec_lo                                      // 000000006860: BE86007E
	v_cmpx_nlt_f32_e32 0.5, v5                                 // 000000006864: 7D3C0AF0
	s_xor_b32 s6, exec_lo, s6                                  // 000000006868: 8D06067E
	v_and_b32_e32 v5, 1, v6                                    // 00000000686C: 360A0C81
	v_add_f32_e32 v6, 1.0, v2                                  // 000000006870: 060C04F2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000006874: BF870122
	v_cmp_eq_u32_e32 vcc_lo, 0, v5                             // 000000006878: 7C940A80
	s_wait_alu depctr_va_vcc(0)                                // 00000000687C: BF88FF9D
	v_cndmask_b32_e32 v2, v6, v2, vcc_lo                       // 000000006880: 02040506
	s_and_not1_saveexec_b32 s6, s6                             // 000000006884: BE863006
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006888: BF870001
	v_add_f32_e32 v2, 1.0, v2                                  // 00000000688C: 060404F2
	s_or_b32 exec_lo, exec_lo, s6                              // 000000006890: 8C7E067E

0000000000006894 <L102>:
	s_wait_alu depctr_sa_sdst(0)                               // 000000006894: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s1                              // 000000006898: 8C7E017E
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000689C: BF870001
	v_mov_b32_e32 v5, v2                                       // 0000000068A0: 7E0A0302

00000000000068a4 <L103>:
	s_wait_alu depctr_sa_sdst(0)                               // 0000000068A4: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s0                              // 0000000068A8: 8C7E007E

00000000000068ac <L104>:
	s_wait_alu depctr_sa_sdst(0)                               // 0000000068AC: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s5                              // 0000000068B0: 8C7E057E
	v_cvt_i32_f32_e32 v2, v5                                   // 0000000068B4: 7E041105
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000068B8: BF870001
	v_lshlrev_b32_e32 v2, 16, v2                               // 0000000068BC: 30040490

00000000000068c0 <L105>:
	s_wait_alu depctr_sa_sdst(0)                               // 0000000068C0: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s4                              // 0000000068C4: 8C7E047E

00000000000068c8 <L106>:
	s_wait_alu depctr_sa_sdst(0)                               // 0000000068C8: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s2                              // 0000000068CC: 8C7E027E
	s_wait_kmcnt 0x0                                           // 0000000068D0: BFC70000
	v_mul_lo_u32 v4, s3, v4                                    // 0000000068D4: D72C0004 02020803
	v_and_or_b32 v0, 0xff00, v1, v0                            // 0000000068DC: D6570000 040202FF 0000FF00
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 0000000068E8: BF8700B2
	v_add_co_u32 v4, s0, s12, v4                               // 0000000068EC: D7000004 0202080C
	s_wait_alu depctr_va_sdst(0)                               // 0000000068F4: BF88F19F
	v_add_co_ci_u32_e64 v5, null, s13, 0, s0                   // 0000000068F8: D5207C05 0001000D
	v_mad_co_i64_i32 v[3:4], null, v3, 3, v[4:5]               // 000000006900: D6FF7C03 04110703
	s_clause 0x1                                               // 000000006908: BF850001
	global_store_d16_hi_b8 v[3:4], v2, off offset:2            // 00000000690C: EE09007C 01000000 00000203
	global_store_b16 v[3:4], v0, off                           // 000000006918: EE06407C 00000000 00000003

0000000000006924 <L107>:
	s_endpgm                                                   // 000000006924: BFB00000
