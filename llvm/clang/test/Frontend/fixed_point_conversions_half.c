// NOTE: Assertions have been autogenerated by utils/update_cc_test_checks.py
// RUN: %clang_cc1 -no-opaque-pointers -ffixed-point -triple arm64-unknown-linux-gnu -S -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,SIGNED
// RUN: %clang_cc1 -no-opaque-pointers -ffixed-point -triple arm64-unknown-linux-gnu -S -emit-llvm %s -o - -fpadding-on-unsigned-fixed-point | FileCheck %s --check-prefixes=CHECK,UNSIGNED

short _Fract sf;
long _Fract lf;

short _Accum sa;
long _Accum la;

unsigned short _Accum usa;
unsigned long _Accum ula;

_Sat short _Fract sf_sat;
_Sat long _Fract lf_sat;

_Sat short _Accum sa_sat;
_Sat long _Accum la_sat;

_Sat unsigned short _Accum usa_sat;
_Sat unsigned long _Accum ula_sat;

_Float16 h;


// CHECK-LABEL: @half_fix1(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = load half, half* @h, align 2
// CHECK-NEXT:    [[TMP1:%.*]] = fmul half [[TMP0]], 0xH5800
// CHECK-NEXT:    [[TMP2:%.*]] = fptosi half [[TMP1]] to i8
// CHECK-NEXT:    store i8 [[TMP2]], i8* @sf, align 1
// CHECK-NEXT:    ret void
//
void half_fix1(void) {
  sf = h;
}

// CHECK-LABEL: @half_fix2(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = load half, half* @h, align 2
// CHECK-NEXT:    [[TMP1:%.*]] = fpext half [[TMP0]] to float
// CHECK-NEXT:    [[TMP2:%.*]] = fmul float [[TMP1]], 0x41E0000000000000
// CHECK-NEXT:    [[TMP3:%.*]] = fptosi float [[TMP2]] to i32
// CHECK-NEXT:    store i32 [[TMP3]], i32* @lf, align 4
// CHECK-NEXT:    ret void
//
void half_fix2(void) {
  lf = h;
}

// CHECK-LABEL: @half_fix3(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = load half, half* @h, align 2
// CHECK-NEXT:    [[TMP1:%.*]] = fmul half [[TMP0]], 0xH5800
// CHECK-NEXT:    [[TMP2:%.*]] = fptosi half [[TMP1]] to i16
// CHECK-NEXT:    store i16 [[TMP2]], i16* @sa, align 2
// CHECK-NEXT:    ret void
//
void half_fix3(void) {
  sa = h;
}

// CHECK-LABEL: @half_fix4(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = load half, half* @h, align 2
// CHECK-NEXT:    [[TMP1:%.*]] = fpext half [[TMP0]] to float
// CHECK-NEXT:    [[TMP2:%.*]] = fmul float [[TMP1]], 0x41E0000000000000
// CHECK-NEXT:    [[TMP3:%.*]] = fptosi float [[TMP2]] to i64
// CHECK-NEXT:    store i64 [[TMP3]], i64* @la, align 8
// CHECK-NEXT:    ret void
//
void half_fix4(void) {
  la = h;
}

// SIGNED-LABEL: @half_fix5(
// SIGNED-NEXT:  entry:
// SIGNED-NEXT:    [[TMP0:%.*]] = load half, half* @h, align 2
// SIGNED-NEXT:    [[TMP1:%.*]] = fpext half [[TMP0]] to float
// SIGNED-NEXT:    [[TMP2:%.*]] = fmul float [[TMP1]], 2.560000e+02
// SIGNED-NEXT:    [[TMP3:%.*]] = fptoui float [[TMP2]] to i16
// SIGNED-NEXT:    store i16 [[TMP3]], i16* @usa, align 2
// SIGNED-NEXT:    ret void
//
// UNSIGNED-LABEL: @half_fix5(
// UNSIGNED-NEXT:  entry:
// UNSIGNED-NEXT:    [[TMP0:%.*]] = load half, half* @h, align 2
// UNSIGNED-NEXT:    [[TMP1:%.*]] = fmul half [[TMP0]], 0xH5800
// UNSIGNED-NEXT:    [[TMP2:%.*]] = fptosi half [[TMP1]] to i16
// UNSIGNED-NEXT:    store i16 [[TMP2]], i16* @usa, align 2
// UNSIGNED-NEXT:    ret void
//
void half_fix5(void) {
  usa = h;
}

// SIGNED-LABEL: @half_fix6(
// SIGNED-NEXT:  entry:
// SIGNED-NEXT:    [[TMP0:%.*]] = load half, half* @h, align 2
// SIGNED-NEXT:    [[TMP1:%.*]] = fpext half [[TMP0]] to float
// SIGNED-NEXT:    [[TMP2:%.*]] = fmul float [[TMP1]], 0x41F0000000000000
// SIGNED-NEXT:    [[TMP3:%.*]] = fptoui float [[TMP2]] to i64
// SIGNED-NEXT:    store i64 [[TMP3]], i64* @ula, align 8
// SIGNED-NEXT:    ret void
//
// UNSIGNED-LABEL: @half_fix6(
// UNSIGNED-NEXT:  entry:
// UNSIGNED-NEXT:    [[TMP0:%.*]] = load half, half* @h, align 2
// UNSIGNED-NEXT:    [[TMP1:%.*]] = fpext half [[TMP0]] to float
// UNSIGNED-NEXT:    [[TMP2:%.*]] = fmul float [[TMP1]], 0x41E0000000000000
// UNSIGNED-NEXT:    [[TMP3:%.*]] = fptosi float [[TMP2]] to i64
// UNSIGNED-NEXT:    store i64 [[TMP3]], i64* @ula, align 8
// UNSIGNED-NEXT:    ret void
//
void half_fix6(void) {
  ula = h;
}


// CHECK-LABEL: @half_sat1(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = load half, half* @h, align 2
// CHECK-NEXT:    [[TMP1:%.*]] = fmul half [[TMP0]], 0xH5800
// CHECK-NEXT:    [[TMP2:%.*]] = call i8 @llvm.fptosi.sat.i8.f16(half [[TMP1]])
// CHECK-NEXT:    store i8 [[TMP2]], i8* @sf_sat, align 1
// CHECK-NEXT:    ret void
//
void half_sat1(void) {
  sf_sat = h;
}

// CHECK-LABEL: @half_sat2(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = load half, half* @h, align 2
// CHECK-NEXT:    [[TMP1:%.*]] = fpext half [[TMP0]] to float
// CHECK-NEXT:    [[TMP2:%.*]] = fmul float [[TMP1]], 0x41E0000000000000
// CHECK-NEXT:    [[TMP3:%.*]] = call i32 @llvm.fptosi.sat.i32.f32(float [[TMP2]])
// CHECK-NEXT:    store i32 [[TMP3]], i32* @lf_sat, align 4
// CHECK-NEXT:    ret void
//
void half_sat2(void) {
  lf_sat = h;
}

// CHECK-LABEL: @half_sat3(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = load half, half* @h, align 2
// CHECK-NEXT:    [[TMP1:%.*]] = fmul half [[TMP0]], 0xH5800
// CHECK-NEXT:    [[TMP2:%.*]] = call i16 @llvm.fptosi.sat.i16.f16(half [[TMP1]])
// CHECK-NEXT:    store i16 [[TMP2]], i16* @sa_sat, align 2
// CHECK-NEXT:    ret void
//
void half_sat3(void) {
  sa_sat = h;
}

// CHECK-LABEL: @half_sat4(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = load half, half* @h, align 2
// CHECK-NEXT:    [[TMP1:%.*]] = fpext half [[TMP0]] to float
// CHECK-NEXT:    [[TMP2:%.*]] = fmul float [[TMP1]], 0x41E0000000000000
// CHECK-NEXT:    [[TMP3:%.*]] = call i64 @llvm.fptosi.sat.i64.f32(float [[TMP2]])
// CHECK-NEXT:    store i64 [[TMP3]], i64* @la_sat, align 8
// CHECK-NEXT:    ret void
//
void half_sat4(void) {
  la_sat = h;
}

// SIGNED-LABEL: @half_sat5(
// SIGNED-NEXT:  entry:
// SIGNED-NEXT:    [[TMP0:%.*]] = load half, half* @h, align 2
// SIGNED-NEXT:    [[TMP1:%.*]] = fpext half [[TMP0]] to float
// SIGNED-NEXT:    [[TMP2:%.*]] = fmul float [[TMP1]], 2.560000e+02
// SIGNED-NEXT:    [[TMP3:%.*]] = call i16 @llvm.fptoui.sat.i16.f32(float [[TMP2]])
// SIGNED-NEXT:    store i16 [[TMP3]], i16* @usa_sat, align 2
// SIGNED-NEXT:    ret void
//
// UNSIGNED-LABEL: @half_sat5(
// UNSIGNED-NEXT:  entry:
// UNSIGNED-NEXT:    [[TMP0:%.*]] = load half, half* @h, align 2
// UNSIGNED-NEXT:    [[TMP1:%.*]] = fmul half [[TMP0]], 0xH5800
// UNSIGNED-NEXT:    [[TMP2:%.*]] = call i16 @llvm.fptosi.sat.i16.f16(half [[TMP1]])
// UNSIGNED-NEXT:    [[TMP3:%.*]] = icmp slt i16 [[TMP2]], 0
// UNSIGNED-NEXT:    [[SATMIN:%.*]] = select i1 [[TMP3]], i16 0, i16 [[TMP2]]
// UNSIGNED-NEXT:    store i16 [[SATMIN]], i16* @usa_sat, align 2
// UNSIGNED-NEXT:    ret void
//
void half_sat5(void) {
  usa_sat = h;
}

// SIGNED-LABEL: @half_sat6(
// SIGNED-NEXT:  entry:
// SIGNED-NEXT:    [[TMP0:%.*]] = load half, half* @h, align 2
// SIGNED-NEXT:    [[TMP1:%.*]] = fpext half [[TMP0]] to float
// SIGNED-NEXT:    [[TMP2:%.*]] = fmul float [[TMP1]], 0x41F0000000000000
// SIGNED-NEXT:    [[TMP3:%.*]] = call i64 @llvm.fptoui.sat.i64.f32(float [[TMP2]])
// SIGNED-NEXT:    store i64 [[TMP3]], i64* @ula_sat, align 8
// SIGNED-NEXT:    ret void
//
// UNSIGNED-LABEL: @half_sat6(
// UNSIGNED-NEXT:  entry:
// UNSIGNED-NEXT:    [[TMP0:%.*]] = load half, half* @h, align 2
// UNSIGNED-NEXT:    [[TMP1:%.*]] = fpext half [[TMP0]] to float
// UNSIGNED-NEXT:    [[TMP2:%.*]] = fmul float [[TMP1]], 0x41E0000000000000
// UNSIGNED-NEXT:    [[TMP3:%.*]] = call i64 @llvm.fptosi.sat.i64.f32(float [[TMP2]])
// UNSIGNED-NEXT:    [[TMP4:%.*]] = icmp slt i64 [[TMP3]], 0
// UNSIGNED-NEXT:    [[SATMIN:%.*]] = select i1 [[TMP4]], i64 0, i64 [[TMP3]]
// UNSIGNED-NEXT:    store i64 [[SATMIN]], i64* @ula_sat, align 8
// UNSIGNED-NEXT:    ret void
//
void half_sat6(void) {
  ula_sat = h;
}


// CHECK-LABEL: @fix_half1(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = load i8, i8* @sf, align 1
// CHECK-NEXT:    [[TMP1:%.*]] = sitofp i8 [[TMP0]] to half
// CHECK-NEXT:    [[TMP2:%.*]] = fmul half [[TMP1]], 0xH2000
// CHECK-NEXT:    store half [[TMP2]], half* @h, align 2
// CHECK-NEXT:    ret void
//
void fix_half1(void) {
  h = sf;
}

// CHECK-LABEL: @fix_half2(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, i32* @lf, align 4
// CHECK-NEXT:    [[TMP1:%.*]] = sitofp i32 [[TMP0]] to float
// CHECK-NEXT:    [[TMP2:%.*]] = fmul float [[TMP1]], 0x3E00000000000000
// CHECK-NEXT:    [[TMP3:%.*]] = fptrunc float [[TMP2]] to half
// CHECK-NEXT:    store half [[TMP3]], half* @h, align 2
// CHECK-NEXT:    ret void
//
void fix_half2(void) {
  h = lf;
}

// CHECK-LABEL: @fix_half3(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = load i16, i16* @sa, align 2
// CHECK-NEXT:    [[TMP1:%.*]] = sitofp i16 [[TMP0]] to half
// CHECK-NEXT:    [[TMP2:%.*]] = fmul half [[TMP1]], 0xH2000
// CHECK-NEXT:    store half [[TMP2]], half* @h, align 2
// CHECK-NEXT:    ret void
//
void fix_half3(void) {
  h = sa;
}

// CHECK-LABEL: @fix_half4(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = load i64, i64* @la, align 8
// CHECK-NEXT:    [[TMP1:%.*]] = sitofp i64 [[TMP0]] to float
// CHECK-NEXT:    [[TMP2:%.*]] = fmul float [[TMP1]], 0x3E00000000000000
// CHECK-NEXT:    [[TMP3:%.*]] = fptrunc float [[TMP2]] to half
// CHECK-NEXT:    store half [[TMP3]], half* @h, align 2
// CHECK-NEXT:    ret void
//
void fix_half4(void) {
  h = la;
}

// SIGNED-LABEL: @fix_half5(
// SIGNED-NEXT:  entry:
// SIGNED-NEXT:    [[TMP0:%.*]] = load i16, i16* @usa, align 2
// SIGNED-NEXT:    [[TMP1:%.*]] = uitofp i16 [[TMP0]] to float
// SIGNED-NEXT:    [[TMP2:%.*]] = fmul float [[TMP1]], 3.906250e-03
// SIGNED-NEXT:    [[TMP3:%.*]] = fptrunc float [[TMP2]] to half
// SIGNED-NEXT:    store half [[TMP3]], half* @h, align 2
// SIGNED-NEXT:    ret void
//
// UNSIGNED-LABEL: @fix_half5(
// UNSIGNED-NEXT:  entry:
// UNSIGNED-NEXT:    [[TMP0:%.*]] = load i16, i16* @usa, align 2
// UNSIGNED-NEXT:    [[TMP1:%.*]] = uitofp i16 [[TMP0]] to half
// UNSIGNED-NEXT:    [[TMP2:%.*]] = fmul half [[TMP1]], 0xH2000
// UNSIGNED-NEXT:    store half [[TMP2]], half* @h, align 2
// UNSIGNED-NEXT:    ret void
//
void fix_half5(void) {
  h = usa;
}

// SIGNED-LABEL: @fix_half6(
// SIGNED-NEXT:  entry:
// SIGNED-NEXT:    [[TMP0:%.*]] = load i64, i64* @ula, align 8
// SIGNED-NEXT:    [[TMP1:%.*]] = uitofp i64 [[TMP0]] to float
// SIGNED-NEXT:    [[TMP2:%.*]] = fmul float [[TMP1]], 0x3DF0000000000000
// SIGNED-NEXT:    [[TMP3:%.*]] = fptrunc float [[TMP2]] to half
// SIGNED-NEXT:    store half [[TMP3]], half* @h, align 2
// SIGNED-NEXT:    ret void
//
// UNSIGNED-LABEL: @fix_half6(
// UNSIGNED-NEXT:  entry:
// UNSIGNED-NEXT:    [[TMP0:%.*]] = load i64, i64* @ula, align 8
// UNSIGNED-NEXT:    [[TMP1:%.*]] = uitofp i64 [[TMP0]] to float
// UNSIGNED-NEXT:    [[TMP2:%.*]] = fmul float [[TMP1]], 0x3E00000000000000
// UNSIGNED-NEXT:    [[TMP3:%.*]] = fptrunc float [[TMP2]] to half
// UNSIGNED-NEXT:    store half [[TMP3]], half* @h, align 2
// UNSIGNED-NEXT:    ret void
//
void fix_half6(void) {
  h = ula;
}