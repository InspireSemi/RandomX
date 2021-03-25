// Copyright (c) 2018-2019, tevador <tevador@gmail.com>
// Copyright (c) 2019, SChernykh    <https://github.com/SChernykh>

// All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 	* Redistributions of source code must retain the above copyright
// 	  notice, this list of conditions and the following disclaimer.
// 	* Redistributions in binary form must reproduce the above copyright
// 	  notice, this list of conditions and the following disclaimer in the
// 	  documentation and/or other materials provided with the distribution.
// 	* Neither the name of the copyright holder nor the
// 	  names of its contributors may be used to endorse or promote products
// 	  derived from this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THxorY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


.option push
.option norvc

#define DECL(x) x
	.text
	.global DECL(randomx_program_rv64)
	.global DECL(randomx_program_rv64_main_loop)
	.global DECL(randomx_program_rv64_vm_instructions)
	.global DECL(randomx_program_rv64_imul_rcp_literals_end)
	.global DECL(randomx_program_rv64_vm_instructions_end)
	.global DECL(randomx_program_rv64_cacheline_align_mask1)
	.global DECL(randomx_program_rv64_cacheline_align_mask2)
	.global DECL(randomx_program_rv64_update_spMix1)
	.global DECL(randomx_program_rv64_vm_instructions_end_light)
	.global DECL(randomx_program_rv64_light_cacheline_align_mask)
	.global DECL(randomx_program_rv64_light_dataset_offset)
	.global DECL(randomx_init_dataset_rv64)
	.global DECL(randomx_init_dataset_rv64_end)
	.global DECL(randomx_calc_dataset_item_rv64)
	.global DECL(randomx_calc_dataset_item_rv64_prefetch)
	.global DECL(randomx_calc_dataset_item_rv64_mix)
	.global DECL(randomx_calc_dataset_item_rv64_store_result)
	.global DECL(randomx_calc_dataset_item_rv64_end)

#include "configuration.h"

//args from c:   void randomx_program_rv64(void* reg, void* mem, void* scratchpad, uint64_t iterations);
// a0 -> pointer to "reg" buffer
// a1 -> pointer to "mem" buffer;       then (when?) to dataset
// a2 -> pointer to scratchpad                
// a3 -> loop iterations 


// Register allocation
// x0    ->  0
#define dsPtr x5 // t0: pointer to dataset
#define spPtr x6 // t1: pointer to scratchpad
#define loopCnt x7 // t2:  loop counter
#define r0 x28   // t3: "r0"
#define r1 x29   // t4: "r1"
#define r2 x30   // t5: "r2"
#define r3 x31   // t6: "r3"
//X8 free   //#define fprc x8 // x8  -> fprc 
#define mamx x9 // s1: mx, ma
#define spMix1 x18   // s2: spMix1
#define regPtr x19 // s3: pointer to reg buffer (OUTPUT only)  //was in x0; x11 was a literal for IMUL_RCP
#define r4 x20 // s4: "r4"
#define r5 x21 // s5: "r5"
#define r6 x22 // s6: "r6"
#define r7 x23 // s7: "r7"
#define spAddr0 x24 // s8: spAddr0
#define spAddr1 x25 // s9: spAddr1
#define temp0 x26 // s10: temporary
#define temp1 x27 // s11: temporary
// x4 -> literal for IMUL_RCP	tp
// x8 -> literal for IMUL_RCP	s0
// x3 -> literal for IMUL_RCP	gp
// x10 -> literal for IMUL_RCP	a0
// x11 -> literal for IMUL_RCP	a1
// x12 -> literal for IMUL_RCP	a2
// x13 -> literal for IMUL_RCP	a3
// x14 -> literal for IMUL_RCP	a4
// x15 -> literal for IMUL_RCP	a5
// x16 -> literal for IMUL_RCP	a6
// x17 -> literal for IMUL_RCP	a7

#use F regs, lower double in fN, upper in fN+1
// f0 -> "f0.0"
// f1 -> "f0.1"      
// f2 -> "f1.0"
// f3 -> "f1.1"
// f4 -> "f2.0"
// f5 -> "f2.1"
// f6 -> "f3.0"
// f7 -> "f3.1"
// f8 -> "e0.0"
// f9 -> "e0.1"
// f10 -> "e1.0"
// f11 -> "e1.1"
// f12 -> "e2.0"
// f13 -> "e2.1"
// f14 -> "e3.0"
// f15 -> "e3.1"
// f16 -> "a0.0"
// f17 -> "a0.1"
// f18 -> "a1.0"
// f19 -> "a1.1"
// f20 -> "a2.0"
// f21 -> "a2.1"
// f22 -> "a3.0"
// f23 -> "a3.1"
// f24 -> temporary.0
// f25 -> temporary.1
// f26 -> E 'and' mask = 0x00ffffffffffffff
// f28 -> E 'or' mask = 0x3*00000000******
// f30 -> scale mask s= 0x81f0000000000000

// v0-v15/v31 -> store 32-bit literals


.macro fxord fd, fs1, fs2, tmp1, tmp2
	fmv.x.d \tmp1, \fs1
	fmv.x.d \tmp2, \fs2
	xor \tmp1, \tmp1, \tmp2
	fmv.d.x \fd, \tmp1
.endm

.macro forrd fd, fs1, fs2, tmp1, tmp2
	fmv.x.d \tmp1, \fs1
	fmv.x.d \tmp2, \fs2
	or \tmp1, \tmp1, \tmp2
	fmv.d.x \fd, \tmp1
.endm

.macro fandd fd, fs1, fs2, tmp1, tmp2
	fmv.x.d \tmp1, \fs1
	fmv.x.d \tmp2, \fs2
	and \tmp1, \tmp1, \tmp2
	fmv.d.x \fd, \tmp1
.endm


	.balign 8



	
// Only symbol called with arguments from C: randomx_program_rv64
// void randomx_program_rv64(void* reg, void* mem, void* scratchpad, uint64_t iterations);
DECL(randomx_program_rv64):
	// Save callee-saved registers (S0-S11, tp, gp)
	addi sp, sp, -192
	sd s0,  0(sp)
	sd s1,  8(sp)
	sd s2, 16(sp)
	sd s3, 24(sp)
	sd s4, 32(sp)
	sd s5, 40(sp)
	sd s6, 48(sp)
	sd s7, 56(sp)
	sd s8, 64(sp)
	sd s9, 72(sp)
	sd s10, 80(sp)
	sd s11, 88(sp)
	sd gp, 96(sp)
	sd tp, 104(sp)
		
	//init scratchpad ptr from argument
	mv regPtr, a0
	
	//init scratchpad ptr from argument
	mv spPtr, a2

	//init counter (ic) with total iterations
	mv loopCnt, a3 

	//init ma,mx	
	ld mamx, 0(a1)  

	//init dataset pointer
	ld dsPtr, 8(a1)

	// Load initial spMix value   //?? not in spec?
	mv	spMix1, mamx

	//init spAddr0 = mx
	//mv spAddr0, mamx

	//init spaddr1 = ma
	//srli spAddr1, mamx, 32

	// Zero integer registers
	li	r0, 0
	li	r1, 0
	li	r2, 0
	li	r3, 0
	li	r4, 0
	li	r5, 0
	li	r6, 0
	li	r7, 0

	// Load group A registers
	// Load from 0xC8 - 0xF8
	fld f16, 0xC8(a0)  //offsets seem off vs spec... check c code
	fld f17, 0xD0(a0)
	fld f18, 0xD8(a0)
	fld f19, 0xE0(a0)
	fld f20, 0xE8(a0)
	fld f21, 0xF0(a0)
	fld f22, 0xF8(a0)
	fld f23, 0x100(a0)


	// Load E 'and' mask
	// 128 bit puts this same value in upper and lower 64 bit words
	li	temp0, 0x00FFFFFFFFFFFFFF
	fmv.d.x	f26, temp0

	// Load E 'or' mask
	// 128 bit puts this value in both upper and lower 64 bit words
	ld	temp0, 64(a0)  
	fmv.d.x	f28, temp0
	
	// Load scale mask
	// 128 bit puts this value in both lower and upper 64 bit words.
	li	temp0, 0x80f0000000000000
	fmv.d.x	f30, temp0

	// Init rounding mode 0 (RandomX and RISC-V use differnt FRM codes!  0 is "Round to Nearest, ties to Even" for both)
	li temp0, 0
	fsrm temp0, temp0  //pseudoinstr swaps new/old fcsr values
	//TODO: save and restore old fsrm value??

	// Read IMUL_RCP literals
	// This nukes s0-s1, a0-a7 registers and tp (thread pointer) register
	ld	x4, literal_x4
	ld	x8, literal_x8
	ld	x3, literal_x3
	ld	x10, literal_x10
	ld	x11, literal_x11
	ld	x12, literal_x12
	ld	x13, literal_x13
	ld	x14, literal_x14
	ld	x15, literal_x15
	ld	x16, literal_x16
	ld	x17, literal_x17

/* fixme: need vector ld instr's - Vector Extension not yet published 
	// Read misc instr literals (not used yet; could use v16-v31 too)
	ld	v0, literal_v0
	ld	v1, literal_v1
	ld	v2, literal_v2
	ld	v3, literal_v3
	ld	v4, literal_v4
	ld	v5, literal_v5
	ld	v6, literal_v6
	ld	v7, literal_v7
	ld	v8, literal_v8
	ld	v9, literal_v9
	ld	v10, literal_v10
	ld	v11, literal_v11
	ld	v12, literal_v12
	ld	v13, literal_v13
	ld	v14, literal_v14
	ld	v15, literal_v15
*/
DECL(randomx_program_rv64_main_loop):

	srli	temp0, spMix1, 32

	//PLACEHOLDERS - instructions with actual mask will be inserted by JIT compiler
	// spAddr0 = spMix1 & ScratchpadL3Mask64;
	// spAddr1 = (spMix1 >> 32) & ScratchpadL3Mask64;
	andi	spAddr0, spMix1, 1 
	andi	spAddr1, temp0, 1 
	andi	spAddr0, spMix1, 1 
	andi	spAddr1, temp0, 1 
		
	//add spAddr offsets + base scratchpad ptr
	add	spAddr0, spAddr0, spPtr  // spAddr0 = scratchpad + spAddr0
	add	spAddr1, spAddr1, spPtr  // spAddr1 = scratchpad + spAddr1

	// 4.6.2 step 2------------------------------------
	// xor integer registers with L3 scratchpad data (from spAddr0)
	ld temp0, 0(spAddr0)
	ld temp1, 8(spAddr0)
	xor	r0, r0, temp0
	xor	r1, r1, temp1
	ld temp0, 16(spAddr0)
	ld temp1, 24(spAddr0)
	xor	r2, r2, temp0
	xor	r3, r3, temp1
	ld temp0, 32(spAddr0)
	ld temp1, 40(spAddr0)
	xor	r4, r4, temp0
	xor	r5, r5, temp1
	ld temp0, 48(spAddr0)
	ld temp1, 56(spAddr0)
	xor	r6, r6, temp0
	xor	r7, r7, temp1
	
	// 4.6.2 step 3------------------------------------
	// Load group F registers with L3 scratchpad data (from spAddr1)
	// NOTE: Loads 32 bit values from spAddr1 sign extends them
	// See ARM code.. 
	// What you get in each fp is 0xYYYYYYYYXXXXXXXX Y = sign extended bits X = Data from spAddr1
	// f0.0 f0.1
	lwu temp0, 0x0(spAddr1)
	fcvt.d.l f0, temp0
	lwu temp0, 0x4(spAddr1)
	fcvt.d.l f1, temp0
	// f1.0 f1.1
	lwu temp0, 0x8(spAddr1)
	fcvt.d.l f2, temp0
	lwu temp0, 0xC(spAddr1)
	fcvt.d.l f3, temp0
	// f2.0 f2.1
	lwu temp0, 0x10(spAddr1)
	fcvt.d.l f4, temp0
	lwu temp0, 0x14(spAddr1)
	fcvt.d.l f5, temp0
	// f3.0 f3.1
	lwu temp0, 0x18(spAddr1)
	fcvt.d.l f6, temp0
	lwu temp0, 0x1C(spAddr1)
	fcvt.d.l f7, temp0

	// Load group E registers (from spAddr1+32
	// e0.0 e0.1
	lwu temp0, 0x20(spAddr1)
	fcvt.d.l f8, temp0
	lwu temp0, 0x24(spAddr1)
	fcvt.d.l f9, temp0
	// e1.0 e1.1
	lwu temp0, 0x28(spAddr1)
	fcvt.d.l f10, temp0
	lwu temp0, 0x2C(spAddr1)
	fcvt.d.l f11, temp0
	// e2.0 e2.1
	lwu temp0, 0x30(spAddr1)
	fcvt.d.l f12, temp0
	lwu temp0, 0x34(spAddr1)
	fcvt.d.l f13, temp0
	// e3.0 e3.1
	lwu temp0, 0x38(spAddr1)
	fcvt.d.l f14, temp0
	lwu temp0, 0x3C(spAddr1)
	fcvt.d.l f15, temp0

	// mask E registers to confine range  (using macros; may add f logic instrs later)
	fandd f8, f8, f26, temp0, temp1
	fandd f9, f9, f26, temp0, temp1
	fandd f10, f10, f26, temp0, temp1
	fandd f11, f11, f26, temp0, temp1
	fandd f12, f12, f26, temp0, temp1
	fandd f13, f13, f26, temp0, temp1
	fandd f14, f14, f26, temp0, temp1
	fandd f15, f15, f26, temp0, temp1
	forrd f8, f8, f28, temp0, temp1
	forrd f9, f9, f28, temp0, temp1
	forrd f10, f10, f28, temp0, temp1
	forrd f11, f11, f28, temp0, temp1
	forrd f12, f12, f28, temp0, temp1
	forrd f13, f13, f28, temp0, temp1	
	forrd f14, f14, f28, temp0, temp1
	forrd f15, f15, f28, temp0, temp1
	
	// 4.6.2 step 4------------------------------------
	// Execute random VM instructions
DECL(randomx_program_rv64_vm_instructions):
	.fill RANDOMX_PROGRAM_SIZE*12,4,0  	// buffer for generated instructions

// IMUL_RCP reg literal pool
literal_x4: .fill 1,8,0
literal_x8: .fill 1,8,0
literal_x3: .fill 1,8,0
literal_x10: .fill 1,8,0
literal_x11: .fill 1,8,0
literal_x12: .fill 1,8,0
literal_x13: .fill 1,8,0
literal_x14: .fill 1,8,0
literal_x15: .fill 1,8,0
literal_x16: .fill 1,8,0
literal_x17: .fill 1,8,0
DECL(randomx_program_rv64_imul_rcp_literals_end):

// NOT actually using this literal pool for now - using LUI/ADDI (will be slower)
literal_v0:  .fill 1,8,0
literal_v1:  .fill 1,8,0
literal_v2:  .fill 1,8,0
literal_v3:  .fill 1,8,0
literal_v4:  .fill 1,8,0
literal_v5:  .fill 1,8,0
literal_v6:  .fill 1,8,0
literal_v7:  .fill 1,8,0
literal_v8:  .fill 1,8,0
literal_v9:  .fill 1,8,0
literal_v10: .fill 1,8,0
literal_v11: .fill 1,8,0
literal_v12: .fill 1,8,0
literal_v13: .fill 1,8,0
literal_v14: .fill 1,8,0
literal_v15: .fill 1,8,0
literal_v16: .fill 1,8,0
literal_v17: .fill 1,8,0
literal_v18: .fill 1,8,0
literal_v19: .fill 1,8,0
literal_v20: .fill 1,8,0
literal_v21: .fill 1,8,0
literal_v22: .fill 1,8,0
literal_v23: .fill 1,8,0
literal_v24: .fill 1,8,0
literal_v25: .fill 1,8,0
literal_v26: .fill 1,8,0
literal_v27: .fill 1,8,0
literal_v28: .fill 1,8,0
literal_v29: .fill 1,8,0
literal_v30: .fill 1,8,0
literal_v31: .fill 1,8,0



DECL(randomx_program_rv64_vm_instructions_end):	
//--4.6.2 step 5------------------------------------
	// mx ^= r[readReg2] ^ r[readReg3];
	// temp0 <= r[readReg2] ^ r[readReg3] set up by JIT compiler
	li temp1, 0xFFFFFFFF
	and temp0, temp0, temp1 //to not clobber MA
	xor	mamx, mamx, temp0 

	
//--4.6.2 step 6------------------------------------
	// Calculate dataset pointer for dataset prefetch
DECL(randomx_program_rv64_cacheline_align_mask1):	//tag for code insertion
//	andi temp0, mamx, 1  // PLACEHOLDER - Actual mask will be inserted by JIT compiler
//	andi temp0, mamx, 1
//	andi temp0, mamx, 1
//	add	temp0, temp0, dsPtr

// Prefetch dataset data
// prfm	pldl2strm, [temp0]  //TODO: just ld to a nearby buffer
	
//--4.6.2 step 7------------------------------------
	// Calculate dataset pointer for dataset read
DECL(randomx_program_rv64_cacheline_align_mask2):	//tag for code insertion
	andi spMix1, mamx, 1 // PLACEHOLDER: Actual mask will be inserted by JIT compiler
	andi spMix1, mamx, 1
	andi spMix1, mamx, 1
	andi spMix1, mamx, 1
	li spMix1, 0x10334
	li spMix1, 0x2adfd
	add	spMix1, spMix1, dsPtr

DECL(randomx_program_rv64_xor_with_dataset_line):
	// xor integer registers with dataset data
	ld temp0,  0(spMix1)
	ld temp1,  8(spMix1)
	xor	r0, r0, temp0
	xor	r1, r1, temp1
	ld temp0, 16(spMix1)
	ld temp1, 24(spMix1)
	xor	r2, r2, temp0
	xor	r3, r3, temp1
	ld temp0, 32(spMix1)
	ld temp1, 40(spMix1)
	xor	r4, r4, temp0
	xor	r5, r5, temp1
	ld temp0, 48(spMix1)
	ld temp1, 56(spMix1)
	xor	r6, r6, temp0
	xor	r7, r7, temp1
	
DECL(randomx_program_rv64_update_spMix1):
	// JIT compiler will replace with "xor spMix1, config.readReg0, config.readReg1"
	xor	spMix1, zero, zero

//--4.6.2 step 8------------------------------------
	// swap mx <-> ma   //ror	mamx, mamx, 32
	slli temp0, mamx, 32
	srli temp1, mamx, 32
	or mamx, temp0, temp1	

	
//--4.6.2 step 9------------------------------------
	// Store integer registers to scratchpad (spAddr1)
	sd r0,   0(spAddr1)
	sd r1,   8(spAddr1)
	sd r2,  16(spAddr1)
	sd r3,  24(spAddr1)
	sd r4, 32(spAddr1)
	sd r5, 40(spAddr1)
	sd r6, 48(spAddr1)
	sd r7, 56(spAddr1)
	
//--4.6.2 step 10------------------------------------
	// xor group F and group E registers
	fxord f0, f0, f8, temp0,  temp1
	fxord f1, f1, f9, temp0,  temp1
	fxord f2, f2, f10, temp0,  temp1
	fxord f3, f3, f11, temp0,  temp1
	fxord f4, f4, f12, temp0,  temp1
	fxord f5, f5, f13, temp0,  temp1
	fxord f6, f6, f14, temp0,  temp1
	fxord f7, f7, f15, temp0,  temp1
	
//--4.6.2 step 11------------------------------------
	// Store FP (f=f^e) registers to scratchpad (spAddr0)
	fsd f0,    0(spAddr0)
	fsd f1,    8(spAddr0)
	fsd f2,   16(spAddr0)
	fsd f3,   24(spAddr0)
	fsd f4,   32(spAddr0)
	fsd f5,   40(spAddr0)
	fsd f6,   48(spAddr0)
	fsd f7,   56(spAddr0)
	
//--4.6.2 step 12------------------------------------
	addi loopCnt, loopCnt, -1
	bnez loopCnt, DECL(randomx_program_rv64_main_loop)

	// DONE WITH MAIN LOOPS
	// Store integer registers
	sd r0,  0(regPtr)
	sd r1,  8(regPtr)
	sd r2, 16(regPtr)
	sd r3, 24(regPtr)
	sd r4, 32(regPtr)
	sd r5, 40(regPtr)
	sd r6, 48(regPtr)
	sd r7, 56(regPtr)

	// Store FP registers (f, e, not a)
	fsd f0,   64(regPtr)
	fsd f1,   72(regPtr)
	fsd f2,   80(regPtr)
	fsd f3,   88(regPtr)
	fsd f4,   96(regPtr)
	fsd f5,  104(regPtr)
	fsd f6,  112(regPtr)
	fsd f7,  120(regPtr)
	fsd f8,  128(regPtr)
	fsd f9,  136(regPtr)
	fsd f10, 144(regPtr)
	fsd f11, 152(regPtr)
	fsd f12, 160(regPtr)
	fsd f13, 168(regPtr)
	fsd f14, 176(regPtr)
	fsd f15, 184(regPtr)
		
	// Restore callee-saved registers (S0-S11)  //todo: also fs0-11, if clobber matters.
	ld s0,  0(sp)
	ld s1,  8(sp)
	ld s2, 16(sp)
	ld s3, 24(sp)
	ld s4, 32(sp)
	ld s5, 40(sp)
	ld s6, 48(sp)
	ld s7, 56(sp)
	ld s8, 64(sp)
	ld s9, 72(sp)
	ld s10, 80(sp)
	ld s11, 88(sp)
	ld gp, 96(sp)
	ld tp, 104(sp)
	addi sp, sp, 192

	ret

DECL(randomx_program_rv64_vm_instructions_end_light):
	addi sp, sp, -96
	sd a0, 64(sp)
	sd a1, 72(sp)
	sd a2, 80(sp)
	// Save ra so we can do the call below
	sd ra, 88(sp)

	// mx ^= r[readReg2] ^ r[readReg3];
	xor	mamx, mamx, temp0

	// swap mx <-> ma //ror	mamx, mamx, 32
	slli temp0, mamx, 32
	srli temp1, mamx, 32
	or mamx, temp0, temp1	

	// a0 -> pointer to cache memory
	mv	a0, dsPtr

	// a1 -> pointer to output
	mv	a1, sp

DECL(randomx_program_rv64_light_cacheline_align_mask):
	// Actual mask will be inserted by JIT compiler with 3 opcodes
	// PLACEHOLDERS
	andi	spPtr, x9, 1
	andi	spPtr, x9, 1
	andi	spPtr, x9, 1

	// x2 -> item number
	srli	a2, a2, 6

DECL(randomx_program_rv64_light_dataset_offset):
	// Apply dataset offset (filled in by JIT compiler)
	// PLACEHOLDERS
	addi	spPtr, spPtr, 0
	addi	spPtr, spPtr, 0
	addi	spPtr, spPtr, 0

	call	DECL(randomx_calc_dataset_item_rv64)

	mv	spMix1, sp
	ld a0, 64(sp)
	ld a1, 72(sp)
	ld a2, 80(sp)
	ld ra, 88(sp)
	add	sp, sp, 96

	j	DECL(randomx_program_rv64_xor_with_dataset_line)

//--DATASET INITALIZATION-----------------------------------------

// Input parameters

// a0 -> pointer (to ptr?) to cache
// a1 -> pointer to dataset memory at startItem
// a2 -> start item
// a3 -> end item

DECL(randomx_init_dataset_rv64):
	// Save return address
	// Save all S registers
	addi sp, sp, -104
	sd s0,  0(sp)
	sd s1,  8(sp)
	sd s2, 16(sp)
	sd s3, 24(sp)
	sd s4, 32(sp)
	sd s5, 40(sp)
	sd s6, 48(sp)
	sd s7, 56(sp)
	sd s8, 64(sp)
	sd s9, 72(sp)
	sd s10, 80(sp)
	sd s11, 88(sp)
	sd ra, 96(sp)

	// Load pointer to cache memory
	ld	a0, 0(a0)

DECL(randomx_init_dataset_rv64_main_loop):
	call	DECL(randomx_calc_dataset_item_rv64)
	add	a1, a1, 64
	add	a2, a2, 1
	bne	a2, a3, DECL(randomx_init_dataset_rv64_main_loop)

	#restore S regs and ra regs
	ld s0,  0(sp)
	ld s1,  8(sp)
	ld s2, 16(sp)
	ld s3, 24(sp)
	ld s4, 32(sp)
	ld s5, 40(sp)
	ld s6, 48(sp)
	ld s7, 56(sp)
	ld s8, 64(sp)
	ld s9, 72(sp)
	ld s10, 80(sp)
	ld s11, 88(sp)
	ld ra, 96(sp)
	addi sp, sp, 104

	ret

DECL(randomx_init_dataset_rv64_end):


//--DATASET CALCULATION------------------------------------
// Input parameters

// a0 -> pointer to cache memory
// a1 -> pointer to output
// a2 -> item number

// Register allocation

// a0-a7 -> output value (calculated dataset item)
// s0 -> pointer to cache memory
// s1 -> pointer to output
// s2 -> registerValue
// s3 -> mixBlock
// t0 -> temporary
// t1 -> temporary

DECL(randomx_calc_dataset_item_rv64):
	#save regs
	addi sp, sp, -112
	sd s0,  0(sp)
	sd s1,  8(sp)
	sd s2, 16(sp)
	sd s3, 24(sp)
	sd a0, 32(sp)
	sd a1, 40(sp)
	sd a2, 48(sp)
	sd a3, 56(sp)
	sd a4, 64(sp)
	sd a5, 72(sp)
	sd a6, 80(sp)
	sd a7, 88(sp)
	sd t1, 96(sp)
	sd t0, 104(sp)

	#save args
	mv	s0, a0
	mv	s1, a1
	mv	s2, a2

	ld	t0, superscalarMul0

	// rl[0] = (itemNumber + 1) * superscalarMul0;ldpsw
	// madd	a0, a2, t0, t0
	addi a2, a2, 1
	mul a0, a2, t0

	// rl[1] = rl[0] ^ superscalarAdd1;
	ld	t0, superscalarAdd1
	xor	a1, a0, t0

	// rl[2] = rl[0] ^ superscalarAdd2;
	ld	t0, superscalarAdd2
	xor	a2, a0, t0

	// rl[3] = rl[0] ^ superscalarAdd3;
	ld	t0, superscalarAdd3
	xor	a3, a0, t0

	// rl[4] = rl[0] ^ superscalarAdd4;
	ld	t0, superscalarAdd4
	xor	a4, a0, t0

	// rl[5] = rl[0] ^ superscalarAdd5;
	ld	t0, superscalarAdd5
	xor	a5, a0, t0

	// rl[6] = rl[0] ^ superscalarAdd6;
	ld	t0, superscalarAdd6
	xor	a6, a0, t0

	// rl[7] = rl[0] ^ superscalarAdd7;
	ld	t0, superscalarAdd7
	xor	a7, a0, t0

	j	DECL(randomx_calc_dataset_item_rv64_prefetch) #hop over superscalarMul data

superscalarMul0: .quad 6364136223846793005
superscalarAdd1: .quad 9298411001130361340
superscalarAdd2: .quad 12065312585734608966
superscalarAdd3: .quad 9306329213124626780
superscalarAdd4: .quad 5281919268842080866
superscalarAdd5: .quad 10536153434571861004
superscalarAdd6: .quad 3398623926847679864
superscalarAdd7: .quad 9549104520008361294

// Prefetch -> SuperScalar hash -> Mix will be repeated N times

DECL(randomx_calc_dataset_item_rv64_prefetch):
	// Actual mask will be inserted by JIT compiler (3 opcodes) 
	and	s3, s2, 1
	and	s3, s2, 1
	and	s3, s2, 1
	#add	s3, s0, s3, lsl 6  #fused sl-add  (rs2<<6)
	slli s3, s3, 6
	add s3, s3, s0
	

	// Prefetch dataset data
	// prfm	pldl2strm, [s3]

// Generated SuperScalar hash program goes here

DECL(randomx_calc_dataset_item_rv64_mix):
	ld t0,  0(s3)
	ld t1,  8(s3)
	xor	a0, a0, t0
	xor	a1, a1, t1
	ld t0, 16(s3)
	ld t1, 24(s3)
	xor	a2, a2, t0
	xor	a3, a3, t1
	ld t0, 32(s3)
	ld t1, 40(s3)
	xor	a4, a4, t0
	xor	a5, a5, t1
	ld t0, 48(s3)
	ld t1, 56(s3)
	xor	a6, a6, t0
	xor	a7, a7, t1

DECL(randomx_calc_dataset_item_rv64_store_result):
	sd a0,  0(s1)
	sd a1,  8(s1)
	sd a2, 16(s1)
	sd a3, 24(s1)
	sd a4, 32(s1)
	sd a5, 40(s1)
	sd a6, 48(s1)
	sd a7, 56(s1)

	#restore callee-saved regs
	ld s0,  0(sp)
	ld s1,  8(sp)
	ld s2, 16(sp)
	ld s3, 24(sp)
	ld a0, 32(sp)
	ld a1, 40(sp)
	ld a2, 48(sp)
	ld a3, 56(sp)
	ld a4, 64(sp)
	ld a5, 72(sp)
	ld a6, 80(sp)
	ld a7, 88(sp)
	ld t1, 96(sp)
	ld t0, 104(sp)
	addi sp, sp, 112

	ret

DECL(randomx_calc_dataset_item_rv64_end):

.option pop
