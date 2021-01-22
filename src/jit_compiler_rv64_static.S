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
#define dsPtr x1 // x1 -> pointer to mem buffer and then to dataset    //FIXME: RA
#define spPtr x2 // x2 -> pointer to scratchpad                  //FIXME: SP
#define loopCnt x3 // x3    -> loop counter                           //FIXME: need GP?
#define r0 x4   // x4  -> "r0"
#define r1 x5   // x5  -> "r1"
#define r2 x6   // x6  -> "r2"
#define r3 x7   // x7  -> "r3"
//X8 free   //#define fprc x8 // x8  -> fprc 
#define mamx x9 // x9  -> mx, ma
#define spMix1 x10   // x10 -> spMix1
#define regPtr x11 // x11 -> pointer to reg buffer (OUTPUT only)  //was in x0; x11 was a literal for IMUL_RCP
#define r4 x12 // x12 -> "r4"
#define r5 x13 // x13 -> "r5"
#define r6 x14 // x14 -> "r6"
#define r7 x15 // x15 -> "r7"
#define spAddr0 x16 // x16 -> spAddr0
#define spAddr1 x17 // x17 -> spAddr1
#define temp0 x18 // x18 -> temporary
#define temp1 x19 // x19 -> temporary
// x20 -> literal for IMUL_RCP
// x21 -> literal for IMUL_RCP
// x22 -> literal for IMUL_RCP
// x23 -> literal for IMUL_RCP
// x24 -> literal for IMUL_RCP
// x25 -> literal for IMUL_RCP
// x26 -> literal for IMUL_RCP
// x27 -> literal for IMUL_RCP
// x28 -> literal for IMUL_RCP
// x29 -> literal for IMUL_RCP
// x30 -> literal for IMUL_RCP

#use F regs, not V for now; lower double in fN, upper in fN+1
// f0 -> "f0"      
// f2 -> "f1"
// f4 -> "f2"
// f6 -> "f3"
// f8 -> "e0"
// f10 -> "e1"
// f12 -> "e2"
// f14 -> "e3"
// f16 -> "a0"
// f18 -> "a1"
// f20 -> "a2"
// f22 -> "a3"
// f24 -> temporary
// f26 -> E 'and' mask = 0x00ffffffffffffff
// f28 -> E 'or' mask  = 0x3*00000000******
// f30 -> scale mask   = 0x81f0000000000000

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
	// Save callee-saved registers (S0-S11)
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
	mv spAddr0, mamx

	//init spaddr1 = ma
	srli spAddr1, mamx, 32

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
	fld f16, 192(a0)  //offsets seem off vs spec... check c code
	fld f17, 200(a0)
	fld f18, 208(a0)
	fld f19, 216(a0)
	fld f20, 224(a0)
	fld f21, 232(a0)
	fld f22, 240(a0)
	fld f23, 248(a0)

	// Load E 'and' mask
	li	spAddr0, 0x00FFFFFFFFFFFFFF
	fmv.d.x	f26, spAddr0
	fmv.d.x	f27, spAddr0

	// Load E 'or' mask (stored in reg.f[0])
	ld	spAddr0, 64(a0)  
	fmv.d.x	f28, spAddr0
	fmv.d.x	f29, spAddr0

	// Load scale mask
	li	spAddr0, 0x80f0000000000000
	fmv.d.x	f30, spAddr0
	fmv.d.x	f31, spAddr0

	// Init rounding mode 0 (RandomX and RISC-V use differnt FRM codes!  0 is "Round to Nearest, ties to Even" for both)
	li temp0, 0
	fsrm temp0, temp0  //pseudoinstr swaps new/old fcsr values
	//TODO: save and restore old fsrm value??

	// Read IMUL_RCP literals 
	ld	x20, literal_x20
	ld	x21, literal_x21
	ld	x22, literal_x22
	ld	x23, literal_x23
	ld	x24, literal_x24
	ld	x25, literal_x25
	ld	x26, literal_x26
	ld	x27, literal_x27
	ld	x28, literal_x28
	ld	x29, literal_x29
	ld	x30, literal_x30


/* fixme: need vector ld instr's - not in toolchain?
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
	// spAddr0 = spMix1 & ScratchpadL3Mask64;
	// spAddr1 = (spMix1 >> 32) & ScratchpadL3Mask64;
	srli	temp0, spMix1, 32

	//PLACEHOLDERS - instructions with actual mask will be inserted by JIT compiler 
	andi	spAddr0, spMix1, 1  //aarch64 only masked lower 32b
	andi	spAddr1, temp0, 1 //aarch64 only masked lower 32b
		
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
	// Load group F registers with L3 scratchpad data (from spAddr1) (32b words to doubles)
	lw temp0, 0(spAddr1)
	lw temp1, 4(spAddr1)
	fcvt.d.w f0, temp0
	fcvt.d.w f1, temp1
	lw temp0, 8(spAddr1)
	lw temp1, 12(spAddr1)
	fcvt.d.w f2, temp0
	fcvt.d.w f3, temp1
	lw temp0, 16(spAddr1)
	lw temp1, 20(spAddr1)
	fcvt.d.w f4, temp0
	fcvt.d.w f5, temp1
	lw temp0, 24(spAddr1)
	lw temp1, 28(spAddr1)
	fcvt.d.w f6, temp0
	fcvt.d.w f7, temp1

	// Load group E registers (from spAddr1+32) (32b words to doubles)  
	lw temp0, 32(spAddr1)
	lw temp1, 36(spAddr1)
	fcvt.d.w f8, temp0
	fcvt.d.w f9, temp1
	lw temp0, 40(spAddr1)
	lw temp1, 44(spAddr1)
	fcvt.d.w f10, temp0
	fcvt.d.w f11, temp1
	lw temp0, 48(spAddr1)
	lw temp1, 52(spAddr1)
	fcvt.d.w f12, temp0
	fcvt.d.w f13, temp1
	lw temp0, 56(spAddr1)
	lw temp1, 60(spAddr1)
	fcvt.d.w f14, temp0
	fcvt.d.w f15, temp1
	// mask E registers to confine range  (using macros; may add f logic instrs later)
	fandd f8, f8, f26, temp0, temp1
	forrd f8, f8, f28, temp0, temp1
	fandd f9, f9, f26, temp0, temp1
	forrd f9, f9, f28, temp0, temp1
	fandd f10, f10, f26, temp0, temp1
	forrd f10, f10, f28, temp0, temp1
	fandd f11, f11, f26, temp0, temp1
	forrd f11, f11, f28, temp0, temp1
	fandd f12, f12, f26, temp0, temp1
	forrd f12, f12, f28, temp0, temp1
	fandd f13, f13, f26, temp0, temp1
	forrd f13, f13, f28, temp0, temp1
	fandd f14, f14, f26, temp0, temp1
	forrd f14, f14, f28, temp0, temp1
	fandd f15, f15, f26, temp0, temp1
	forrd f15, f15, f28, temp0, temp1

	
	// 4.6.2 step 4------------------------------------
	// Execute random VM instructions
DECL(randomx_program_rv64_vm_instructions):
	.fill RANDOMX_PROGRAM_SIZE*12,4,0  	// buffer for generated instructions // FDIV_M is the largest instruction taking up to 12 ARMv8 instructions


// IMUL_RCP reg literal pool
literal_x20: .fill 1,8,0
literal_x21: .fill 1,8,0
literal_x22: .fill 1,8,0
literal_x23: .fill 1,8,0
literal_x24: .fill 1,8,0
literal_x25: .fill 1,8,0
literal_x26: .fill 1,8,0
literal_x27: .fill 1,8,0
literal_x28: .fill 1,8,0
literal_x29: .fill 1,8,0
literal_x30: .fill 1,8,0
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
	andi temp0, mamx, 1  // PLACEHOLDER - Actual mask will be inserted by JIT compiler
	add	temp0, temp0, dsPtr

	// Prefetch dataset data
	// prfm	pldl2strm, [temp0]  //TODO: just ld to a nearby buffer
	
//--4.6.2 step 7------------------------------------
	// Calculate dataset pointer for dataset read
DECL(randomx_program_rv64_cacheline_align_mask2):	//tag for code insertion
	andi spMix1, mamx, 1 // PLACEHOLDER: Actual mask will be inserted by JIT compiler
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
	addi sp, sp, 192

	ret

DECL(randomx_program_rv64_vm_instructions_end_light):
	addi sp, sp, -96
	sd x0, 64(sp)
	sd x1, 72(sp)
	sd x2, 80(sp)
	sd x30, 88(sp)

	// mx ^= r[readReg2] ^ r[readReg3];
	xor	mamx, mamx, temp0

	// swap mx <-> ma //ror	mamx, mamx, 32
	slli temp0, mamx, 32
	srli temp1, mamx, 32
	or mamx, temp0, temp1	

	// x0 -> pointer to cache memory
	mv	x0, x1

	// x1 -> pointer to output
	mv	x1, sp

DECL(randomx_program_rv64_light_cacheline_align_mask):
	// Actual mask will be inserted by JIT compiler
	andi	x2, x9, 1

	// x2 -> item number
	srli	x2, x2, 6

DECL(randomx_program_rv64_light_dataset_offset):
	// Apply dataset offset (filled in by JIT compiler)
	addi	x2, x2, 0
	addi	x2, x2, 0

	call	DECL(randomx_calc_dataset_item_rv64)

	mv	spMix1, sp
	ld x0, 64(sp)
	ld x1, 72(sp)
	ld x2, 80(sp)
	ld x30, 88(sp)
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
	sd	ra, -16(sp)

	// Load pointer to cache memory
	ld	a0, 0(a0)

DECL(randomx_init_dataset_rv64_main_loop):
	call	DECL(randomx_calc_dataset_item_rv64)
	add	a1, a1, 64
	add	a2, a2, 1
	bne	a2, a3, DECL(randomx_init_dataset_rv64_main_loop)

	// Restore return address
	ld	ra, 16(sp)

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
	#save regs s0-s11
	addi sp, sp, -96
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

	#save args
	mv	s0, a0
	mv	s1, a1
	mv	s2, a2


	ld	t0, superscalarMul0

	// rl[0] = (itemNumber + 1) * superscalarMul0;
	// madd	a0, a2, t0, t0
	mul a0, a2, t0
	add a0, a0, t0

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
	// Actual mask will be inserted by JIT compiler
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
	ld s4, 32(sp)
	ld s5, 40(sp)
	ld s6, 48(sp)
	ld s7, 56(sp)
	ld s8, 64(sp)
	ld s9, 72(sp)
	ld s10, 80(sp)
	ld s11, 88(sp)
	addi sp, sp, 96

	ret

DECL(randomx_calc_dataset_item_rv64_end):