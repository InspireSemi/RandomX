/*
Copyright (c) 2018-2019, tevador <tevador@gmail.com>
Copyright (c) 2019, SChernykh    <https://github.com/SChernykh>

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
	* Redistributions of source code must retain the above copyright
	  notice, this list of conditions and the following disclaimer.
	* Redistributions in binary form must reproduce the above copyright
	  notice, this list of conditions and the following disclaimer in the
	  documentation and/or other materials provided with the distribution.
	* Neither the name of the copyright holder nor the
	  names of its contributors may be used to endorse or promote products
	  derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "jit_compiler_rv64.hpp"
#include "superscalar.hpp"
#include "program.hpp"
#include "reciprocal.h"
#include "virtual_memory.hpp"


#define LUI(rd, imm)      imm<<12 | rd<<7 | 0b0110111
#define AUIPC(rd, imm)    imm<<12 | rd<<7 | 0b0010111

#define JAL_BIT20 				0x100000
#define JAL_BIT20_SHFT			20
#define JAL_BIT20_OPCODE_SHFT	31

#define JAL_BITS10_1 				0x7FE
#define JAL_BITS10_1_SHFT			1
#define JAL_BITS10_1_OPCODE_SHFT	21

#define JAL_BIT11				0x800
#define JAL_BIT11_SHFT			11
#define JAL_BIT11_OPCODE_SHFT	20

#define JAL_BITS19_12				0xFF000
#define JAL_BITS12_12_SHFT			12
#define JAL_BITS19_12_OPCODEE_SHFT	12

#define JAL(rd, imm) ( (imm & JAL_BIT20) >> JAL_BIT20_SHFT) <<  JAL_BIT20_OPCODE_SHFT | \
					( (imm & JAL_BITS10_1) >> JAL_BITS10_1_SHFT) << JAL_BITS10_1_OPCODE_SHFT | \
					( (imm & JAL_BIT11) >> JAL_BIT11_SHFT) << JAL_BIT11_OPCODE_SHFT | \
					( (imm & JAL_BITS19_12) >> JAL_BITS12_12_SHFT) << JAL_BITS19_12_OPCODEE_SHFT |\
					rd<<7 | 0b1101111

//#define BLT(rs1,rs2, imm) imm<<25 | rs2<<20 | rs1<<15 | 0b100<<12 | 0b1100011 //FIXME: slice and dice imm

#define ADD(rd,rs1,rs2)   0b0000000<<25 | rs2<<20 | rs1<<15 | 0b000<<12 | rd<<7 | 0b0110011
#define SUB(rd,rs1,rs2)   0b0100000<<25 | rs2<<20 | rs1<<15 | 0b000<<12 | rd<<7 | 0b0110011
#define SRL(rd,rs1,rs2)   0b0000000<<25 | rs2<<20 | rs1<<15 | 0b101<<12 | rd<<7 | 0b0110011
#define SLL(rd, rs1,rs2)  0b0000000<<25 | rs2<<20 | rs1<<15 | 0b001<<12 | rd<<7 | 0b0110011
#define AND(rd,rs1,rs2)   0b0000000<<25 | rs2<<20 | rs1<<15 | 0b111<<12 | rd<<7 | 0b0110011
#define XOR(rd,rs1,rs2)   0b0000000<<25 | rs2<<20 | rs1<<15 | 0b100<<12 | rd<<7 | 0b0110011
#define MUL(rd,rs1,rs2)   0b0000001<<25 | rs2<<20 | rs1<<15 | 0b000<<12 | rd<<7 | 0b0110011
#define MULHU(rd,rs1,rs2) 0b0000001<<25 | rs2<<20 | rs1<<15 | 0b011<<12 | rd<<7 | 0b0110011
#define MULHS(rd,rs1,rs2) 0b0000001<<25 | rs2<<20 | rs1<<15 | 0b001<<12 | rd<<7 | 0b0110011
#define OR(rd,rs1,rs2)    0b0000000<<25 | rs2<<20 | rs1<<15 | 0b111<<12 | rd<<7 | 0b0110011
#define ORI(rd, rs1, imm) imm<<20 | rs1<<15 | 0b110<<12 | rd<<7 | 0b0010011

#define CSRRW(rd,csr,rs1)                 csr<<20 | rs1<<15 | 0b001<<12 | rd<<7 | 0b1110011
#define FCSR 0x3

#define ADDI(rd,rs1,imm)  imm<<20 | rs1<<15 | 0b000<<12 | rd<<7 | 0b0010011
#define ADDIW(rd,rs1,imm)  imm<<20 | rs1<<15 | 0b000<<12 | rd<<7 | 0b0011011
#define ANDI(rd,rs1,imm)  imm<<20 | rs1<<15 | 0b111<<12 | rd<<7 | 0b0010011
#define SLLI(rd,rs1,imm)  0b000000<<26 | imm<<20 | rs1<<15 | 0b001<<12 | rd<<7 | 0b0010011
#define SRLI(rd,rs1,imm)  0b000000<<26 | imm<<20 | rs1<<15 | 0b101<<12 | rd<<7 | 0b0010011


#define BEQ_BIT12  				0x1000
#define BEQ_BIT12_SHFT			12
#define BEQ_BIT12_OPCODE_SHFT	31

#define BEQ_BIT11  				0x0800
#define BEQ_BIT11_SHFT			11
#define BEQ_BIT11_OPCODE_SHFT	7

#define BEQ_BITS10_5 				0x07E0
#define BEQ_BITS10_5_SHFT			5
#define BEQ_BITS10_5_OPCODE_SHFT	25

#define BEQ_BITS4_1					0x001E
#define BEQ_BITS4_1_SHFT			1
#define BEQ_BITS4_1_OPCODDE_SHFT	8

#define BEQ(offset, rs1, rs2) ( ((offset & BEQ_BIT12) >> BEQ_BIT12_SHFT) << BEQ_BIT12_OPCODE_SHFT ) | \
	( ((offset & BEQ_BIT11) >> BEQ_BIT11_SHFT) << BEQ_BIT11_OPCODE_SHFT ) | \
	( ((offset & BEQ_BITS10_5 ) >> BEQ_BITS10_5_SHFT) << BEQ_BITS10_5_OPCODE_SHFT ) | \
	( ((offset & BEQ_BITS4_1) >> BEQ_BITS4_1_SHFT) << BEQ_BITS4_1_OPCODDE_SHFT ) | \
	(rs1 << 15) | (rs2 << 20) | (0b000 << 12) | (0b1100011)

#define LD(rd,rs1,imm)    imm<<20           | rs1<<15 | 0b011<<12 | rd<<7  | 0b0000011
#define LW(rd,rs1,imm)    imm<<20           | rs1<<15 | 0b010<<12 | rd<<7  | 0b0000011
#define LWU(rd,rs1,imm)    imm<<20           | rs1<<15 | 0b110<<12 | rd<<7  | 0b0000011
#define SD(rs2,rs1,imm)   imm<<25 | rs2<<20 | rs1<<15 | 0b011<<12 | imm<<7 | 0b0100011  //fixme: slice imm


#define FADD_D(rd,rs1,rs2,rm) 0b0000001<<25 | rs2<<20 | rs1<<15 | rm<<12 | rd<<7 | 0b1010011
#define FSUB_D(rd,rs1,rs2,rm) 0b0000101<<25 | rs2<<20 | rs1<<15 | rm<<12 | rd<<7 | 0b1010011
#define FMUL_D(rd,rs1,rs2,rm) 0b0001001<<25 | rs2<<20 | rs1<<15 | rm<<12 | rd<<7 | 0b1010011
#define FDIV_D(rd,rs1,rs2,rm) 0b0001101<<25 | rs2<<20 | rs1<<15 | rm<<12 | rd<<7 | 0b1010011
#define FSQRT_D(rd,rs1,rm)    0b0101101<<25 | rs1<<15 |           rm<<12 | rd<<7 | 0b1010011
#define FCVTDW(rd,rs1,rm)     0b1101001<<25 | 0b0<<20 | rs1<<15 | rm<<12 | rd<<7 | 0b1010011
#define FSGNJ_D(rd,rs1,rs2)   0b0010001<<25 | rs2<<20 | rs1<<15 | 0b000<<12 | rd<<7 | 0b1010011

#define FMV_X_D(rd, rs1)	  0b1110001<<25 | 0b0000 << 20 | rs1 << 15 | 0b000 << 12 | rd << 7 | 0b1010011
#define FMV_D_X(rd, rs1)	  0b1111001<<25 | 0b0000 << 20 | rs1 << 15 | 0b000 << 12 | rd << 7 | 0b1010011


//custom F reg bitwise logic - pack with SGNJ, upper funct3 range.
#define FXOR_D(rd,rs1,rs2)    0b0010001<<25 | rs2<<20 | rs1<<15 | 0b101<<12 | rd<<7 | 0b1010011 //fixme: Add instr
#define FORR_D(rd,rs1,rs2)    0b0010001<<25 | rs2<<20 | rs1<<15 | 0b110<<12 | rd<<7 | 0b1010011 //fixme: Add instr
#define FAND_D(rd,rs1,rs2)    0b0010001<<25 | rs2<<20 | rs1<<15 | 0b111<<12 | rd<<7 | 0b1010011 //fixme: Add instr

#ifdef BITMANIP	
	//New instructions: FXOR_D, ROR, ROL, RORI
	#define ROR(rd,rs1,rs2)   0b0110000<<25 | rs2<<20 | rs1<<15 | 0b101<<12 | rd<<7 | 0b0110011
	#define ROL(rd,rs1,rs2)   0b0110000<<25 | rs2<<20 | rs1<<15 | 0b001<<12 | rd<<7 | 0b0110011
	#define RORI(rd,rs1,imm)  0b01100<<27 | imm<<20 | rs1<<15 | 0b101<<12 | rd<<7 | 0b0010011
#endif


//TODO: shift-add #define SLLI_ADDI(rd, rs1, slamt, imm)

namespace randomx {

static const size_t CodeSize = ((uint8_t*)randomx_init_dataset_rv64_end) - ((uint8_t*)randomx_program_rv64);
static const size_t MainLoopBegin = ((uint8_t*)randomx_program_rv64_main_loop) - ((uint8_t*)randomx_program_rv64);
static const size_t PrologueSize = ((uint8_t*)randomx_program_rv64_vm_instructions) - ((uint8_t*)randomx_program_rv64);
static const size_t ImulRcpLiteralsEnd = ((uint8_t*)randomx_program_rv64_imul_rcp_literals_end) - ((uint8_t*)randomx_program_rv64);

static const size_t CalcDatasetItemSize =
	// Prologue
	((uint8_t*)randomx_calc_dataset_item_rv64_prefetch - (uint8_t*)randomx_calc_dataset_item_rv64) +
	// Main loop
	RANDOMX_CACHE_ACCESSES * (
		// Main loop prologue
		((uint8_t*)randomx_calc_dataset_item_rv64_mix - ((uint8_t*)randomx_calc_dataset_item_rv64_prefetch)) + 4 +
		// Inner main loop (instructions)
		((RANDOMX_SUPERSCALAR_LATENCY * 3) + 2) * 16 +
		// Main loop epilogue
		((uint8_t*)randomx_calc_dataset_item_rv64_store_result - (uint8_t*)randomx_calc_dataset_item_rv64_mix) + 4
	) +
	// Epilogue
	((uint8_t*)randomx_calc_dataset_item_rv64_end - (uint8_t*)randomx_calc_dataset_item_rv64_store_result);

constexpr uint32_t IntRegMap[8] = { 28, 29, 30, 31, 20, 21, 22, 23 };
constexpr uint32_t IntRegMap_SuperScalar[8] = { 10, 11, 12, 13, 14, 15, 16, 17 };


template<typename T> static constexpr size_t Log2(T value) { return (value > 1) ? (Log2(value / 2) + 1) : 0; }

JitCompilerRV64::JitCompilerRV64()
	: code((uint8_t*) allocMemoryPages(CodeSize + CalcDatasetItemSize))
	, literalPos(ImulRcpLiteralsEnd)
	, num32bitLiterals(0)
{
	memset(reg_changed_offset, 0, sizeof(reg_changed_offset));
	memcpy(code, (void*) randomx_program_rv64, CodeSize);
}

JitCompilerRV64::~JitCompilerRV64()
{
	freePagedMemory(code, CodeSize + CalcDatasetItemSize);
}

void JitCompilerRV64::enableWriting()
{
	setPagesRW(code, CodeSize + CalcDatasetItemSize);
}

void JitCompilerRV64::enableExecution()
{
	setPagesRX(code, CodeSize + CalcDatasetItemSize);
}

void JitCompilerRV64::enableAll()
{
	setPagesRWX(code, CodeSize + CalcDatasetItemSize);
}

void JitCompilerRV64::generateProgram(Program& program, ProgramConfiguration& config)
{
	uint32_t codePos = MainLoopBegin + 4;
	// These values are calculated from the ARM code AND opcodes using the following web pages:
	// https://gist.github.com/dinfuehr/51a01ac58c0b23e4de9aac313ed6a06a
	// https://dinfuehr.github.io/blog/encoding-of-immediate-values-on-aarch64/
	// https://www.calculator.net/log-calculator.html?xv=2147483648&base=2&yv=&x=43&y=23
	// And the ARM Architecture Reference Manual Armv8, for Armv8-A architecure profile
	//const uint32_t ScpadL3Mask64 = 0x1fffc0;
	uint32_t ScpadL3Mask64_hi = ScratchpadL3Mask64 >> 12; //0x200; 
	uint32_t ScpadL3Mask64_lo = ScratchpadL3Mask64 & ((1 << 12) - 1); //-64;
	if (ScpadL3Mask64_lo & 0x800)
	{
		ScpadL3Mask64_hi += 1;
		ScpadL3Mask64_lo = ScpadL3Mask64_lo - 0x1000;
	}
	//const uint32_t RandomxDataSetBaseSizeMask = 0x7fffffc0;
	uint32_t RandomxDataSetBaseSizeMask_hi = CacheLineAlignMask >> 12; //0x80000; 
	uint32_t RandomxDataSetBaseSizeMask_lo = CacheLineAlignMask & ((1 << 12) - 1); //-64;
	if (RandomxDataSetBaseSizeMask_lo & 0x800)
	{
		RandomxDataSetBaseSizeMask_hi += 1;
		RandomxDataSetBaseSizeMask_lo = RandomxDataSetBaseSizeMask_lo - 0x1000;
	}

	uint32_t temp1 = 27;
	uint32_t temp0 = 26;
	uint32_t spAddr0 = 24;
	uint32_t spMix1 = 18;
	uint32_t spAddr1 = 25;
	uint32_t spPtr = 6;

	// Load ScpadL3Mask64 into temp1
	emit32( LUI(temp1, ScpadL3Mask64_hi), code, codePos ); //overwrites placeholder in asm
	// Add in the lower value so we have the mask in temp1
	emit32(ADDIW(temp1, temp1, ScpadL3Mask64_lo), code, codePos);

	// and spAddr0, spMix1, ScpadL3Mask64
	emit32( AND(spAddr0, spMix1, temp1), code, codePos);  //overwrites placeholder in asm

	// and spAddr1, temp0, ScpadL3Mask64
	emit32( AND(spAddr1, temp0, temp1), code, codePos); //overwrites placeholder in asm

	codePos = PrologueSize;
	literalPos = ImulRcpLiteralsEnd;
	num32bitLiterals = 0;

	for (uint32_t i = 0; i < RegistersCount; ++i)
		reg_changed_offset[i] = codePos;

	//Generate random instructions
	for (uint32_t i = 0; i < program.getSize(); ++i)
	{
		Instruction& instr = program(i);
		instr.src %= RegistersCount;
		instr.dst %= RegistersCount;
		(this->*opTable[instr.opcode])(instr, codePos); //Jump to h_INSTR emmision function by opcode
	}

	//main loop final steps

	// Update spMix1
	// eor w18, config.readReg2, config.readReg3
	emit32( XOR(18, IntRegMap[config.readReg2], IntRegMap[config.readReg3]), code, codePos);

	// Jump back to the main loop
	// This will be encode as a J <imm> and not a JAL. Because the rd = x0
	const uint32_t offset = (((uint8_t*)randomx_program_rv64_vm_instructions_end) - ((uint8_t*)randomx_program_rv64)) - codePos;
	emit32(JAL(0, offset), code, codePos);

	//insert masks

#if 0 // This is for a prefetch we do not do in Risc V now...	
	// and temp0, ma,mx, CacheLineAlignMask
	codePos = (((uint8_t*)randomx_program_rv64_cacheline_align_mask1) - ((uint8_t*)randomx_program_rv64));
	// Load ScratchpadL3Mask64 into temp1
	emit32( LUI(temp1, RandomxDataSetBaseSizeMask_hi), code, codePos ); //overwrites placeholder in asm
	// Add in the lower value so we have the mask in temp1
	emit32(ADDIW(temp1, temp1, RandomxDataSetBaseSizeMask_lo), code, codePos);
	emit32(AND(26, 26, temp1), code, codePos);
#endif

	// and spMix1, ma,mx, CacheLineAlignMask
	codePos = (((uint8_t*)randomx_program_rv64_cacheline_align_mask2) - ((uint8_t*)randomx_program_rv64));
	// Load ScratchpadL3Mask64 into temp1
	emit32( LUI(temp1, RandomxDataSetBaseSizeMask_hi), code, codePos ); //overwrites placeholder in asm
	// Add in the lower value so we have the mask in temp1	
	emit32(ADDIW(temp1, temp1, RandomxDataSetBaseSizeMask_lo), code, codePos);
	emit32(AND(18, 18, temp1), code, codePos);

	// Update spMix1
	// eor x10, config.readReg0, config.readReg1
	codePos = ((uint8_t*)randomx_program_rv64_update_spMix1) - ((uint8_t*)randomx_program_rv64);
	emit32( XOR(18, IntRegMap[config.readReg0], IntRegMap[config.readReg1]), code, codePos);

#ifdef PRINT_GEN_PROGRAM

	printf("Program in memory after insertions\n");
	printf("##################################\n");
	printf("Main program starts at : %x\n", (uint64_t)randomx_program_rv64);
	printf("Main program copied to : %x\n", (uint64_t)code);
	for (uint32_t x = 0; x < CodeSize; x+=4)
	{
		//if ( (*(uint32_t *)(code + x) != 0) && (*(uint32_t *)(randomx_program_rv64 + x) == 0) )
		if (*(uint32_t *)(code + x) != 0)
			printf("Opcode %x : %x \t %x\n", x, *(uint32_t *)(randomx_program_rv64 + x), *(uint32_t *)(code + x) );
	}

	uint8_t* p1 = (uint8_t*)randomx_calc_dataset_item_rv64;
	uint8_t* p2 = (uint8_t*)randomx_calc_dataset_item_rv64_prefetch;
	uint32_t psize = p2 - p1;

	codePos = ((uint8_t*)randomx_init_dataset_rv64_end) - ((uint8_t*)randomx_program_rv64);
	printf("SuperscalarHash program\n");
	printf("Program Size %d\n", psize);
	printf("codePos %x\n", codePos);
	printf("##################################\n");
	for (uint32_t x = 0; x < psize; x+=4)
	{
		printf("Opcode %x : %x \n", x+codePos, *(uint32_t *)(code + codePos + x) );
	}
	
	codePos += psize;
	printf("SuperscalarHash program2\n");
	printf("##################################\n");
	for (uint32_t x = 40000; x < 40600; x+=4) //1000 opcodes.. 
	{
		printf("Opcode %x : %x \n", x+codePos, *(uint32_t *)(code + codePos + x) );
	}
#endif



#ifdef __GNUC__
	__builtin___clear_cache(reinterpret_cast<char*>(code + MainLoopBegin), reinterpret_cast<char*>(code + codePos));
#endif
}

void JitCompilerRV64::generateProgramLight(Program& program, ProgramConfiguration& config, uint32_t datasetOffset)
{
	uint32_t codePos = MainLoopBegin + 4;
	uint32_t temp1 = 27;
	uint32_t temp0 = 26;
	uint32_t spAddr0 = 24;
	uint32_t spMix1 = 18;
	uint32_t spAddr1 = 25;
	uint32_t spPtr = 6;
	// These values are calculated from the ARM code AND opcodes using the following web pages:
	// https://gist.github.com/dinfuehr/51a01ac58c0b23e4de9aac313ed6a06a
	// https://dinfuehr.github.io/blog/encoding-of-immediate-values-on-aarch64/
	// https://www.calculator.net/log-calculator.html?xv=2147483648&base=2&yv=&x=43&y=23
	// And the ARM Architecture Reference Manual Armv8, for Armv8-A architecure profile
	//const uint32_t ScpadL3Mask64 = 0x1fffc0;
	uint32_t ScpadL3Mask64_hi = ScratchpadL3Mask64 >> 12; //0x200; 
	uint32_t ScpadL3Mask64_lo = ScratchpadL3Mask64 & ((1 << 12) - 1); //-64;
	if (ScpadL3Mask64_lo & 0x800)
	{
		ScpadL3Mask64_hi += 1;
		ScpadL3Mask64_lo = ScpadL3Mask64_lo - 0x1000;
	}

	//const uint32_t RandomxDataSetBaseSizeMask = 0x7fffffc0;
	uint32_t RandomxDataSetBaseSizeMask_hi = CacheLineAlignMask >> 12; //0x80000; 
	uint32_t RandomxDataSetBaseSizeMask_lo = CacheLineAlignMask & ((1 << 12) - 1); //-64;
	if (RandomxDataSetBaseSizeMask_lo & 0x800)
	{
		RandomxDataSetBaseSizeMask_hi += 1;
		RandomxDataSetBaseSizeMask_lo = RandomxDataSetBaseSizeMask_lo - 0x1000;
	}

	// Load ScpadL3Mask64 into temp1
	emit32( LUI(temp1, ScpadL3Mask64_hi), code, codePos ); //overwrites placeholder in asm
	// Add in the lower value so we have the mask in temp1
	emit32(ADDIW(temp1, temp1, ScpadL3Mask64_lo), code, codePos);

	// and spAddr0, spMix1, ScpadL3Mask64
	emit32( AND(spAddr0, spMix1, temp1), code, codePos);  //overwrites placeholder in asm

	// and spAddr1, temp0, ScpadL3Mask64
	emit32( AND(spAddr1, temp0, temp1), code, codePos); //overwrites placeholder in asm

	codePos = PrologueSize;
	literalPos = ImulRcpLiteralsEnd;
	num32bitLiterals = 0;

	for (uint32_t i = 0; i < RegistersCount; ++i)
		reg_changed_offset[i] = codePos;

	for (uint32_t i = 0; i < program.getSize(); ++i)
	{
		Instruction& instr = program(i);
		instr.src %= RegistersCount;
		instr.dst %= RegistersCount;
		(this->*opTable[instr.opcode])(instr, codePos);  //Jump to h_INSTR emmision function by opcode
	}

	// Update spMix
	// eor w18, config.readReg2, config.readReg3
	emit32( XOR(18, IntRegMap[config.readReg2], IntRegMap[config.readReg3]), code, codePos);

	// Jump back to the main loop
	// This will be encode as a J <imm> and not a JAL. Because the rd = x0
	const uint32_t offset = (((uint8_t*)randomx_program_rv64_vm_instructions_end_light) - ((uint8_t*)randomx_program_rv64)) - codePos;
	emit32(JAL(0,offset), code, codePos);

	// and a2, ma,mx, CacheLineAlignMask
	//This a2 will be passed as a parameter to the randomx_calc_dataset_item_rv64() function 
	codePos = (((uint8_t*)randomx_program_rv64_light_cacheline_align_mask) - ((uint8_t*)randomx_program_rv64));
	// Load ScratchpadL3Mask64 into temp1
	emit32( LUI(temp1, RandomxDataSetBaseSizeMask_hi), code, codePos ); //overwrites placeholder in asm
	// Add in the lower value so we have the mask in temp1
	emit32(ADDIW(temp1, temp1, RandomxDataSetBaseSizeMask_lo), code, codePos);
	emit32(AND(12, 9, temp1), code, codePos);

	// Update spMix1
	// eor x10, config.readReg0, config.readReg1
	codePos = ((uint8_t*)randomx_program_rv64_update_spMix1) - ((uint8_t*)randomx_program_rv64);
	emit32( XOR(18, IntRegMap[config.readReg0], IntRegMap[config.readReg1]), code, codePos);

	// Apply dataset offset to a2 which is then passed to the randomx_calc_dataset_item_rv64() function.
	codePos = ((uint8_t*)randomx_program_rv64_light_dataset_offset) - ((uint8_t*)randomx_program_rv64);

	datasetOffset /= CacheLineSize;
	//printf("datasetOffset %x\n", datasetOffset);
	int32_t imm_lo = datasetOffset & ((1 << 12) - 1);
	uint32_t imm_hi = datasetOffset >> 12;
	// Check for upper bit (signed bit set)
	if (imm_lo & 0x800)
	{
		imm_hi += 1;
		imm_lo = imm_lo - 0x1000;
	}
	//printf("imm hi %x imm_lo %x\n", imm_hi, imm_lo);
	//overwrites placeholders in asm
	emit32( LUI(temp1, imm_hi), code, codePos ); 
	// Add in the lower value so we have the mask in temp1
	emit32(ADDIW(temp1, temp1, imm_lo), code, codePos);
	// Add the mask to the Scratchpad Pointer
	emit32(ADD(12, 12, temp1), code, codePos);

#ifdef PRINT_GEN_PROGRAM

	printf("Program in memory after insertions\n");
	printf("##################################\n");
	printf("Main program starts at : %x\n", (uint64_t)randomx_program_rv64);
	printf("Main program copied to : %x\n", (uint64_t)code);
	for (uint32_t x = 0; x < CodeSize; x+=4)
	{
		//if ( (*(uint32_t *)(code + x) != 0) && (*(uint32_t *)(randomx_program_rv64 + x) == 0) )
		if (*(uint32_t *)(code + x) != 0)
			printf("Opcode %x : %x \t %x\n", x, *(uint32_t *)(randomx_program_rv64 + x), *(uint32_t *)(code + x) );
	}

	uint8_t* p1 = (uint8_t*)randomx_calc_dataset_item_rv64;
	uint8_t* p2 = (uint8_t*)randomx_calc_dataset_item_rv64_prefetch;
	uint32_t psize = p2 - p1;

	codePos = ((uint8_t*)randomx_init_dataset_rv64_end) - ((uint8_t*)randomx_program_rv64);
	printf("SuperscalarHash program\n");
	printf("Program Size %d\n", psize);
	printf("codePos %x\n", codePos);
	printf("##################################\n");
	for (uint32_t x = 0; x < psize; x+=4)
	{
		printf("Opcode %x : %x \n", x+codePos, *(uint32_t *)(code + codePos + x) );
	}
	
	codePos += psize;
	printf("SuperscalarHash program2\n");
	printf("##################################\n");
	for (uint32_t x = 40000; x < 40600; x+=4) //1000 opcodes.. 
	{
		printf("Opcode %x : %x \n", x+codePos, *(uint32_t *)(code + codePos + x) );
	}
#endif

#ifdef __GNUC__
	__builtin___clear_cache(reinterpret_cast<char*>(code + MainLoopBegin), reinterpret_cast<char*>(code + codePos));
#endif
}

template<size_t N>
void JitCompilerRV64::generateSuperscalarHash(SuperscalarProgram(&programs)[N], std::vector<uint64_t> &reciprocalCache)
{
	uint32_t codePos = CodeSize;

	uint8_t* p1 = (uint8_t*)randomx_calc_dataset_item_rv64;
	uint8_t* p2 = (uint8_t*)randomx_calc_dataset_item_rv64_prefetch;
	// These values are calculated from the ARM code AND opcodes using the following web pages:
	// https://gist.github.com/dinfuehr/51a01ac58c0b23e4de9aac313ed6a06a
	// https://dinfuehr.github.io/blog/encoding-of-immediate-values-on-aarch64/
	// https://www.calculator.net/log-calculator.html?xv=2147483648&base=2&yv=&x=43&y=23
	// And the ARM Architecture Reference Manual Armv8, for Armv8-A architecure profile
	// This is based on the CacheSize / CacheLineSize - 1 calculation and the ARM Opcode
	// and x11, x10, CacheSize / CacheLineSize - 1
	// If either cacheSize or CacheLineSize changes this must be recalculated.
	//const uint32_t CacheSizeMask = 0x3fffff;
	uint32_t CacheSizeMask_hi = 0x400; 
	uint32_t CacheSizeMask_lo = -1;

	memcpy(code + codePos, p1, p2 - p1);
	codePos += p2 - p1;

	num32bitLiterals = 64;
	constexpr uint32_t temp0 = 26;
	constexpr uint32_t temp1 = 27;

	for (size_t i = 0; i < N; ++i)
	{
		// Offset used for IMUL_RCP to load literal values
		uint32_t literal_offset = 0;

		// and x8(second lit for IMUL_RCP), x18(spMix1), CacheSize / CacheLineSize - 1
		// move the value into temp1.
		// Load ScratchpadL3Mask64 into temp1
		emit32( LUI(temp1, CacheSizeMask_hi), code, codePos ); //overwrites placeholder in asm
		// Add in the lower value so we have the mask in temp1
		emit32(ADDIW(temp1, temp1, CacheSizeMask_lo), code, codePos);
		emit32(AND(19, 18, temp1), code, codePos);

		//emit32(0xffffffff, code, codePos);

		p1 = ((uint8_t*)randomx_calc_dataset_item_rv64_prefetch) + 12;
		p2 = (uint8_t*)randomx_calc_dataset_item_rv64_mix;
		memcpy(code + codePos, p1, p2 - p1);
		codePos += p2 - p1;

		SuperscalarProgram& prog = programs[i];
		const size_t progSize = prog.getSize();

		uint32_t jmp_pos = codePos;
		codePos += 4;

		// Fill in literal pool
		for (size_t j = 0; j < progSize; ++j)
		{
			const Instruction& instr = prog(j);
			if (static_cast<SuperscalarInstructionType>(instr.opcode) == randomx::SuperscalarInstructionType::IMUL_RCP)
				emit64(reciprocalCache[instr.getImm32()], code, codePos);
		}

		// Jump over literal pool
		uint32_t literal_pos = jmp_pos;
		emit32(JAL(0, (codePos - jmp_pos)), code, literal_pos);

		for (size_t j = 0; j < progSize; ++j)
		{
			const Instruction& instr = prog(j);
			const uint32_t src = IntRegMap_SuperScalar[instr.src];
			const uint32_t dst = IntRegMap_SuperScalar[instr.dst];

			switch (static_cast<SuperscalarInstructionType>(instr.opcode))
			{
			case randomx::SuperscalarInstructionType::ISUB_R:
				emit32( SUB(dst,dst,src), code, codePos);
				break;
			case randomx::SuperscalarInstructionType::IXOR_R:
				emit32( XOR(dst, dst, src), code, codePos);
				break;
			case randomx::SuperscalarInstructionType::IADD_RS:
				//printf("IADD_RS\n");
				if ( instr.getModShift() == 0 )
				{
					emit32(ADD(dst, dst, src), code, codePos);
				}
				else
				{
					emit32(SLLI(temp0, src, instr.getModShift()), code, codePos);
					emit32(ADD(dst, dst, temp0), code, codePos);
				}
				break;
			case randomx::SuperscalarInstructionType::IMUL_R:
				emit32( MUL(dst, dst, src), code, codePos);
				break;
			case randomx::SuperscalarInstructionType::IROR_C:
#ifdef BITMANIP			
				emit32( RORI(dst, dst, (instr.getImm32() & 0x3F )), code, codePos);
#else
				{
					constexpr uint32_t tmp0 = 26;
					constexpr uint32_t tmp1 = 27;
					uint32_t rori_amt;
					rori_amt = instr.getImm32() & 0x3F; // limit imm to 6 bits, 0x3f or less
					//printf("IROR_C masked %x\n", rori_amt);
					//printf("IROR_C unmasked %x\n", instr.getImm32());
					emit32( ORI(tmp0, 0, 63), code, codePos); // temp0 now has 63
					emit32( ORI(tmp1, 0, rori_amt), code, codePos); // temp1 now has imm
					emit32( SUB(tmp1, tmp0, tmp1), code, codePos); // temp1 now has 63 - imm
					emit32( SRLI(tmp0, dst, rori_amt), code, codePos); // shift the dst right and put it into temp0
					emit32( SLL(dst, dst, tmp1), code, codePos); // shift the dst left and put it into dst
					emit32( OR(dst, dst, tmp0), code, codePos); // Now or the two values together to get the ror
				}
#endif
				break;
			case randomx::SuperscalarInstructionType::IADD_C7:
			case randomx::SuperscalarInstructionType::IADD_C8:
			case randomx::SuperscalarInstructionType::IADD_C9:
				emitAddImmediate(dst, dst, instr.getImm32(), code, codePos);
				break;
			case randomx::SuperscalarInstructionType::IXOR_C7:
			case randomx::SuperscalarInstructionType::IXOR_C8:
			case randomx::SuperscalarInstructionType::IXOR_C9:
				emitMovImmediate(temp0, instr.getImm32(), code, codePos);
				emit32(XOR(dst, dst, temp0), code, codePos);
				break;
			case randomx::SuperscalarInstructionType::IMULH_R:
				emit32( MULHU(dst, dst, src), code, codePos);
				break;
			case randomx::SuperscalarInstructionType::ISMULH_R:
				emit32( MULHS(dst, dst, src), code, codePos);
				break;
			case randomx::SuperscalarInstructionType::IMUL_RCP:
				{
					//This will be a 64 bit address
					int64_t literal_addr = ((uint64_t)code + (literal_pos + literal_offset));
					int64_t literal_addr_A, literal_addr_B, literal_addr_C, literal_addr_D, literal_addr_E;

					literal_addr_A = literal_addr >> 48; 
					literal_addr_B = (literal_addr & 0x0000fff000000000) >> 36;
					literal_addr_C = (literal_addr & 0x0000000fff000000) >> 24;
					literal_addr_D = (literal_addr & 0x0000000000fff000) >> 12;
					literal_addr_E = (literal_addr & 0x0000000000000fff); 
					if (literal_addr_B & 0x800)
					{
						literal_addr_A +=1;
						literal_addr_B = literal_addr_B - 0x1000;
					}
					if (literal_addr_C & 0x800)
					{
						literal_addr_B +=1;
						literal_addr_C = literal_addr_C - 0x1000;
					}
					if (literal_addr_D & 0x800)
					{
						literal_addr_C +=1;
						literal_addr_D = literal_addr_D - 0x1000;
					}
					if (literal_addr_E & 0x800)
					{
						literal_addr_D +=1;
						literal_addr_E = literal_addr_E - 0x1000;
					}

					printf("literal address %lx\n", literal_addr);
					//printf("literal address A %lx\n", literal_addr_A);
					//printf("literal address B %lx\n", literal_addr_B);
					//printf("literal address C %lx\n", literal_addr_C);
					//printf("literal address D %lx\n", literal_addr_D);
					//printf("literal address E %lx\n", literal_addr_E);

					emit32( LUI(temp1, literal_addr_A), code, codePos );
					emit32(ADDIW(temp1, temp1, literal_addr_B), code, codePos);
					emit32(SLLI(temp1, temp1, 0xC), code, codePos);
					emit32(ADDI(temp1, temp1, literal_addr_C), code, codePos);
					emit32(SLLI(temp1, temp1, 0xC), code, codePos);
					emit32(ADDI(temp1, temp1, literal_addr_D), code, codePos);
					emit32(SLLI(temp1, temp1, 0xC), code, codePos);
					emit32(ADDI(temp1, temp1, literal_addr_E), code, codePos);
					
					emit32( LWU(temp0, temp1, 0), code, codePos);
					literal_offset += 4;

	emit32(0xffffffff, code, codePos);

					// mul dst, dst, temp0
					emit32( MUL(dst,dst,temp0), code, codePos);
				}
				break;
			default:
				break;
			}
		}

		p1 = (uint8_t*)randomx_calc_dataset_item_rv64_mix;
		p2 = (uint8_t*)randomx_calc_dataset_item_rv64_store_result;
		memcpy(code + codePos, p1, p2 - p1);
		codePos += p2 - p1;

		// Update registerValue
		emit32(ADDI(18, prog.getAddressRegister(), 0), code, codePos);
	}

	p1 = (uint8_t*)randomx_calc_dataset_item_rv64_store_result;
	p2 = (uint8_t*)randomx_calc_dataset_item_rv64_end;
	memcpy(code + codePos, p1, p2 - p1);
	codePos += p2 - p1;

#ifdef PRINT_SUPERSCALAR_PROGRAM

	uint8_t* px = (uint8_t*)randomx_calc_dataset_item_rv64;
	uint8_t* py = (uint8_t*)randomx_calc_dataset_item_rv64_prefetch;
	uint32_t psize = py - px;

	codePos = ((uint8_t*)randomx_init_dataset_rv64_end) - ((uint8_t*)randomx_program_rv64);
	printf("SuperscalarHash program\n");
	printf("Program Size %d\n", psize);
	printf("codePos %x\n", codePos);
	printf("##################################\n");
	for (uint32_t x = 0; x < psize; x+=4)
	{
		printf("Opcode %x : %x \n", x+codePos, *(uint32_t *)(code + codePos + x) );
	}
	
	codePos += psize;
	printf("SuperscalarHash program2\n");
	printf("##################################\n");
	for (uint32_t x = 0; x < 1000; x+=4) //1000 opcodes.. 
	{
		printf("Opcode %x : %x \n", x+codePos, *(uint32_t *)(code + codePos + x) );
	}
#endif



}

template void JitCompilerRV64::generateSuperscalarHash(SuperscalarProgram(&programs)[RANDOMX_CACHE_ACCESSES], std::vector<uint64_t> &reciprocalCache);

DatasetInitFunc* JitCompilerRV64::getDatasetInitFunc()
{
#ifdef DEBUG_MINING	
	printf("function address %lx\n", (uint64_t)(code + (((uint8_t*)randomx_init_dataset_rv64) - ((uint8_t*)randomx_program_rv64))));
#endif
	return (DatasetInitFunc*)(code + (((uint8_t*)randomx_init_dataset_rv64) - ((uint8_t*)randomx_program_rv64)));
}

size_t JitCompilerRV64::getCodeSize()
{
	return CodeSize;
}

void JitCompilerRV64::emitMovImmediate(uint32_t dst, uint32_t imm, uint8_t* code, uint32_t& codePos)
{
	uint32_t k = codePos;

	uint32_t imm_lo = imm & ((1 << 12) - 1);
	uint32_t imm_hi = imm >> 12;
	constexpr uint32_t temp0 = 26;
	// Check for upper bit (signed bit set)
	if (imm_lo & 0x800)
	{
		imm_hi += 1;
		imm_lo = imm_lo - 0x1000;
	}
	// Load the upper value into temp0
	emit32( LUI(temp0, imm_hi), code, k );	
	// Add the imm_hi + imm_lo -> dst
	emit32(ADDIW(dst, temp0, imm_lo), code, k);

	codePos = k;
}

void JitCompilerRV64::emitAddImmediate(uint32_t dst, uint32_t src, uint32_t imm, uint8_t* code, uint32_t& codePos)
{
	uint32_t k = codePos;

	uint32_t imm_lo = imm & ((1 << 12) - 1);
	uint32_t imm_hi = imm >> 12;
	constexpr uint32_t temp1 = 27;
	// Check for upper bit (signed bit set)
	if (imm_lo & 0x800)
	{
		imm_hi += 1;
		imm_lo = imm_lo - 0x1000;
	}
	if (imm_lo && imm_hi)
	{
		// Load the upper value into temp1
		emit32( LUI(temp1, imm_hi), code, k );
		// Add in the lower value so we have the imm in temp1
		emit32(ADDIW(temp1, temp1, imm_lo), code, k);
	}
	else if (imm_lo)
	{
		// move the lower value into temp1
		emitMovImmediate(temp1, imm_lo, code, k);
	}
	else
	{
		// Load the upper value into temp1
		emit32( LUI(temp1, imm_hi), code, k );
	}

	// add src and temp0 -> dst
	emit32(ADD(dst, src, temp1), code, k);

	codePos = k;
}

// tmp = 26 = temp0
template<uint32_t tmp>
void JitCompilerRV64::emitMemLoad(uint32_t dst, uint32_t src, Instruction& instr, uint8_t* code, uint32_t& codePos)
{
	uint32_t k = codePos;
	constexpr uint32_t temp1 = 27;

	uint32_t imm = instr.getImm32();
	uint32_t mask_hi, mask_lo;

	if (src != dst) //load from L1 or L2
	{
		if (instr.getModMem()) { //Mod.mem != 0, load from L1
			imm &= (RANDOMX_SCRATCHPAD_L1 - 1);
			emitAddImmediate(tmp, src, imm, code, k);
			mask_lo = (RANDOMX_SCRATCHPAD_L1 - 8) & ((1 << 12) - 1);
			mask_hi = (RANDOMX_SCRATCHPAD_L1 - 8) >> 12;
			// Check for upper bit (signed bit set)
			if (mask_lo & 0x800)
			{
				mask_hi += 1;
				mask_lo = mask_lo - 0x1000;
			}
			// Load the upper value
			emit32( LUI(temp1, mask_hi), code, k );
			// Add in the lower value so we have the imm in temp1
			emit32(ADDIW(temp1, temp1, mask_lo), code, k);
			emit32( AND(tmp, tmp, temp1), code, k);			 
		}
		else { //Mod.mem==0, load from L2
			imm &= (RANDOMX_SCRATCHPAD_L2 - 1);
			emitAddImmediate(tmp, src, imm, code, k);
			mask_lo = (RANDOMX_SCRATCHPAD_L2 - 8) & ((1 << 12) - 1);
			mask_hi = (RANDOMX_SCRATCHPAD_L2 - 8) >> 12;
			// Check for upper bit (signed bit set)
			if (mask_lo & 0x800)
			{
				mask_hi += 1;
				mask_lo = mask_lo - 0x1000;
			}			
			// Load the upper value
			emit32( LUI(temp1, mask_hi), code, k );
			// Add in the lower value so we have the imm in temp1
			emit32(ADDIW(temp1, temp1, mask_lo), code, k);
			emit32( AND(tmp, tmp, temp1), code, k);
		}
	}
	else //src==dst, load from L3 scratch range
	{
		// Clear out lower 3 bits. No idea why ARM was doing this...??
		imm = (imm & ScratchpadL3Mask) >> 3;
		imm = (imm & ScratchpadL3Mask) << 3;
		emitMovImmediate(tmp, imm, code, k);
	}

	// Add temp0 to pointer to scratchpad
	//emit32(ADD(tmp, 6, tmp), code, k);

	// Load from scratchpad into dst 
	//emit32(LD(dst, tmp, 0), code, k);

	codePos = k;
}

template<uint32_t tmp_fp>
void JitCompilerRV64::emitMemLoadFP(uint32_t src, Instruction& instr, uint8_t* code, uint32_t& codePos)
{
	uint32_t k = codePos;

	uint32_t imm = instr.getImm32();
	uint32_t mask_hi, mask_lo;
	constexpr uint32_t temp0 = 26;
	constexpr uint32_t temp1 = 27;
	
	//Address calculation
	//loadfp only from L1 or L2; only loads mem src operands, so src=dst not sensible; src is an integer reg
	if (instr.getModMem()) { //Mod.mem != 0, load from L1
		imm &= (RANDOMX_SCRATCHPAD_L1 - 1);
		emitAddImmediate(temp0, src, imm, code, k);
		mask_lo = (RANDOMX_SCRATCHPAD_L1 - 8) & ((1 << 12) - 1);
		mask_hi = (RANDOMX_SCRATCHPAD_L1 - 8) >> 12;
		// Check for upper bit (signed bit set)
		if (mask_lo & 0x800)
		{
			mask_hi += 1;
			mask_lo = mask_lo - 0x1000;
		}		
		// Load the upper value
		emit32( LUI(temp1, mask_hi), code, k );
		// Add in the lower value so we have the imm in temp1
		emit32(ADDIW(temp1, temp1, mask_lo), code, k);
		emit32( AND(temp0, temp0, temp1), code, k);	
	}
	else { //Mod.mem==0, load from L2
		imm &= (RANDOMX_SCRATCHPAD_L2 - 1);
		emitAddImmediate(temp0, src, imm, code, k);
		mask_lo = (RANDOMX_SCRATCHPAD_L2 - 8) & ((1 << 12) - 1);
		mask_hi = (RANDOMX_SCRATCHPAD_L2 - 8) >> 12;
		// Check for upper bit (signed bit set)
		if (mask_lo & 0x800)
		{
			mask_hi += 1;
			mask_lo = mask_lo - 0x1000;
		}		
		// Load the upper value
		emit32( LUI(temp1, mask_hi), code, k );
		// Add in the lower value so we have the imm in temp1
		emit32(ADDIW(temp1, temp1, mask_lo), code, k);
		emit32( AND(temp0, temp0, temp1), code, k);
	}

	// Add temp0 to pointer to scratchpad
	emit32(ADD(temp0, 6, temp0), code, k);
	
	emit32(LW(temp1, temp0, 0), code, k); //load lower word to int reg for conversion
	emit32(FCVTDW(tmp_fp, temp1, 7), code, k); //convert int32 to double

	emit32(LW(temp1, temp0, 4), code, k); //load upper word
	emit32(FCVTDW(tmp_fp+1, temp1, 7), code, k); //convert int32 to double

	//caller will complete conversion masking (different for mul or add/sub)

	codePos = k;
}


void JitCompilerRV64::h_IADD_RS(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = IntRegMap[instr.src];
	const uint32_t dst = IntRegMap[instr.dst];
	const uint32_t shift = instr.getModShift();

	constexpr uint32_t tmp = 26;

	// add dst, src << shift
	emit32(SLLI(tmp, src, shift), code, k);
	emit32(ADD(dst, dst, tmp), code, k);

	reg_changed_offset[instr.dst] = k;
	codePos = k;
}

void JitCompilerRV64::h_IADD_M(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = IntRegMap[instr.src];
	const uint32_t dst = IntRegMap[instr.dst];

	constexpr uint32_t tmp = 26;
	emitMemLoad<tmp>(dst, src, instr, code, k);

	// add dst, dst, tmp
	emit32( ADD(dst,dst,tmp), code, k);

	reg_changed_offset[instr.dst] = k;
	codePos = k;
}

void JitCompilerRV64::h_ISUB_R(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = IntRegMap[instr.src];
	const uint32_t dst = IntRegMap[instr.dst];
	constexpr uint32_t tmp0 = 26;
	constexpr uint32_t tmp1 = 27;
	uint32_t sub_imm;

	if (src != dst)
	{
		// sub dst, dst, src
		emit32( SUB(dst,dst,src), code, k);
	}
	else //src==dst, subtract immediate instead (else get zero(
	{
		sub_imm = instr.getImm32() & 0xFFF; // Grab the immediate and limit it to 12 bits
		emit32( ORI(tmp0, 0, sub_imm), code, k); // Put imm into tmp0
		emit32( SUB(dst,dst,tmp0), code, k);
	}

	reg_changed_offset[instr.dst] = k;
	codePos = k;
}

void JitCompilerRV64::h_ISUB_M(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = IntRegMap[instr.src];
	const uint32_t dst = IntRegMap[instr.dst];

	constexpr uint32_t tmp = 26;
	emitMemLoad<tmp>(dst, src, instr, code, k);

	// sub dst, dst, tmp
	emit32( SUB(dst,dst,tmp), code, k);

	reg_changed_offset[instr.dst] = k;
	codePos = k;
}

void JitCompilerRV64::h_IMUL_R(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	uint32_t src = IntRegMap[instr.src];
	const uint32_t dst = IntRegMap[instr.dst];

	if (src == dst)
	{
		src = 26;
		emitMovImmediate(src, instr.getImm32(), code, k);
	}

	// mul dst, dst, src
	emit32( MUL(dst,dst,src), code, k);

	reg_changed_offset[instr.dst] = k;
	codePos = k;
}

void JitCompilerRV64::h_IMUL_M(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = IntRegMap[instr.src];
	const uint32_t dst = IntRegMap[instr.dst];

	constexpr uint32_t tmp = 26;
	emitMemLoad<tmp>(dst, src, instr, code, k);

	// sub dst, dst, tmp
	emit32( MUL(dst, dst, tmp), code, k);

	reg_changed_offset[instr.dst] = k;
	codePos = k;
}

void JitCompilerRV64::h_IMULH_R(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = IntRegMap[instr.src];
	const uint32_t dst = IntRegMap[instr.dst];

	// umulh dst, dst, src
	emit32(MULHU(dst, dst, src), code, k);

	reg_changed_offset[instr.dst] = k;
	codePos = k;
}

void JitCompilerRV64::h_IMULH_M(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = IntRegMap[instr.src];
	const uint32_t dst = IntRegMap[instr.dst];

	constexpr uint32_t tmp = 26;
	emitMemLoad<tmp>(dst, src, instr, code, k);

	// umulh dst, dst, tmp
	emit32(MULHU(dst, dst, tmp), code, k);

	reg_changed_offset[instr.dst] = k;
	codePos = k;
}

void JitCompilerRV64::h_ISMULH_R(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = IntRegMap[instr.src];
	const uint32_t dst = IntRegMap[instr.dst];

	// smulh dst, dst, src
	emit32(MULHS(dst, dst, src), code, k);

	reg_changed_offset[instr.dst] = k;
	codePos = k;
}

void JitCompilerRV64::h_ISMULH_M(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = IntRegMap[instr.src];
	const uint32_t dst = IntRegMap[instr.dst];

	constexpr uint32_t tmp = 26;
	emitMemLoad<tmp>(dst, src, instr, code, k);

	// smulh dst, dst, tmp
	emit32(MULHS(dst, dst, tmp), code, k);

	reg_changed_offset[instr.dst] = k;
	codePos = k;
}

void JitCompilerRV64::h_IMUL_RCP(Instruction& instr, uint32_t& codePos)
{
	const uint64_t divisor = instr.getImm32();
	if (isZeroOrPowerOf2(divisor))
		return;

	uint32_t k = codePos;

	constexpr uint32_t tmp = 26;
	constexpr uint32_t tmp1 = 27;
	const uint32_t dst = IntRegMap[instr.dst];

	constexpr uint64_t N = 1ULL << 63;
	const uint64_t q = N / divisor;
	const uint64_t r = N % divisor;

	//count leading zeros
	uint64_t shift = 32;
	for (uint64_t k = 1U << 31; (k & divisor) == 0; k >>= 1)
		--shift;

	const uint32_t literal_id = (ImulRcpLiteralsEnd - literalPos) / sizeof(uint64_t);

	literalPos -= sizeof(uint64_t);
	*(uint64_t*)(code + literalPos) = (q << shift) + ((r << shift) / divisor);

	if (literal_id < 11) //some RCP literals stashed in regs
	{
		static constexpr uint32_t literal_regs[13] = { 17, 16, 15, 14, 13, 12, 11, 10, 3, 8, 4 };

		// mul dst, dst, literal_reg
		emit32(MUL(dst, dst, literal_regs[literal_id]), code, k);
	}
	else
	{
		// ldr tmp, reciprocal
		const uint32_t offset = (literalPos - k);
		emit32( AUIPC(tmp1, 0), code , k);
		emit32( LD(tmp, tmp1, offset), code, k);

		// mul dst, dst, tmp
		emit32(MUL(dst, dst, tmp), code, k);
	}

	reg_changed_offset[instr.dst] = k;
	codePos = k;
}

void JitCompilerRV64::h_INEG_R(Instruction& instr, uint32_t& codePos)
{
	const uint32_t dst = IntRegMap[instr.dst];

	// sub dst, xzr, dst
	emit32( SUB(dst, 0, dst), code, codePos);

	reg_changed_offset[instr.dst] = codePos;
}

void JitCompilerRV64::h_IXOR_R(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	uint32_t src = IntRegMap[instr.src];
	const uint32_t dst = IntRegMap[instr.dst];

	if (src == dst)
	{
		src = 26;
		emitMovImmediate(src, instr.getImm32(), code, k);
	}

	// eor dst, dst, src
	emit32( XOR(dst,dst,src), code, k);

	reg_changed_offset[instr.dst] = k;
	codePos = k;
}

void JitCompilerRV64::h_IXOR_M(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = IntRegMap[instr.src];
	const uint32_t dst = IntRegMap[instr.dst];

	constexpr uint32_t tmp = 26;
	emitMemLoad<tmp>(dst, src, instr, code, k);

	// eor dst, dst, tmp
	emit32( XOR(dst, dst, tmp), code, k);

	reg_changed_offset[instr.dst] = k;
	codePos = k;
}

void JitCompilerRV64::h_IROR_R(Instruction& instr, uint32_t& codePos)
{
	const uint32_t src = IntRegMap[instr.src];
	const uint32_t dst = IntRegMap[instr.dst];
	uint32_t rori_amt;

// Turn this on when bit manip extention is present
#ifdef BITMANIP
	if (src != dst)
	{
		// ror dst, dst, src
		emit32( ROR(dst, dst, src), code, codePos);
	}
	else
	{
		// ror dst, dst, imm
		rori_amt = instr.getImm32() & 63;
		//emit32( RORI(dst, dst, rori_amt), code, codePos);
		emit32((rori_amt << 20 | dst << 15 | 0b101 << 12 | dst << 7 | 0b001001), code, codePos);
	}
#else // emulate ror with other opcodes... see below
	constexpr uint32_t tmp0 = 26;
	constexpr uint32_t tmp1 = 27;
	if (src != dst)
	{
		emit32( ORI(tmp0, 0, 64), code, codePos); // temp0 now has 64
		emit32( SUB(tmp1, tmp0, src), code, codePos); // temp1 now has 64 - src
		emit32( SRL(tmp0, dst, src), code, codePos); // shift the dst right and put it into temp0
		emit32( SLL(dst, dst, tmp1), code, codePos); // shift the dst left and put it into dst
		emit32( OR(dst, dst, tmp0), code, codePos); // Now or the two values together to get the ror
	}	
	else
	{
		rori_amt = instr.getImm32() & 0x3F; // limit imm to 6 bits, 0x3f or less
		emit32( ORI(tmp0, 0, 64), code, codePos); // temp0 now has 64
		emit32( ORI(tmp1, 0, rori_amt), code, codePos); // temp1 now has imm
		emit32( SUB(tmp1, tmp0, tmp1), code, codePos); // temp1 now has 64 - imm
		emit32( SRLI(tmp0, dst, rori_amt), code, codePos); // shift the dst right and put it into temp0
		emit32( SLL(dst, dst, tmp1), code, codePos); // shift the dst left and put it into dst
		emit32( OR(dst, dst, tmp0), code, codePos); // Now or the two values together to get the ror
	}
#endif
	reg_changed_offset[instr.dst] = codePos;
}

void JitCompilerRV64::h_IROL_R(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = IntRegMap[instr.src];
	const uint32_t dst = IntRegMap[instr.dst];
	uint32_t rori_amt;
// Turn this on when bit manip extention is present
#ifdef BITMANIP
	if (src != dst)
	{
		constexpr uint32_t tmp = 26;

		// sub tmp, xzr, src
		emit32( SUB(tmp, 0, src), code, k); //todo: hard ROL

		// ror dst, dst, tmp
		emit32( ROR(dst, dst, tmp), code, k);
	}
	else
	{
		// ror dst, dst, imm
		rori_amt = -instr.getImm32() & 63;
		
		//emit32( RORI(dst, dst, rori_amt), code, k);
		emit32( (rori_amt << 20 | dst << 15 | 0b101 << 12 | dst << 7 | 0b001001), code, k);
	}
#else // emulate ror with other opcodes... see below
	constexpr uint32_t tmp0 = 26;
	constexpr uint32_t tmp1 = 27;
	if (src != dst)
	{
		emit32( ORI(tmp0, 0, 32), code, codePos); // temp0 now has 32
		emit32( SUB(tmp1, tmp0, src), code, codePos); // temp1 now has 32 - src
		emit32( SLL(tmp0, dst, src), code, codePos); // shift the dst right and put it into temp0
		emit32( SRL(dst, dst, tmp1), code, codePos); // shift the dst left and put it into dst
		emit32( OR(dst, dst, tmp0), code, codePos); // Now or the two values together to get the ror
	}	
	else
	{
		rori_amt = instr.getImm32() & 0xFFF; // limit imm to 12 bits
		emit32( ORI(tmp0, 0, 32), code, codePos); // temp0 now has 32
		emit32( ORI(tmp1, 0, rori_amt), code, codePos); // temp1 now has imm
		emit32( SUB(tmp1, tmp0, tmp1), code, codePos); // temp1 now has 32 - imm
		emit32( SLLI(tmp0, dst, rori_amt), code, codePos); // shift the dst right and put it into temp0
		emit32( SRL(dst, dst, tmp1), code, codePos); // shift the dst left and put it into dst
		emit32( OR(dst, dst, tmp0), code, codePos); // Now or the two values together to get the ror
	}
#endif

	reg_changed_offset[instr.dst] = k;
	codePos = k;
}

void JitCompilerRV64::h_ISWAP_R(Instruction& instr, uint32_t& codePos)
{
	const uint32_t src = IntRegMap[instr.src];
	const uint32_t dst = IntRegMap[instr.dst];

	if (src == dst)
		return;

	uint32_t k = codePos;

	//todo: just swap indices in IntRegMap?  XOR trick?
	constexpr uint32_t tmp = 26;
	emit32( ADDI(tmp, dst, 0), code, k);
	emit32( ADDI(dst, src, 0), code, k);
	emit32 (ADDI(src, tmp, 0), code, k);

	reg_changed_offset[instr.src] = k;
	reg_changed_offset[instr.dst] = k;
	codePos = k;
}

//swap upper and lower doubles in dst
void JitCompilerRV64::h_FSWAP_R(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t dst = instr.dst + 16;
	const uint32_t src = instr.dst + 17;
	constexpr uint32_t tmp_fp = 29;


	emit32(FSGNJ_D(tmp_fp, dst, dst), code, k);
	emit32(FSGNJ_D(dst, src, src), code, k);
	emit32(FSGNJ_D(src, tmp_fp, tmp_fp), code, k);

	codePos = k;
}

void JitCompilerRV64::h_FADD_R(Instruction& instr, uint32_t& codePos)
{
	const uint32_t src = (instr.src % 4) + 24;
	const uint32_t dst = (instr.dst % 4) + 16;

	emit32( FADD_D(dst, dst, src, 7), code, codePos);
}

void JitCompilerRV64::h_FADD_M(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = IntRegMap[instr.src];
	const uint32_t dst = (instr.dst % 4) + 16;

	constexpr uint32_t tmp_fp = 29;
	emitMemLoadFP<tmp_fp>(src, instr, code, k);
	
	emit32( FADD_D(dst, dst, tmp_fp, 7), code, k);

	codePos = k;
}

void JitCompilerRV64::h_FSUB_R(Instruction& instr, uint32_t& codePos)
{
	const uint32_t src = (instr.src % 4) + 24;
	const uint32_t dst = (instr.dst % 4) + 16;

	emit32( FSUB_D(dst,dst,src, 7), code, codePos);
}

void JitCompilerRV64::h_FSUB_M(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = IntRegMap[instr.src];
	const uint32_t dst = (instr.dst % 4) + 16;

	constexpr uint32_t tmp_fp = 29;
	emitMemLoadFP<tmp_fp>(src, instr, code, k);
	
	emit32( FSUB_D(dst, dst, tmp_fp, 7), code, k);

	codePos = k;
}

void JitCompilerRV64::h_FSCAL_R(Instruction& instr, uint32_t& codePos)
{
	const uint32_t dst = (instr.dst % 4) + 16;
	constexpr uint32_t tmp0 = 26;
	constexpr uint32_t tmp1 = 27;
	// fp30 contains the scale mask
	emit32( FMV_X_D(tmp0, 30), code, codePos);
	emit32( FMV_X_D(tmp1, dst), code, codePos);
	emit32( XOR(tmp0,tmp0,tmp1), code, codePos);
	emit32( FMV_D_X(dst, tmp0), code, codePos);
	//XOR FP reg with 0x80F0000000000000  (const stored in reg 31)
	//emit32( FXOR_D(dst,dst,31), code, codePos);
}

void JitCompilerRV64::h_FMUL_R(Instruction& instr, uint32_t& codePos)
{
	const uint32_t src = (instr.src % 4) + 24;
	const uint32_t dst = (instr.dst % 4) + 20;

	emit32( FMUL_D(dst,dst,src, 7), code, codePos);
}

void JitCompilerRV64::h_FDIV_M(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = IntRegMap[instr.src];
	const uint32_t dst = (instr.dst % 4) + 20;

	constexpr uint32_t tmp_fp = 29;
	emitMemLoadFP<tmp_fp>(src, instr, code, k);


// Not sure if these are needed?  They look specific to ARM??
//***************************************************************
	// and tmp_fp, tmp_fp, and_mask_reg
//	emit32(FAND_D(tmp_fp, tmp_fp, 26), code, codePos);

	// orr tmp_fp, tmp_fp, or_mask_reg
//	emit32(FORR_D(tmp_fp, tmp_fp, 28), code, codePos);
//***************************************************************

	emit32( FDIV_D(dst, dst, tmp_fp, 7), code, k);

	codePos = k;
}

void JitCompilerRV64::h_FSQRT_R(Instruction& instr, uint32_t& codePos)
{
	const uint32_t dst = (instr.dst % 4) + 20;

	emit32(FSQRT_D(dst,dst,7), code, codePos);
}

void JitCompilerRV64::h_CBRANCH(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t dst = IntRegMap[instr.dst];
	const uint32_t modCond = instr.getModCond();
	const uint32_t shift = modCond + ConditionOffset;
	const uint32_t imm = (instr.getImm32() | (1U << shift)) & ~(1U << (shift - 1));
	constexpr uint32_t temp0 = 26;
	constexpr uint32_t temp1 = 27;

	emitAddImmediate(dst, dst, imm, code, k);

	// li temp0, ConditionMask
	emit32(ORI(temp0, 0, ConditionMask), code ,k);

	// Shift Left ConditionMask
	emit32(SLLI(temp0,temp0,shift), code, k);

	// Mask off the bits we need to check
	emit32(AND(temp0,dst,temp0), code, k);

	int32_t offset = reg_changed_offset[instr.dst];

	// Offset from current, this should already be on at least on a mulitple of 2 bytes
	offset = (offset - k) & ((1 << 13) - 1);

	emit32(BEQ(offset, 0, temp0), code, k);

	for (uint32_t i = 0; i < RegistersCount; ++i)
		reg_changed_offset[i] = k;

	codePos = k;
}

void JitCompilerRV64::h_CFROUND(Instruction& instr, uint32_t& codePos)
{
	//CFROUND is very rare - 1/256 instrs

	//Rounding Mode			RX	RISCV
	//roundTiesToEven		0	0
	//roundTowardNegative	1	2
	//roundTowardPositive	2	3
	//roundTowardZero		3	1
	constexpr uint8_t frmlut[4] = { 0, 64, 96, 32 };   //32b lookup table  for fcsr[7:0]   (frm in [6:5])

	uint32_t k = codePos;

	const uint32_t src = IntRegMap[instr.src];

	constexpr uint32_t temp0 = 26;
	constexpr uint32_t temp1 = 27;

	//rotate src by imm to get new rounding mode bits, mask off, scale upto LUT index
#ifdef BITMANIP
	emit32(((instr.getImm32()&63) << 20 | src << 15 | 0b101 << 12 | temp0 << 7 | 0b001001), code, k);  //emit32(RORI(tmp, src, rori_amt), code, k);  //rori macro oddly doesn't work
#else
	{
		uint32_t rori_amt;
		rori_amt = instr.getImm32() & 0x3f; // limit imm to 6 bits 
		emit32( ORI(temp0, 0, 64), code, k); // temp0 now has 64
		emit32( ORI(temp1, 0, rori_amt), code, k); // temp1 now has imm
		emit32( SUB(temp1, temp0, temp1), code, k); // temp1 now has 64 - imm
		emit32( SRLI(temp0, src, rori_amt), code, k); // shift the src right and put it into temp0
		emit32( SLL(temp1, src, temp1), code, k); // shift the src left and put it into dst
		emit32( OR(temp0, temp1, temp0), code, k); // Now or the two values together to get the ror
	}	
#endif
	emit32(ANDI(temp0, temp0, 0x3), code, k);
	emit32(SLLI(temp0, temp0, 0x3), code, k);

	//load immediate from lookup board, rotate to proper fcsr code, mask off
	emitAddImmediate(temp1, 0, *((uint32_t*)frmlut), code, k);
	emit32(SRL(temp1, temp1, temp0), code, k);
	emit32(ANDI(temp1, temp1, 0xFF), code, k);

	//write to fcsr
	emit32( CSRRW(0, FCSR, temp1), code, k); 	

	codePos = k;
}

void JitCompilerRV64::h_ISTORE(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = IntRegMap[instr.src];
	const uint32_t dst = IntRegMap[instr.dst];
	constexpr uint32_t tmp0 = 26;
	constexpr uint32_t tmp1 = 27;

	uint32_t imm = instr.getImm32();
	uint32_t mask_hi, mask_lo;

	if (instr.getModCond() < StoreL3Condition) { 
		if (instr.getModMem()) { //store to L1
			imm &= RANDOMX_SCRATCHPAD_L1 - 1;
			emitAddImmediate(tmp0, dst, imm, code, k);
			mask_lo = (RANDOMX_SCRATCHPAD_L1 - 8) & ((1 << 12) - 1);
			mask_hi = (RANDOMX_SCRATCHPAD_L1 - 8) >> 12;
			// Load the upper value
			emit32( LUI(tmp1, mask_hi), code, k );
			// Add in the lower value so we have the imm in temp1
			emit32(ADDI(tmp1, tmp1, mask_lo), code, k);
			emit32( AND(tmp0, tmp0, tmp1), code, k);
		}
		else { //store to L2
			imm &= RANDOMX_SCRATCHPAD_L2 - 1;
			emitAddImmediate(tmp0, dst, imm, code, k);
			mask_lo = (RANDOMX_SCRATCHPAD_L2 - 8) & ((1 << 12) - 1);
			mask_hi = (RANDOMX_SCRATCHPAD_L2 - 8) >> 12;
			// Load the upper value
			emit32( LUI(tmp1, mask_hi), code, k );
			// Add in the lower value so we have the imm in temp1
			emit32(ADDI(tmp1, tmp1, mask_lo), code, k);
			emit32( AND(tmp0, tmp0, tmp1), code, k);
		}
	}
	else { //store to L3
		imm &= RANDOMX_SCRATCHPAD_L3 - 1;
		emitAddImmediate(tmp0, dst, imm, code, k);
		mask_lo = (RANDOMX_SCRATCHPAD_L3 - 8) & ((1 << 12) - 1);
		mask_hi = (RANDOMX_SCRATCHPAD_L3 - 8) >> 12;
		// Load the upper value
		emit32( LUI(tmp1, mask_hi), code, k );
		// Add in the lower value so we have the imm in temp1
		emit32(ADDI(tmp1, tmp1, mask_lo), code, k);
		emit32( AND(tmp0, tmp0, tmp1), code, k);
	}

	emit32(ADD(tmp0, tmp0, 6), code, k);  //add scatchpad ptr from x6
	emit32(SD(src, tmp0, 0), code, k);

	codePos = k;
}

void JitCompilerRV64::h_NOP(Instruction& instr, uint32_t& codePos)
{
}

//--JUMP TABLE to h_INSTR emmission functions by array index-------------------
#include "instruction_weights.hpp"
#define INST_HANDLE(x) REPN(&JitCompilerRV64::h_##x, WT(x))

	InstructionGeneratorRV64 JitCompilerRV64::opTable[256] = {
		INST_HANDLE(IADD_RS)
		INST_HANDLE(IADD_M)
		INST_HANDLE(ISUB_R)
		INST_HANDLE(ISUB_M)
		INST_HANDLE(IMUL_R)
		INST_HANDLE(IMUL_M)
		INST_HANDLE(IMULH_R)
		INST_HANDLE(IMULH_M)
		INST_HANDLE(ISMULH_R)
		INST_HANDLE(ISMULH_M)
		INST_HANDLE(IMUL_RCP)
		INST_HANDLE(INEG_R)
		INST_HANDLE(IXOR_R)
		INST_HANDLE(IXOR_M)
		INST_HANDLE(IROR_R)
		INST_HANDLE(IROL_R)
		INST_HANDLE(ISWAP_R)
		INST_HANDLE(FSWAP_R)
		INST_HANDLE(FADD_R)
		INST_HANDLE(FADD_M)
		INST_HANDLE(FSUB_R)
		INST_HANDLE(FSUB_M)
		INST_HANDLE(FSCAL_R)
		INST_HANDLE(FMUL_R)
		INST_HANDLE(FDIV_M)
		INST_HANDLE(FSQRT_R)
		INST_HANDLE(CBRANCH)
		INST_HANDLE(CFROUND)
		INST_HANDLE(ISTORE)
		INST_HANDLE(NOP)
	};
}
