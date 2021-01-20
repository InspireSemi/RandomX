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
#define JAL(rd, imm)      imm<<12 | rd<<7 | 0b1101111 //FIXME: slice and dice imm
#define BLT(rs1,rs2, imm) imm<<25 | rs2<<20 | rs1<<15 | 0b100<<12 | 0b1100011 //FIXME: slice and dice imm

#define ADD(rd,rs1,rs2)   0b0000000<<25 | rs2<<20 | rs1<<15 | 0b000<<12 | rd<<7 | 0b0110011
#define SUB(rd,rs1,rs2)   0b0100000<<25 | rs2<<20 | rs1<<15 | 0b000<<12 | rd<<7 | 0b0110011
#define SRL(rd,rs1,rs2)   0b0000000<<25 | rs2<<20 | rs1<<15 | 0b101<<12 | rd<<7 | 0b0110011
#define AND(rd,rs1,rs2)   0b0000000<<25 | rs2<<20 | rs1<<15 | 0b111<<12 | rd<<7 | 0b0110011
#define XOR(rd,rs1,rs2)   0b0000000<<25 | rs2<<20 | rs1<<15 | 0b100<<12 | rd<<7 | 0b0110011
#define MUL(rd,rs1,rs2)   0b0000001<<25 | rs2<<20 | rs1<<15 | 0b000<<12 | rd<<7 | 0b0110011
#define MULHU(rd,rs1,rs2) 0b0000001<<25 | rs2<<20 | rs1<<15 | 0b011<<12 | rd<<7 | 0b0110011
#define MULHS(rd,rs1,rs2) 0b0000001<<25 | rs2<<20 | rs1<<15 | 0b001<<12 | rd<<7 | 0b0110011

#define ROR(rd,rs1,rs2)   0b0110000<<25 | rs2<<20 | rs1<<15 | 0b101<<12 | rd<<7 | 0b0110011
#define ROL(rd,rs1,rs2)   0b0110000<<25 | rs2<<20 | rs1<<15 | 0b001<<12 | rd<<7 | 0b0110011

#define CSRRW(rd,csr,rs1)                 csr<<20 | rs1<<15 | 0b001<<12 | rd<<7 | 0b1110011
#define FCSR 0  //FIXME right csr index

#define ADDI(rd,rs1,imm)  imm<<20 | rs1<<15 | 0b000<<12 | rd<<7 | 0b0010011
#define ANDI(rd,rs1,imm)  imm<<20 | rs1<<15 | 0b111<<12 | rd<<7 | 0b0010011
#define SLLI(rd,rs1,imm)  0b00000<<27 | imm<<20 | rs1<<15 | 0b001<<12 | rd<<7 | 0b0010011

#define RORI(rd,rs1,imm)  0b01100<<27 | imm<<20 | rs1<<15 | 0b101<<12 | rd<<7 | 0b0010011


#define LD(rd,rs1,imm)    imm<<20           | rs1<<15 | 0b011<<12 | rd<<7  | 0b0000011
#define LW(rd,rs1,imm)    imm<<20           | rs1<<15 | 0b010<<12 | rd<<7  | 0b0000011
#define SD(rs2,rs1,imm)   imm<<25 | rs2<<20 | rs1<<15 | 0b011<<12 | imm<<7 | 0b0100011  //fixme: slice imm


#define FADD_D(rd,rs1,rs2,rm) 0b0000001<<25 | rs2<<20 | rs1<<15 | rm<<12 | rd<<7 | 0b1010011
#define FSUB_D(rd,rs1,rs2,rm) 0b0000101<<25 | rs2<<20 | rs1<<15 | rm<<12 | rd<<7 | 0b1010011
#define FMUL_D(rd,rs1,rs2,rm) 0b0001001<<25 | rs2<<20 | rs1<<15 | rm<<12 | rd<<7 | 0b1010011
#define FDIV_D(rd,rs1,rs2,rm) 0b0001101<<25 | rs2<<20 | rs1<<15 | rm<<12 | rd<<7 | 0b1010011
#define FSQRT_D(rd,rs1,rm)    0b0101101<<25 | rs1<<15 |           rm<<12 | rd<<7 | 0b1010011
#define FCVTDW(rd,rs1,rm)     0b1101001<<25 | 0b0<<20 | rs1<<15 | rm<<12 | rd<<7 | 0b1010011
#define FSGNJ_D(rd,rs1,rs2)   0b0010001<<25 | rs2<<20 | rs1<<15 | 0b000<<12 | rd<<7 | 0b1010011

//custom F reg bitwise logic - pack with SGNJ, upper funct3 range.
#define FXOR_D(rd,rs1,rs2)    0b0010001<<25 | rs2<<20 | rs1<<15 | 0b101<<12 | rd<<7 | 0b1010011 //fixme: Add instr
#define FORR_D(rd,rs1,rs2)    0b0010001<<25 | rs2<<20 | rs1<<15 | 0b110<<12 | rd<<7 | 0b1010011 //fixme: Add instr
#define FAND_D(rd,rs1,rs2)    0b0010001<<25 | rs2<<20 | rs1<<15 | 0b111<<12 | rd<<7 | 0b1010011 //fixme: Add instr

//New instructions: FXOR_D, ROR, ROL, RORI

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

constexpr uint32_t IntRegMap[8] = { 4, 5, 6, 7, 12, 13, 14, 15 }; //TODO: may become variable for SW speculation; and may rename for swaps

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

	// and spAddr0, spMix1, ScratchpadL3Mask64
	emit32( ANDI(16, 10, (Log2(RANDOMX_SCRATCHPAD_L3) - 7)), code, codePos);  //overwrites placeholder in asm

	// and spAddr0, temp0, ScratchpadL3Mask64
	emit32( ANDI(17, 18, (Log2(RANDOMX_SCRATCHPAD_L3) - 7)), code, codePos); //overwrites placeholder in asm

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

	// Update spMix2
	// eor w18, config.readReg2, config.readReg3
	emit32( XOR(18, IntRegMap[config.readReg2], IntRegMap[config.readReg3]), code, codePos);

	// Jump back to the main loop
	const uint32_t offset = (((uint8_t*)randomx_program_rv64_vm_instructions_end) - ((uint8_t*)randomx_program_rv64)) - codePos;
	emit32(JAL(0, offset), code, codePos);

	//insert masks

	// and w18, w18, CacheLineAlignMask
	codePos = (((uint8_t*)randomx_program_rv64_cacheline_align_mask1) - ((uint8_t*)randomx_program_rv64));
	emit32(ANDI(18, 9, (Log2(RANDOMX_DATASET_BASE_SIZE) - 7)), code, codePos);

	// and w10, w10, CacheLineAlignMask
	codePos = (((uint8_t*)randomx_program_rv64_cacheline_align_mask2) - ((uint8_t*)randomx_program_rv64));
	emit32(ANDI(10, 9, (Log2(RANDOMX_DATASET_BASE_SIZE) - 7)), code, codePos);

	// Update spMix1
	// eor x10, config.readReg0, config.readReg1
	codePos = ((uint8_t*)randomx_program_rv64_update_spMix1) - ((uint8_t*)randomx_program_rv64);
	emit32( XOR(10, IntRegMap[config.readReg0], IntRegMap[config.readReg1]), code, codePos);

//#ifdef __GNUC__
//	__builtin___clear_cache(reinterpret_cast<char*>(code + MainLoopBegin), reinterpret_cast<char*>(code + codePos));
//#endif
}

void JitCompilerRV64::generateProgramLight(Program& program, ProgramConfiguration& config, uint32_t datasetOffset)
{
	uint32_t codePos = MainLoopBegin + 4;

	// and w16, w10, ScratchpadL3Mask64
	emit32(0x121A0000 | 16 | (10 << 5) | ((Log2(RANDOMX_SCRATCHPAD_L3) - 7) << 10), code, codePos);

	// and w17, w18, ScratchpadL3Mask64
	emit32(0x121A0000 | 17 | (18 << 5) | ((Log2(RANDOMX_SCRATCHPAD_L3) - 7) << 10), code, codePos);

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
		(this->*opTable[instr.opcode])(instr, codePos);//Jump to h_INSTR emmision function by opcode
	}

	// Update spMix2
	// eor w18, config.readReg2, config.readReg3
	emit32( XOR(18, IntRegMap[config.readReg2], IntRegMap[config.readReg3]), code, codePos);

	// Jump back to the main loop
	const uint32_t offset = (((uint8_t*)randomx_program_rv64_vm_instructions_end_light) - ((uint8_t*)randomx_program_rv64)) - codePos;
	emit32(JAL(0,offset/4), code, codePos); //FIXME: proper offset

	// and w2, w9, CacheLineAlignMask
	codePos = (((uint8_t*)randomx_program_rv64_light_cacheline_align_mask) - ((uint8_t*)randomx_program_rv64));
	emit32(ANDI(2, 9, (Log2(RANDOMX_DATASET_BASE_SIZE) - 7)), code, codePos);

	// Update spMix1
	// eor x10, config.readReg0, config.readReg1
	codePos = ((uint8_t*)randomx_program_rv64_update_spMix1) - ((uint8_t*)randomx_program_rv64);
	emit32( XOR(10, IntRegMap[config.readReg0], IntRegMap[config.readReg1]), code, codePos);

	// Apply dataset offset
	codePos = ((uint8_t*)randomx_program_rv64_light_dataset_offset) - ((uint8_t*)randomx_program_rv64);

	datasetOffset /= CacheLineSize;
	const uint32_t imm_lo = datasetOffset & ((1 << 12) - 1);
	const uint32_t imm_hi = datasetOffset >> 12;

	emit32(ADDI(2,2,imm_lo), code, codePos); //ARMV8A::ADD_IMM_LO | 2 | (2 << 5) | (imm_lo << 10)
	emit32(ADDI(2,2,imm_hi), code, codePos); //ARMV8A::ADD_IMM_HI | 2 | (2 << 5) | (imm_hi << 10)

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
	memcpy(code + codePos, p1, p2 - p1);
	codePos += p2 - p1;

	num32bitLiterals = 64;
	constexpr uint32_t tmp = 12;

	for (size_t i = 0; i < N; ++i)
	{
		// and x11, x10, CacheSize / CacheLineSize - 1
		emit32(0x92400000 | 11 | (10 << 5) | ((Log2(CacheSize / CacheLineSize) - 1) << 10), code, codePos);

		p1 = ((uint8_t*)randomx_calc_dataset_item_rv64_prefetch) + 4;
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
		emit32(JAL(0, (codePos - jmp_pos)/4), code, codePos);  //fixme : jump offset (omit /4)? or /2?

		for (size_t j = 0; j < progSize; ++j)
		{
			const Instruction& instr = prog(j);
			const uint32_t src = instr.src;
			const uint32_t dst = instr.dst;

			switch (static_cast<SuperscalarInstructionType>(instr.opcode))
			{
			case randomx::SuperscalarInstructionType::ISUB_R:
				emit32( SUB(dst,dst,src), code, codePos);
				break;
			case randomx::SuperscalarInstructionType::IXOR_R:
				emit32( XOR(dst, dst, src), code, codePos);
				break;
			case randomx::SuperscalarInstructionType::IADD_RS:
				emit32(SLLI(tmp, src, instr.getModShift()), code, codePos);
				emit32(ADD(dst, dst, tmp), code, codePos);
				break;
			case randomx::SuperscalarInstructionType::IMUL_R:
				emit32( MUL(dst, dst, src), code, codePos);
				break;
			case randomx::SuperscalarInstructionType::IROR_C:
				emit32( RORI(dst, dst, (instr.getImm32()&63)), code, codePos);
				break;
			case randomx::SuperscalarInstructionType::IADD_C7:
			case randomx::SuperscalarInstructionType::IADD_C8:
			case randomx::SuperscalarInstructionType::IADD_C9:
				emitAddImmediate(dst, dst, instr.getImm32(), code, codePos);
				break;
			case randomx::SuperscalarInstructionType::IXOR_C7:
			case randomx::SuperscalarInstructionType::IXOR_C8:
			case randomx::SuperscalarInstructionType::IXOR_C9:
				emitMovImmediate(tmp, instr.getImm32(), code, codePos);
				emit32(XOR(dst, dst, tmp), code, codePos);
				break;
			case randomx::SuperscalarInstructionType::IMULH_R:
				emit32( MULHU(dst, dst, src), code, codePos);
				break;
			case randomx::SuperscalarInstructionType::ISMULH_R:
				emit32( MULHS(dst, dst, src), code, codePos);
				break;
			case randomx::SuperscalarInstructionType::IMUL_RCP:
				{
					int32_t offset = (literal_pos - codePos) / 4;
					offset &= (1 << 19) - 1;
					literal_pos += 8;
					// load 32b immediate reciprocal
					emit32(LW(tmp, offset, 0), code, codePos);  

					// mul dst, dst, tmp
					emit32( MUL(dst,dst,tmp), code, codePos);
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
		emit32(ADDI(10, prog.getAddressRegister(), 0), code, codePos);
	}

	p1 = (uint8_t*)randomx_calc_dataset_item_rv64_store_result;
	p2 = (uint8_t*)randomx_calc_dataset_item_rv64_end;
	memcpy(code + codePos, p1, p2 - p1);
	codePos += p2 - p1;

}

template void JitCompilerRV64::generateSuperscalarHash(SuperscalarProgram(&programs)[RANDOMX_CACHE_ACCESSES], std::vector<uint64_t> &reciprocalCache);

DatasetInitFunc* JitCompilerRV64::getDatasetInitFunc()
{
	return (DatasetInitFunc*)(code + (((uint8_t*)randomx_init_dataset_rv64) - ((uint8_t*)randomx_program_rv64)));
}

size_t JitCompilerRV64::getCodeSize()
{
	return CodeSize;
}

void JitCompilerRV64::emitMovImmediate(uint32_t dst, uint32_t imm, uint8_t* code, uint32_t& codePos)
{
	uint32_t k = codePos;

	emit32(ADDI(dst, 0, imm), code, k);
	emit32( LUI(dst, imm), code, k);

	codePos = k;
}

void JitCompilerRV64::emitAddImmediate(uint32_t dst, uint32_t src, uint32_t imm, uint8_t* code, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t imm_lo = imm & ((1 << 12) - 1);
	const uint32_t imm_hi = imm >> 12;

	if (imm_hi)
	{
		constexpr uint32_t tmp = 18;
		emitMovImmediate(tmp, imm, code, k);

		// add dst, src, tmp
		emit32( ADD(dst, src, tmp), code, k);
	}
	else if (imm_lo)
	{
		emit32(ADDI(dst,src,imm_lo), code, k);
	}

	codePos = k;
}

// dst <= mem[src+imm32]
//have to ADD imm, AND with mask manually
//make an LD with mask built in?
template<uint32_t tmp>
void JitCompilerRV64::emitMemLoad(uint32_t dst, uint32_t src, Instruction& instr, uint8_t* code, uint32_t& codePos)
{
	uint32_t k = codePos;

	uint32_t imm = instr.getImm32();

	if (src != dst) //load from L1 or L2
	{
		if (instr.getModMem()) { //Mod.mem != 0, load from L1
			imm &= (RANDOMX_SCRATCHPAD_L1 - 1);
			emitAddImmediate(tmp, src, imm, code, k); 
			emit32(ANDI(tmp, tmp, (Log2(RANDOMX_SCRATCHPAD_L1) - 4)), code, k);
		}
		else { //Mod.mem==0, load from L2
			imm &= (RANDOMX_SCRATCHPAD_L2 - 1);
			emitAddImmediate(tmp, src, imm, code, k);
			emit32(ANDI(tmp, tmp, (Log2(RANDOMX_SCRATCHPAD_L2) - 4)), code, k);
		}
	}
	else //src==dst, load from L3 scratch range
	{
		imm = (imm & ScratchpadL3Mask) >> 3;
		emitMovImmediate(tmp, imm, code, k);
	}

	// ldr tmp, [x2, tmp]  //(0xf8606840 | tmp | (tmp << 16)
	emit32(LD(dst, tmp, 0), code, k);

	codePos = k;
}

template<uint32_t tmp_fp>
void JitCompilerRV64::emitMemLoadFP(uint32_t src, Instruction& instr, uint8_t* code, uint32_t& codePos)
{
	uint32_t k = codePos;

	uint32_t imm = instr.getImm32();
	constexpr uint32_t tmp = 18;
	
	//Address calculation
	//loadfp only from L1 or L2; only loads mem src operands, so src=dst not sensible; src is an integer reg
	if (instr.getModMem()) { //Mod.mem != 0, load from L1
		imm &= (RANDOMX_SCRATCHPAD_L1 - 1);
		emitAddImmediate(tmp, src, imm, code, k);
		emit32(ANDI(tmp, tmp, (Log2(RANDOMX_SCRATCHPAD_L1) - 4)), code, k);
	}
	else { //Mod.mem==0, load from L2
		imm &= (RANDOMX_SCRATCHPAD_L2 - 1);
		emitAddImmediate(tmp, src, imm, code, k);
		emit32(ANDI(tmp, tmp, (Log2(RANDOMX_SCRATCHPAD_L2) - 4)), code, k);
	}
	
	emit32(LW(tmp, tmp, 0), code, k); //load lower word to int reg for conversion
	emit32(FCVTDW(tmp_fp, tmp, 7), code, k); //convert int32 to double

	emit32(LW(tmp, tmp, 4), code, k); //load upper word
	emit32(FCVTDW(tmp_fp+1, tmp, 7), code, k); //convert int32 to double

	//caller will complete conversion masking (different for mul or add/sub)

	codePos = k;
}


void JitCompilerRV64::h_IADD_RS(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = IntRegMap[instr.src];
	const uint32_t dst = IntRegMap[instr.dst];
	const uint32_t shift = instr.getModShift();

	constexpr uint32_t tmp = 18;

	// add dst, src << shift
	emit32(SLLI(tmp, src, shift), code, k);
	emit32(ADD(dst, dst, tmp), code, k);

	if (instr.dst == RegisterNeedsDisplacement)
		emitAddImmediate(dst, dst, instr.getImm32(), code, k);

	reg_changed_offset[instr.dst] = k;
	codePos = k;
}

void JitCompilerRV64::h_IADD_M(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = IntRegMap[instr.src];
	const uint32_t dst = IntRegMap[instr.dst];

	constexpr uint32_t tmp = 18;
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

	if (src != dst)
	{
		// sub dst, dst, src
		emit32( SUB(dst,dst,src), code, k);
	}
	else //src==dst, subtract immediate instead (else get zero(
	{
		emitAddImmediate(dst, dst, -instr.getImm32(), code, k);
	}

	reg_changed_offset[instr.dst] = k;
	codePos = k;
}

void JitCompilerRV64::h_ISUB_M(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = IntRegMap[instr.src];
	const uint32_t dst = IntRegMap[instr.dst];

	constexpr uint32_t tmp = 18;
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
		src = 18;
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

	constexpr uint32_t tmp = 18;
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

	constexpr uint32_t tmp = 18;
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

	constexpr uint32_t tmp = 18;
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

	constexpr uint32_t tmp = 18;
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
		static constexpr uint32_t literal_regs[13] = { 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20 };

		// mul dst, dst, literal_reg
		emit32(MUL(dst, dst, literal_regs[literal_id]), code, k);
	}
	else
	{
		// ldr tmp, reciprocal
		const uint32_t offset = (literalPos - k) / 4;
		emit32( LD(tmp, 0, offset), code, k);

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
		src = 18;
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

	constexpr uint32_t tmp = 18;
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

	reg_changed_offset[instr.dst] = codePos;
}

void JitCompilerRV64::h_IROL_R(Instruction& instr, uint32_t& codePos)
{
	uint32_t k = codePos;

	const uint32_t src = IntRegMap[instr.src];
	const uint32_t dst = IntRegMap[instr.dst];
	uint32_t rori_amt;

	if (src != dst)
	{
		constexpr uint32_t tmp = 18;

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
	constexpr uint32_t tmp = 18;
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
	constexpr uint32_t tmp_fp = 28;


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

	constexpr uint32_t tmp_fp = 28;
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

	constexpr uint32_t tmp_fp = 28;
	emitMemLoadFP<tmp_fp>(src, instr, code, k);
	
	emit32( FSUB_D(dst, dst, tmp_fp, 7), code, k);

	codePos = k;
}

void JitCompilerRV64::h_FSCAL_R(Instruction& instr, uint32_t& codePos)
{
	const uint32_t dst = (instr.dst % 4) + 16;
	//XOR FP reg with 0x80F0000000000000  (const stored in reg 31)
	emit32( FXOR_D(dst,dst,31), code, codePos);
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

	constexpr uint32_t tmp_fp = 28;
	emitMemLoadFP<tmp_fp>(src, instr, code, k);

	// and tmp_fp, tmp_fp, and_mask_reg
	emit32(FAND_D(tmp_fp, tmp_fp, 26), code, codePos);

	// orr tmp_fp, tmp_fp, or_mask_reg
	emit32(FORR_D(tmp_fp, tmp_fp, 28), code, codePos);

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

	emitAddImmediate(dst, dst, imm, code, k);

	// tst dst, mask
	static_assert((ConditionMask == 0xFF) && (ConditionOffset == 8), "Update tst encoding for different mask and offset");
	emit32((0xF2781C1F - (modCond << 16)) | (dst << 5), code, k);

	int32_t offset = reg_changed_offset[instr.dst];
	offset = ((offset - k) >> 2) & ((1 << 19) - 1);

	// beq target
	emit32(0x54000000 | (offset << 5), code, k);

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

	constexpr uint32_t temp0 = 18;
	constexpr uint32_t temp1 = 19;

	//rotate src by imm to get new rouding mode bits, mask off, scale upto LUT index
	emit32(((instr.getImm32()&63) << 20 | src << 15 | 0b101 << 12 | temp0 << 7 | 0b001001), code, k);  //emit32(RORI(tmp, src, rori_amt), code, k);  //rori macro oddly doesn't work
	emit32(ANDI(temp0, temp0, 0x3), code, k);
	emit32(SLLI(temp0, temp0, 0x3), code, k);

	//load immediate frm lookup board, rotate to proper fcsr code, mask off
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
	constexpr uint32_t tmp = 18;

	uint32_t imm = instr.getImm32();

	if (instr.getModCond() < StoreL3Condition) { 
		if (instr.getModMem()) { //store to L1
			imm &= RANDOMX_SCRATCHPAD_L1 - 1;
			emitAddImmediate(tmp, dst, imm, code, k);
			emit32(ANDI(tmp, tmp, (Log2(RANDOMX_SCRATCHPAD_L1) - 4)), code, k);
		}
		else { //store to L2
			imm &= RANDOMX_SCRATCHPAD_L2 - 1;
			emitAddImmediate(tmp, dst, imm, code, k);
			emit32(ANDI(tmp, tmp, (Log2(RANDOMX_SCRATCHPAD_L2) - 4)), code, k);
		}
	}
	else { //store to L3
		imm &= RANDOMX_SCRATCHPAD_L3 - 1;
		emitAddImmediate(tmp, dst, imm, code, k);
		emit32(ANDI(tmp, tmp, (Log2(RANDOMX_SCRATCHPAD_L3) - 4)), code, k);
	}

	emit32(ADD(tmp, tmp, 2), code, k);  //add scatchpad ptr from x2 (todo: eliminate, use translation?)
	emit32(SD(src, tmp, 0), code, k);

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
