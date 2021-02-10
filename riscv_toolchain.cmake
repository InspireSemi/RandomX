set(CMAKE_SYSTEM_NAME Linux)  # Tell CMake we're cross-compiling
set(CMAKE_SYSTEM_PROCESSOR riscv)
set(ARCH_ID riscv)

set(CMAKE_C_COMPILER riscv64-unknown-linux-gnu-gcc)
#set(CMAKE_C_COMPILER riscv64-unknown-elf-gcc)

set(CMAKE_FIND_LIBRARY_SUFFIXES .a)
set(BUILD_SHARED_LIBS OFF)
set(CMAKE_EXE_LINKER_FLAGS -static)
