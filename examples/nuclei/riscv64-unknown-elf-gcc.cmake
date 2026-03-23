set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR riscv)

set(RISCV_TOOLCHAIN_PREFIX
    "riscv64-unknown-elf"
    CACHE STRING "GNU toolchain prefix for the RISC-V baremetal toolchain"
)

set(CMAKE_C_COMPILER "${RISCV_TOOLCHAIN_PREFIX}-gcc" CACHE STRING "C compiler")
set(CMAKE_CXX_COMPILER "${RISCV_TOOLCHAIN_PREFIX}-g++" CACHE STRING "C++ compiler")
set(CMAKE_ASM_COMPILER "${RISCV_TOOLCHAIN_PREFIX}-gcc" CACHE STRING "ASM compiler")
set(CMAKE_AR "${RISCV_TOOLCHAIN_PREFIX}-ar" CACHE STRING "Archiver")
set(CMAKE_RANLIB "${RISCV_TOOLCHAIN_PREFIX}-ranlib" CACHE STRING "Ranlib")
set(CMAKE_STRIP "${RISCV_TOOLCHAIN_PREFIX}-strip" CACHE STRING "Strip")
set(CMAKE_LINKER "${RISCV_TOOLCHAIN_PREFIX}-ld" CACHE STRING "Linker")
set(CMAKE_OBJCOPY "${RISCV_TOOLCHAIN_PREFIX}-objcopy" CACHE STRING "Objcopy")
set(CMAKE_OBJDUMP "${RISCV_TOOLCHAIN_PREFIX}-objdump" CACHE STRING "Objdump")
set(CMAKE_SIZE "${RISCV_TOOLCHAIN_PREFIX}-size" CACHE STRING "Size")

set(CMAKE_EXECUTABLE_SUFFIX ".elf")
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Select C/C++ version
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

set(RISCV_ARCH "rv32imafdc" CACHE STRING "Target RISC-V ISA string")
set(RISCV_ABI "ilp32d" CACHE STRING "Target RISC-V ABI")
set(RISCV_MODEL "medany" CACHE STRING "Target RISC-V code model")
option(SEMIHOSTING "Link with semihosting support instead of nosys stubs" OFF)

add_compile_options(
  -march=${RISCV_ARCH}
  -mabi=${RISCV_ABI}
  -mcmodel=${RISCV_MODEL}
  -ffunction-sections
  -fdata-sections
  "$<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions;-fno-rtti;-fno-unwind-tables>"
)

add_compile_definitions("$<$<NOT:$<CONFIG:DEBUG>>:NDEBUG>")

add_link_options(
  -march=${RISCV_ARCH}
  -mabi=${RISCV_ABI}
  -mcmodel=${RISCV_MODEL}
  LINKER:--gc-sections
)

if(SEMIHOSTING)
  add_link_options(--specs=rdimon.specs)
else()
  add_link_options(--specs=nosys.specs)
endif()
