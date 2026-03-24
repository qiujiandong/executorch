TARGET = nuclei_example

CORE ?= n900fd

ARCH_EXT ?=

SRCDIRS = .

INCDIRS = . ../../cmake-out-nuclei/include ../../.. ../../runtime/core/portable_type/c10

COMMON_FLAGS := -O2 -DC10_USING_CUSTOM_GENERATED_MACROS -DET_BUNDLE_IO -DET_LOG_DUMP_OUTPUT

LIBDIRS := ../../cmake-out-nuclei/lib

DOWNLOAD ?= ddr

STDCLIB ?= newlib_small

LDLIBS := -lexecutorch -Wl,--whole-archive -lquantized_ops_lib -lnuclei_ops_lib -lportable_ops_lib -Wl,--no-whole-archive -lquantized_kernels -lnuclei_kernels -lNMSISNN -lportable_kernels -lextension_runner_util -lexecutorch_core -lbundled_program -lm

include $(NUCLEI_SDK_ROOT)/Build/Makefile.base

