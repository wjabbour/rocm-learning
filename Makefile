# ==========================================
# ROCm Inference Study - Build System
# ==========================================

# Compiler: hipcc is the AMD wrapper that handles mixing C++ and Device code
CXX := hipcc

# Compilation Flags
# -O3:        Maximize optimization (aggressive loop unrolling, inlining)
# -std=c++17: Modern C++ features (structured bindings, etc.)
# -Wall:      Enable all warnings (catch bugs early)
CXXFLAGS := -O3 -std=c++17 -Wall -Wextra -I. -Isrc

# Target Architecture
# "native" tells hipcc to query the GPU and compile specifically for its GFX version.
# This ensures the latest RDNA instruction sets available to the hardware are used
ARCH_FLAGS := --offload-arch=native

# Project Files
TARGET := kernel
SRC := src/kernels/reduction/v0_naive_pairwise/kernel.hip.cpp
# List headers here so Make knows to re-compile
HEADERS := src/utils/hip_check.hpp src/utils/random_int.hpp

# Default Target
all: $(TARGET)

# Build Rule
$(TARGET): $(SRC) $(HEADERS)
	@echo "Building for generic host + specific device architecture..."
	$(CXX) $(CXXFLAGS) $(ARCH_FLAGS) $(SRC) -o $(TARGET)
	@echo "Build complete: ./$(TARGET)"

# Clean Rule
clean:
	rm -f $(TARGET)

# Execution Rule (Convenience)
run: $(TARGET)
	./$(TARGET)

# Phony targets prevent confusion with files named "clean" or "run"
.PHONY: all clean run