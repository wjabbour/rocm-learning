# ==========================================
# ROCm Inference Study - Build System
# ==========================================

# Compiler
CXX := hipcc

# -------------------------------------------------------------------------
# Dynamic Source Selection
# -------------------------------------------------------------------------
# Default Source: The kernel used if you just type 'make' or 'make run'
# You can override this via command line: make run SRC=src/kernels/gemm/v0_tiled_gemm/kernel.hip.cpp
SRC ?= src/kernels/reduction/v0_naive_pairwise/kernel.hip.cpp

# Output Handling
# 1. Strip directory path (src/kernels/.../kernel.hip.cpp -> kernel.hip.cpp)
# 2. Strip extension (kernel.hip.cpp -> kernel.hip)
# 3. Save binary to a 'build' folder so root dir stays clean
BUILD_DIR := build
TARGET_NAME := $(basename $(notdir $(SRC)))
TARGET := $(BUILD_DIR)/$(TARGET_NAME)

# -------------------------------------------------------------------------
# Compilation Flags
# -------------------------------------------------------------------------
# -Isrc  : Look for headers in src/ (Handles src/utils/...)
CXXFLAGS := -O3 -std=c++17 -Wall -Wextra -Isrc

# Target Architecture
# "native" tells hipcc to query the GPU and compile specifically for its GFX version
ARCH_FLAGS := --offload-arch=native

# Headers (Global dependency check)
# Automatically detects ANY header change in src/utils/
HEADERS := $(wildcard src/utils/*.hpp)

# -------------------------------------------------------------------------
# Rules
# -------------------------------------------------------------------------

# Default Target
all: $(TARGET)

# Build Rule
$(TARGET): $(SRC) $(HEADERS)
	@mkdir -p $(BUILD_DIR)
	@echo "------------------------------------------------"
	@echo "Compiling: $(SRC)"
	@echo "Output:    $(TARGET)"
	@echo "Arch Flag: $(ARCH_FLAGS)"
	@echo "------------------------------------------------"
	$(CXX) $(CXXFLAGS) $(ARCH_FLAGS) $(SRC) -o $(TARGET)

# Execution Rule
run: $(TARGET)
	@echo "Running $(TARGET)..."
	@./$(TARGET)

# Clean Rule (Removes the entire build directory)
clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean run