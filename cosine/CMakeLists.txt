# Use modern CMake version, 3.10 or higher is recommended
cmake_minimum_required(VERSION 3.10) 
project(hnsw_lib_project CXX)

## --- 1. Standard and Extension Configuration ---
# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Disable GNU extensions to ensure code complies with standard C++ specifications
set(CMAKE_CXX_EXTENSIONS OFF) 

# Source file definitions
set(SOURCE_EXE main.cpp)
set(SOURCE_LIB pag.cpp)

## --- 2. Compilation Flags and Optimization ---

# Unified optimization and architecture flags
set(COMMON_COMPILE_FLAGS 
    -Ofast             # Aggressive optimization
    -march=native      # Enable all instruction sets supported by the current CPU (including basic AVX/AVX512)
    -fPIC              # Generate position-independent code (used for shared libraries)
    -w                 # Suppress all warnings
)

# Force add complete AVX-512 extension flags
# This is the key to resolving the missing declaration of the specific non-standard intrinsic (_mm512_extract_epi64)
set(AVX512_EXT_FLAGS 
    -mavx512f 
    -mavx512dq 
    -mavx512cd 
    -mavx512bw 
    -mavx512vl
    -mavx512vnni
)

## --- 3. OpenMP Configuration ---
# Find OpenMP package
find_package(OpenMP REQUIRED)

## --- 4. Library Configuration: pag ---
add_library(pag_cos STATIC ${SOURCE_LIB})

# Apply compilation options: general optimization + AVX-512 extensions
target_compile_options(pag_cos PRIVATE 
    ${COMMON_COMPILE_FLAGS} 
    ${AVX512_EXT_FLAGS}
)
# === New: Add macro definitions based on CMake variables ===
if(ENABLE_DEBUG_MODE)
    target_compile_definitions(pag_cos PRIVATE DEBUG_MODE)
endif()

if(ENABLE_FBIN_SUPPORT)
    target_compile_definitions(pag_cos PRIVATE FBIN_SUPPORT=1)
endif()

if(ENABLE_FULL_FBIN_LOAD)
    target_compile_definitions(pag_cos PRIVATE FULL_FBIN_LOAD=1)
endif()
if(WITHOUT_PES)
    target_compile_definitions(pag_cos PRIVATE WITHOUT_PES)
endif()
# ====================================
# Link OpenMP library and set compilation flags
if(OpenMP_CXX_FOUND)
    target_link_libraries(pag_cos PRIVATE OpenMP::OpenMP_CXX)
    # Add -fopenmp flag to compilation options
    target_compile_options(pag_cos PRIVATE ${OpenMP_CXX_FLAGS})
endif()

## --- 5. Executable Configuration: PAG ---
add_executable(PAG_cos ${SOURCE_EXE})
# --- Automatically modify output filename based on macro definitions ---
if(WITHOUT_PES)
    set_target_properties(PAG_cos PROPERTIES OUTPUT_NAME "PAG_cos_wopes")
else()
    set_target_properties(PAG_cos PROPERTIES OUTPUT_NAME "PAG_cos")
endif()
# Apply compilation options: consistent with the pag library
target_compile_options(PAG_cos PRIVATE 
    ${COMMON_COMPILE_FLAGS} 
    ${AVX512_EXT_FLAGS}
)

# === New: Add macro definitions for the executable as well (if there is code in main.cpp) ===
if(ENABLE_DEBUG_MODE)
    target_compile_definitions(PAG_cos PRIVATE DEBUG_MODE)
endif()

if(ENABLE_FBIN_SUPPORT)
    target_compile_definitions(PAG_cos PRIVATE FBIN_SUPPORT=1)
endif()

if(ENABLE_FULL_FBIN_LOAD)
    target_compile_definitions(PAG_cos PRIVATE FULL_FBIN_LOAD=1)
endif()
if(WITHOUT_PES)
    target_compile_definitions(PAG_cos PRIVATE WITHOUT_PES)
endif()

# Link pag library and OpenMP library
target_link_libraries(PAG_cos PRIVATE pag_cos)

if(OpenMP_CXX_FOUND)
    target_link_libraries(PAG_cos PRIVATE OpenMP::OpenMP_CXX)
    # Add -fopenmp flag to compilation options
    target_compile_options(PAG_cos PRIVATE ${OpenMP_CXX_FLAGS})
endif()