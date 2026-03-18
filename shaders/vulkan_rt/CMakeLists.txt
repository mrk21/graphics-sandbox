cmake_minimum_required(VERSION 3.22)

# Cross-compile for Windows with MinGW
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_C_COMPILER x86_64-w64-mingw32-gcc)
set(CMAKE_CXX_COMPILER x86_64-w64-mingw32-g++)
set(CMAKE_FIND_ROOT_PATH /usr/x86_64-w64-mingw32)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

project(vulkan_rt LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# Vulkan Headers
FetchContent_Declare(
    vulkan_headers
    URL https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/v1.4.346.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(vulkan_headers)

# GLM (math library)
FetchContent_Declare(
    glm
    URL https://github.com/g-truc/glm/archive/refs/tags/1.0.1.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(glm)

add_executable(vulkan_rt main.cpp)

target_include_directories(vulkan_rt PRIVATE
    ${vulkan_headers_SOURCE_DIR}/include
    ${glm_SOURCE_DIR}
)

# Static link MinGW runtime
target_link_options(vulkan_rt PRIVATE -static-libgcc -static-libstdc++)

# Link libraries
# vulkan-1: the Vulkan loader (vulkan-1.dll on Windows)
# To cross-compile, you need either:
#   - A vulkan-1.lib or libvulkan-1.a import library for MinGW
#   - Or use dlopen/GetProcAddress to load vulkan-1.dll at runtime
# For native Windows build with Vulkan SDK, this links automatically.
target_link_libraries(vulkan_rt PRIVATE
    glfw3
    gdi32
    user32
    shell32
    winmm
    imm32
)

# ---- Compile GLSL shaders to SPIR-V ----
# Ubuntu 24.04: apt install glslang-tools
find_program(GLSLANG_VALIDATOR NAMES glslangValidator REQUIRED)


set(SHADER_SOURCES
    ${CMAKE_SOURCE_DIR}/shaders/raygen.rgen
    ${CMAKE_SOURCE_DIR}/shaders/miss.rmiss
    ${CMAKE_SOURCE_DIR}/shaders/shadow.rmiss
    ${CMAKE_SOURCE_DIR}/shaders/closesthit.rchit
)

set(SPV_FILES "")
foreach(SHADER ${SHADER_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    set(SPV_FILE ${CMAKE_BINARY_DIR}/shaders/${SHADER_NAME}.spv)
    add_custom_command(
        OUTPUT ${SPV_FILE}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/shaders
        COMMAND ${GLSLANG_VALIDATOR}
            --target-env vulkan1.2
            -o ${SPV_FILE}
            ${SHADER}
        DEPENDS ${SHADER}
        COMMENT "Compiling ${SHADER_NAME} -> SPIR-V"
    )
    list(APPEND SPV_FILES ${SPV_FILE})
endforeach()

add_custom_target(compile_shaders ALL DEPENDS ${SPV_FILES})
add_dependencies(vulkan_rt compile_shaders)
