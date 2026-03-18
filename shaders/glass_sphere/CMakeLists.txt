cmake_minimum_required(VERSION 3.22)

# Cross-compile for Windows with MinGW
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_C_COMPILER x86_64-w64-mingw32-gcc)
set(CMAKE_CXX_COMPILER x86_64-w64-mingw32-g++)
set(CMAKE_FIND_ROOT_PATH /usr/x86_64-w64-mingw32)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

project(glass_sphere LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Download and build GLEW from source
include(FetchContent)
FetchContent_Declare(
    glew
    URL https://github.com/nigels-com/glew/releases/download/glew-2.2.0/glew-2.2.0.tgz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(glew)

add_library(glew_static STATIC ${glew_SOURCE_DIR}/src/glew.c)
target_include_directories(glew_static PUBLIC ${glew_SOURCE_DIR}/include)
target_compile_definitions(glew_static PUBLIC GLEW_STATIC)

add_executable(glass_sphere main.cpp)

# Download stb_image.h (header-only image loader)
FetchContent_Declare(
    stb
    URL https://github.com/nothings/stb/archive/refs/heads/master.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(stb)
target_include_directories(glass_sphere PRIVATE ${stb_SOURCE_DIR})

# Static link MinGW runtime to avoid libgcc/libstdc++ DLL dependency
target_link_options(glass_sphere PRIVATE -static-libgcc -static-libstdc++)

# MinGW cross-compile: link Windows libraries directly
target_link_libraries(glass_sphere PRIVATE
    glew_static
    glfw3
    opengl32
    gdi32
    user32
    shell32
    winmm
    imm32
)

# Shader files - track as dependencies so changes trigger rebuild
file(GLOB SHADER_FILES "${CMAKE_SOURCE_DIR}/shaders/*.glsl")

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/shaders/.stamp
    DEPENDS ${SHADER_FILES}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/shaders"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/shaders"
        "${CMAKE_BINARY_DIR}/shaders"
    COMMAND ${CMAKE_COMMAND} -E touch "${CMAKE_BINARY_DIR}/shaders/.stamp"
    COMMENT "Copying shader files..."
)
add_custom_target(copy_shaders ALL DEPENDS ${CMAKE_BINARY_DIR}/shaders/.stamp)
add_dependencies(glass_sphere copy_shaders)

# Texture files - copy to build directory
file(GLOB TEXTURE_FILES "${CMAKE_SOURCE_DIR}/../../textures/sand_block_256x256.png")

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/textures/.stamp
    DEPENDS ${TEXTURE_FILES}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/textures"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_SOURCE_DIR}/../../textures/sand_block_256x256.png"
        "${CMAKE_BINARY_DIR}/textures/sand_block_256x256.png"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_SOURCE_DIR}/../../textures/sand_block_256x256_normal.png"
        "${CMAKE_BINARY_DIR}/textures/sand_block_256x256_normal.png"
    COMMAND ${CMAKE_COMMAND} -E touch "${CMAKE_BINARY_DIR}/textures/.stamp"
    COMMENT "Copying texture files..."
)
add_custom_target(copy_textures ALL DEPENDS ${CMAKE_BINARY_DIR}/textures/.stamp)
add_dependencies(glass_sphere copy_textures)
