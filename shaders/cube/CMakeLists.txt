cmake_minimum_required(VERSION 3.22)

# Cross-compile for Windows with MinGW
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_C_COMPILER x86_64-w64-mingw32-gcc)
set(CMAKE_CXX_COMPILER x86_64-w64-mingw32-g++)
set(CMAKE_FIND_ROOT_PATH /usr/x86_64-w64-mingw32)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

project(cube_shader LANGUAGES C CXX)

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

add_executable(cube_shader main.cpp)

# Static link MinGW runtime to avoid libgcc/libstdc++ DLL dependency
target_link_options(cube_shader PRIVATE -static-libgcc -static-libstdc++)

# MinGW cross-compile: link Windows libraries directly
target_link_libraries(cube_shader PRIVATE
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
add_dependencies(cube_shader copy_shaders)
