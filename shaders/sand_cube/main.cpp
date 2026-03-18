#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// ----- Math utilities (minimal, no GLM dependency) -----

struct Vec3 {
    float x, y, z;
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
};

static float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
static Vec3 normalize(Vec3 v) {
    float len = std::sqrt(dot(v, v));
    return (len > 1e-8f) ? v * (1.0f / len) : Vec3{0, 0, 0};
}

// Column-major 4x4 matrix
struct Mat4 {
    float m[16]{};
    float& operator()(int row, int col) { return m[col * 4 + row]; }
    float operator()(int row, int col) const { return m[col * 4 + row]; }
    const float* data() const { return m; }
};

static Mat4 mat4Identity() {
    Mat4 r{};
    r(0, 0) = r(1, 1) = r(2, 2) = r(3, 3) = 1.0f;
    return r;
}

static Mat4 mat4Mul(const Mat4& a, const Mat4& b) {
    Mat4 r{};
    for (int c = 0; c < 4; ++c)
        for (int row = 0; row < 4; ++row)
            for (int k = 0; k < 4; ++k)
                r(row, c) += a(row, k) * b(k, c);
    return r;
}

static Mat4 mat4Perspective(float fovRad, float aspect, float near, float far) {
    Mat4 r{};
    float tanHalf = std::tan(fovRad / 2.0f);
    r(0, 0) = 1.0f / (aspect * tanHalf);
    r(1, 1) = 1.0f / tanHalf;
    r(2, 2) = -(far + near) / (far - near);
    r(2, 3) = -2.0f * far * near / (far - near);
    r(3, 2) = -1.0f;
    return r;
}

static Mat4 mat4LookAt(Vec3 eye, Vec3 center, Vec3 up) {
    Vec3 f = normalize(center - eye);
    Vec3 s = normalize(cross(f, up));
    Vec3 u = cross(s, f);
    Mat4 r = mat4Identity();
    r(0, 0) = s.x;  r(0, 1) = s.y;  r(0, 2) = s.z;
    r(1, 0) = u.x;  r(1, 1) = u.y;  r(1, 2) = u.z;
    r(2, 0) = -f.x; r(2, 1) = -f.y; r(2, 2) = -f.z;
    r(0, 3) = -dot(s, eye);
    r(1, 3) = -dot(u, eye);
    r(2, 3) = dot(f, eye);
    return r;
}

static Mat4 mat4RotateY(float rad) {
    Mat4 r = mat4Identity();
    float c = std::cos(rad), s = std::sin(rad);
    r(0, 0) = c;  r(0, 2) = s;
    r(2, 0) = -s; r(2, 2) = c;
    return r;
}

static Mat4 mat4RotateX(float rad) {
    Mat4 r = mat4Identity();
    float c = std::cos(rad), s = std::sin(rad);
    r(1, 1) = c;  r(1, 2) = -s;
    r(2, 1) = s;  r(2, 2) = c;
    return r;
}

// Extract upper-left 3x3 (for normal matrix)
static void mat4ToMat3(const Mat4& src, float dst[9]) {
    for (int c = 0; c < 3; ++c)
        for (int r = 0; r < 3; ++r)
            dst[c * 3 + r] = src(r, c);
}

// ----- Shader helpers -----

static std::string loadFile(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) {
        std::cerr << "Failed to open: " << path << "\n";
        std::exit(1);
    }
    std::ostringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

static GLuint compileShader(GLenum type, const std::string& source) {
    GLuint shader = glCreateShader(type);
    const char* src = source.c_str();
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    GLint ok;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        std::cerr << "Shader compile error:\n" << log << "\n";
        std::exit(1);
    }
    return shader;
}

static GLuint createProgram(const std::string& vertPath, const std::string& fragPath) {
    GLuint vs = compileShader(GL_VERTEX_SHADER, loadFile(vertPath));
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, loadFile(fragPath));
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    GLint ok;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetProgramInfoLog(prog, sizeof(log), nullptr, log);
        std::cerr << "Program link error:\n" << log << "\n";
        std::exit(1);
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

// ----- Texture loading -----

static GLuint loadTexture(const std::string& path) {
    int w, h, channels;
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &channels, 4);
    if (!data) {
        std::cerr << "Failed to load texture: " << path << "\n";
        std::exit(1);
    }

    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    stbi_image_free(data);
    return tex;
}

// ----- Cube geometry -----

// 6 faces x 2 triangles x 3 vertices = 36 vertices
// Each vertex: position(3) + normal(3) + uv(2)
// clang-format off
static const float cubeVertices[] = {
    // Back face (z = -0.5)
    -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f, 1.0f,
     0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f, 0.0f,
     0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f, 1.0f,
     0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f, 0.0f,
    -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f, 1.0f,
    -0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f, 0.0f,
    // Front face (z = +0.5)
    -0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  0.0f, 1.0f,
     0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  1.0f, 1.0f,
     0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  1.0f, 0.0f,
     0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  1.0f, 0.0f,
    -0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  0.0f, 0.0f,
    -0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  0.0f, 1.0f,
    // Left face (x = -0.5)
    -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  1.0f, 0.0f,
    -0.5f,  0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  0.0f, 0.0f,
    -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  0.0f, 1.0f,
    -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  0.0f, 1.0f,
    -0.5f, -0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  1.0f, 1.0f,
    -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  1.0f, 0.0f,
    // Right face (x = +0.5)
     0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  0.0f, 0.0f,
     0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f,
     0.5f,  0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f,
     0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f,
     0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  0.0f, 0.0f,
     0.5f, -0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  0.0f, 1.0f,
    // Bottom face (y = -0.5)
    -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  0.0f, 1.0f,
     0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  1.0f, 1.0f,
     0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  1.0f, 0.0f,
     0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  1.0f, 0.0f,
    -0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  0.0f, 0.0f,
    -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  0.0f, 1.0f,
    // Top face (y = +0.5)
    -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  0.0f, 0.0f,
     0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  1.0f, 1.0f,
     0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  1.0f, 0.0f,
     0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  1.0f, 1.0f,
    -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  0.0f, 0.0f,
    -0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  0.0f, 1.0f,
};
// clang-format on

// ----- Window callback -----

static int winW = 800, winH = 600;
static void framebufferSizeCB(GLFWwindow*, int w, int h) {
    winW = w;
    winH = h;
    glViewport(0, 0, w, h);
}

// ----- Main -----

int main() {
    if (!glfwInit()) {
        std::cerr << "Failed to init GLFW\n";
        return 1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(winW, winH, "Sand Block Cube", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create window\n";
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCB);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to init GLEW\n";
        return 1;
    }

    glEnable(GL_DEPTH_TEST);

    // Build shader program
    GLuint program = createProgram("shaders/vertex.glsl", "shaders/fragment.glsl");

    // Load sand block texture
    GLuint sandTex = loadTexture("textures/sand_block_256x256.png");

    // Upload cube geometry
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVertices), cubeVertices, GL_STATIC_DRAW);
    // position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // normal
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // uv
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    // Uniform locations
    GLint locModel = glGetUniformLocation(program, "uModel");
    GLint locView = glGetUniformLocation(program, "uView");
    GLint locProj = glGetUniformLocation(program, "uProjection");
    GLint locNorm = glGetUniformLocation(program, "uNormalMatrix");
    GLint locLightPos = glGetUniformLocation(program, "uLightPos");
    GLint locLightCol = glGetUniformLocation(program, "uLightColor");
    GLint locViewPos = glGetUniformLocation(program, "uViewPos");
    GLint locTex = glGetUniformLocation(program, "uTexture");

    Vec3 lightPos = {2.0f, 3.0f, 2.0f};
    Vec3 lightColor = {1.0f, 1.0f, 1.0f};
    Vec3 cameraPos = {0.0f, 1.5f, 3.5f};

    while (!glfwWindowShouldClose(window)) {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        glClearColor(0.1f, 0.1f, 0.12f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(program);

        float time = static_cast<float>(glfwGetTime());

        // Model: rotate cube
        Mat4 model = mat4Mul(mat4RotateY(time * 0.7f), mat4RotateX(time * 0.4f));

        // View
        Mat4 view = mat4LookAt(cameraPos, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f});

        // Projection
        float aspect = static_cast<float>(winW) / static_cast<float>(winH);
        Mat4 proj = mat4Perspective(3.14159f / 4.0f, aspect, 0.1f, 100.0f);

        // Normal matrix (upper-left 3x3 of model)
        float normalMat[9];
        mat4ToMat3(model, normalMat);

        glUniformMatrix4fv(locModel, 1, GL_FALSE, model.data());
        glUniformMatrix4fv(locView, 1, GL_FALSE, view.data());
        glUniformMatrix4fv(locProj, 1, GL_FALSE, proj.data());
        glUniformMatrix3fv(locNorm, 1, GL_FALSE, normalMat);
        glUniform3f(locLightPos, lightPos.x, lightPos.y, lightPos.z);
        glUniform3f(locLightCol, lightColor.x, lightColor.y, lightColor.z);
        glUniform3f(locViewPos, cameraPos.x, cameraPos.y, cameraPos.z);

        // Bind sand texture to unit 0
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, sandTex);
        glUniform1i(locTex, 0);

        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, 36);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteTextures(1, &sandTex);
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glDeleteProgram(program);
    glfwTerminate();
    return 0;
}
