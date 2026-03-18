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
        char log[1024];
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
        char log[1024];
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

// ----- Fullscreen quad -----

// Two triangles covering [-1,1] x [-1,1]
static const float quadVertices[] = {
    -1.0f, -1.0f,
     1.0f, -1.0f,
     1.0f,  1.0f,
    -1.0f, -1.0f,
     1.0f,  1.0f,
    -1.0f,  1.0f,
};

// ----- Window & mouse state -----

static int winW = 1280, winH = 720;
static float camAngleX = 0.0f;   // horizontal orbit angle (radians)
static float camAngleY = 0.4f;   // vertical angle (radians, clamped)
static float camDist = 6.0f;     // distance from target
static bool mouseDown = false;
static double lastMouseX = 0.0, lastMouseY = 0.0;

static void framebufferSizeCB(GLFWwindow*, int w, int h) {
    winW = w;
    winH = h;
    glViewport(0, 0, w, h);
}

static void mouseButtonCB(GLFWwindow* window, int button, int action, int /*mods*/) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        mouseDown = (action == GLFW_PRESS);
        if (mouseDown)
            glfwGetCursorPos(window, &lastMouseX, &lastMouseY);
    }
}

static void cursorPosCB(GLFWwindow*, double xpos, double ypos) {
    if (!mouseDown) return;
    double dx = xpos - lastMouseX;
    double dy = ypos - lastMouseY;
    lastMouseX = xpos;
    lastMouseY = ypos;
    camAngleX -= static_cast<float>(dx) * 0.005f;
    camAngleY += static_cast<float>(dy) * 0.005f;
    // Clamp vertical angle
    if (camAngleY < -1.4f) camAngleY = -1.4f;
    if (camAngleY > 1.4f) camAngleY = 1.4f;
}

static void scrollCB(GLFWwindow*, double /*xoffset*/, double yoffset) {
    camDist -= static_cast<float>(yoffset) * 0.5f;
    if (camDist < 1.5f) camDist = 1.5f;
    if (camDist > 20.0f) camDist = 20.0f;
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

    GLFWwindow* window = glfwCreateWindow(winW, winH, "Glass Sphere - Ray Tracing", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create window\n";
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCB);
    glfwSetMouseButtonCallback(window, mouseButtonCB);
    glfwSetCursorPosCallback(window, cursorPosCB);
    glfwSetScrollCallback(window, scrollCB);
    glfwSwapInterval(1); // VSync

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to init GLEW\n";
        return 1;
    }

    // Build shader program
    GLuint program = createProgram("shaders/vertex.glsl", "shaders/fragment.glsl");

    // Load sand block textures
    GLuint sandTex = loadTexture("textures/sand_block_256x256.png");
    GLuint sandNormalTex = loadTexture("textures/sand_block_256x256_normal.png");

    // Upload fullscreen quad
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Uniform locations
    GLint locTime = glGetUniformLocation(program, "uTime");
    GLint locRes = glGetUniformLocation(program, "uResolution");
    GLint locSandTex = glGetUniformLocation(program, "uSandTex");
    GLint locSandNormal = glGetUniformLocation(program, "uSandNormalMap");
    GLint locCamAngleX = glGetUniformLocation(program, "uCamAngleX");
    GLint locCamAngleY = glGetUniformLocation(program, "uCamAngleY");
    GLint locCamDist = glGetUniformLocation(program, "uCamDist");

    while (!glfwWindowShouldClose(window)) {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(program);

        float time = static_cast<float>(glfwGetTime());
        glUniform1f(locTime, time);
        glUniform2f(locRes, static_cast<float>(winW), static_cast<float>(winH));
        glUniform1f(locCamAngleX, camAngleX);
        glUniform1f(locCamAngleY, camAngleY);
        glUniform1f(locCamDist, camDist);

        // Bind sand texture to texture unit 0
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, sandTex);
        glUniform1i(locSandTex, 0);

        // Bind sand normal map to texture unit 1
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, sandNormalTex);
        glUniform1i(locSandNormal, 1);

        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteTextures(1, &sandNormalTex);
    glDeleteTextures(1, &sandTex);
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glDeleteProgram(program);
    glfwTerminate();
    return 0;
}
