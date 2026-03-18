// Vulkan Hardware Ray Tracing Sample
// Renders a scene with ground plane + cubes using RTX RT cores.
// Demonstrates: acceleration structures (BLAS/TLAS), RT pipeline, SBT, shadow rays.

#define VK_NO_PROTOTYPES  // We load all functions dynamically
#include <vulkan/vulkan.h>

// GLFW with Vulkan support
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
#include <vector>
#include <fstream>
#include <cstring>
#include <cassert>
#include <chrono>
#include <algorithm>

// ========================================================================
// Macros
// ========================================================================

#define VK_CHECK(x) do { \
    VkResult _r = (x); \
    if (_r != VK_SUCCESS) { \
        fprintf(stderr, "Vulkan error %d at %s:%d\n", _r, __FILE__, __LINE__); \
        abort(); \
    } \
} while(0)

// ========================================================================
// Data types (shared with shaders)
// ========================================================================

struct Vertex {
    float pos[3];
    float normal[3];
};

// Material types (stored in color[3])
static const float MAT_DIFFUSE = 0.0f;
static const float MAT_MIRROR  = 1.0f;
static const float MAT_GLASS   = 2.0f;

struct ObjDesc {
    uint64_t vertexAddress;
    uint64_t indexAddress;
    float color[4];  // rgb + material type
};

struct CameraUBO {
    glm::mat4 viewInverse;
    glm::mat4 projInverse;
};

// ========================================================================
// Constants
// ========================================================================

static const uint32_t WIDTH  = 1280;
static const uint32_t HEIGHT = 720;

static const char* VALIDATION_LAYERS[] = { "VK_LAYER_KHRONOS_validation" };
static const uint32_t VALIDATION_LAYER_COUNT = 1;

static const char* DEVICE_EXTENSIONS[] = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
    VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
    VK_KHR_SPIRV_1_4_EXTENSION_NAME,
    VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME,
};
static const uint32_t DEVICE_EXTENSION_COUNT = 7;

// ========================================================================
// Vulkan function pointers (loaded dynamically)
// ========================================================================

// Global-level
static PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr;

// Instance-level (subset we need)
static PFN_vkCreateInstance vkCreateInstance;
static PFN_vkEnumerateInstanceLayerProperties vkEnumerateInstanceLayerProperties;
static PFN_vkDestroyInstance vkDestroyInstance;
static PFN_vkEnumeratePhysicalDevices vkEnumeratePhysicalDevices;
static PFN_vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties;
static PFN_vkGetPhysicalDeviceProperties2 vkGetPhysicalDeviceProperties2;
static PFN_vkGetPhysicalDeviceFeatures2 vkGetPhysicalDeviceFeatures2;
static PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties;
static PFN_vkGetPhysicalDeviceQueueFamilyProperties vkGetPhysicalDeviceQueueFamilyProperties;
static PFN_vkGetPhysicalDeviceSurfaceSupportKHR vkGetPhysicalDeviceSurfaceSupportKHR;
static PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR vkGetPhysicalDeviceSurfaceCapabilitiesKHR;
static PFN_vkGetPhysicalDeviceSurfaceFormatsKHR vkGetPhysicalDeviceSurfaceFormatsKHR;
static PFN_vkGetPhysicalDeviceSurfacePresentModesKHR vkGetPhysicalDeviceSurfacePresentModesKHR;
static PFN_vkEnumerateDeviceExtensionProperties vkEnumerateDeviceExtensionProperties;
static PFN_vkCreateDevice vkCreateDevice;
static PFN_vkDestroySurfaceKHR vkDestroySurfaceKHR;

// Debug
static PFN_vkCreateDebugUtilsMessengerEXT vkCreateDebugUtilsMessengerEXT;
static PFN_vkDestroyDebugUtilsMessengerEXT vkDestroyDebugUtilsMessengerEXT;

// Device-level
static PFN_vkDestroyDevice vkDestroyDevice;
static PFN_vkGetDeviceQueue vkGetDeviceQueue;
static PFN_vkCreateSwapchainKHR vkCreateSwapchainKHR;
static PFN_vkDestroySwapchainKHR vkDestroySwapchainKHR;
static PFN_vkGetSwapchainImagesKHR vkGetSwapchainImagesKHR;
static PFN_vkAcquireNextImageKHR vkAcquireNextImageKHR;
static PFN_vkQueuePresentKHR vkQueuePresentKHR;
static PFN_vkCreateCommandPool vkCreateCommandPool;
static PFN_vkDestroyCommandPool vkDestroyCommandPool;
static PFN_vkAllocateCommandBuffers vkAllocateCommandBuffers;
static PFN_vkFreeCommandBuffers vkFreeCommandBuffers;
static PFN_vkBeginCommandBuffer vkBeginCommandBuffer;
static PFN_vkEndCommandBuffer vkEndCommandBuffer;
static PFN_vkCmdPipelineBarrier vkCmdPipelineBarrier;
static PFN_vkCmdCopyImage vkCmdCopyImage;
static PFN_vkQueueSubmit vkQueueSubmit;
static PFN_vkQueueWaitIdle vkQueueWaitIdle;
static PFN_vkDeviceWaitIdle vkDeviceWaitIdle;
static PFN_vkCreateSemaphore vkCreateSemaphore;
static PFN_vkDestroySemaphore vkDestroySemaphore;
static PFN_vkCreateFence vkCreateFence;
static PFN_vkDestroyFence vkDestroyFence;
static PFN_vkWaitForFences vkWaitForFences;
static PFN_vkResetFences vkResetFences;
static PFN_vkCreateBuffer vkCreateBuffer;
static PFN_vkDestroyBuffer vkDestroyBuffer;
static PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements;
static PFN_vkAllocateMemory vkAllocateMemory;
static PFN_vkFreeMemory vkFreeMemory;
static PFN_vkBindBufferMemory vkBindBufferMemory;
static PFN_vkMapMemory vkMapMemory;
static PFN_vkUnmapMemory vkUnmapMemory;
static PFN_vkCreateImage vkCreateImage;
static PFN_vkDestroyImage vkDestroyImage;
static PFN_vkGetImageMemoryRequirements vkGetImageMemoryRequirements;
static PFN_vkBindImageMemory vkBindImageMemory;
static PFN_vkCreateImageView vkCreateImageView;
static PFN_vkDestroyImageView vkDestroyImageView;
static PFN_vkCreateDescriptorSetLayout vkCreateDescriptorSetLayout;
static PFN_vkDestroyDescriptorSetLayout vkDestroyDescriptorSetLayout;
static PFN_vkCreateDescriptorPool vkCreateDescriptorPool;
static PFN_vkDestroyDescriptorPool vkDestroyDescriptorPool;
static PFN_vkAllocateDescriptorSets vkAllocateDescriptorSets;
static PFN_vkUpdateDescriptorSets vkUpdateDescriptorSets;
static PFN_vkCreatePipelineLayout vkCreatePipelineLayout;
static PFN_vkDestroyPipelineLayout vkDestroyPipelineLayout;
static PFN_vkDestroyPipeline vkDestroyPipeline;
static PFN_vkCreateShaderModule vkCreateShaderModule;
static PFN_vkDestroyShaderModule vkDestroyShaderModule;
static PFN_vkCmdBindPipeline vkCmdBindPipeline;
static PFN_vkCmdBindDescriptorSets vkCmdBindDescriptorSets;
static PFN_vkFlushMappedMemoryRanges vkFlushMappedMemoryRanges;

// RT extension functions
static PFN_vkGetBufferDeviceAddressKHR _vkGetBufferDeviceAddress;
static PFN_vkCreateAccelerationStructureKHR _vkCreateAccelerationStructure;
static PFN_vkDestroyAccelerationStructureKHR _vkDestroyAccelerationStructure;
static PFN_vkGetAccelerationStructureBuildSizesKHR _vkGetAccelerationStructureBuildSizes;
static PFN_vkCmdBuildAccelerationStructuresKHR _vkCmdBuildAccelerationStructures;
static PFN_vkGetAccelerationStructureDeviceAddressKHR _vkGetAccelerationStructureDeviceAddress;
static PFN_vkCreateRayTracingPipelinesKHR _vkCreateRayTracingPipelines;
static PFN_vkGetRayTracingShaderGroupHandlesKHR _vkGetRayTracingShaderGroupHandles;
static PFN_vkCmdTraceRaysKHR _vkCmdTraceRays;

// ========================================================================
// Globals
// ========================================================================

static GLFWwindow* gWindow;
static VkInstance gInstance;
static VkDebugUtilsMessengerEXT gDebugMessenger;
static VkSurfaceKHR gSurface;
static VkPhysicalDevice gPhysicalDevice;
static VkDevice gDevice;
static VkQueue gQueue;
static uint32_t gQueueFamily;
static VkSwapchainKHR gSwapchain;
static std::vector<VkImage> gSwapImages;
static VkFormat gSwapFormat;
static VkExtent2D gSwapExtent;
static VkCommandPool gCmdPool;
static VkSemaphore gImageAvailSem, gRenderDoneSem;
static VkFence gInFlightFence;

// RT output image
static VkImage gRTImage;
static VkDeviceMemory gRTImageMem;
static VkImageView gRTImageView;

// Geometry
struct GeometryData {
    VkBuffer vertexBuf, indexBuf;
    VkDeviceMemory vertexMem, indexMem;
    uint32_t indexCount;
    VkDeviceAddress vertexAddr, indexAddr;
};
static std::vector<GeometryData> gGeometries;

// Acceleration structures
struct AccelStruct {
    VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceAddress address = 0;
};
static std::vector<AccelStruct> gBLAS;
static AccelStruct gTLAS;

// Descriptors and pipeline
static VkBuffer gObjDescBuf;
static VkDeviceMemory gObjDescMem;
static VkBuffer gUBOBuf;
static VkDeviceMemory gUBOMem;
static void* gUBOMapped;
static VkDescriptorSetLayout gDescLayout;
static VkPipelineLayout gPipeLayout;
static VkPipeline gRTPipeline;
static VkDescriptorPool gDescPool;
static VkDescriptorSet gDescSet;

// SBT
static VkBuffer gSBTBuf;
static VkDeviceMemory gSBTMem;
static VkStridedDeviceAddressRegionKHR gRgenRegion{}, gMissRegion{}, gHitRegion{}, gCallRegion{};

// RT properties
static VkPhysicalDeviceRayTracingPipelinePropertiesKHR gRTProps{};

// ========================================================================
// Load Vulkan function pointers
// ========================================================================

static void loadGlobalFunctions() {
    vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)glfwGetInstanceProcAddress(nullptr, "vkGetInstanceProcAddr");
    vkCreateInstance = (PFN_vkCreateInstance)vkGetInstanceProcAddr(nullptr, "vkCreateInstance");
    vkEnumerateInstanceLayerProperties = (PFN_vkEnumerateInstanceLayerProperties)vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceLayerProperties");
}

#define LOAD_INSTANCE_FUNC(fn) fn = (PFN_##fn)vkGetInstanceProcAddr(gInstance, #fn)

static void loadInstanceFunctions() {
    LOAD_INSTANCE_FUNC(vkDestroyInstance);
    LOAD_INSTANCE_FUNC(vkEnumeratePhysicalDevices);
    LOAD_INSTANCE_FUNC(vkGetPhysicalDeviceProperties);
    LOAD_INSTANCE_FUNC(vkGetPhysicalDeviceProperties2);
    LOAD_INSTANCE_FUNC(vkGetPhysicalDeviceFeatures2);
    LOAD_INSTANCE_FUNC(vkGetPhysicalDeviceMemoryProperties);
    LOAD_INSTANCE_FUNC(vkGetPhysicalDeviceQueueFamilyProperties);
    LOAD_INSTANCE_FUNC(vkGetPhysicalDeviceSurfaceSupportKHR);
    LOAD_INSTANCE_FUNC(vkGetPhysicalDeviceSurfaceCapabilitiesKHR);
    LOAD_INSTANCE_FUNC(vkGetPhysicalDeviceSurfaceFormatsKHR);
    LOAD_INSTANCE_FUNC(vkGetPhysicalDeviceSurfacePresentModesKHR);
    LOAD_INSTANCE_FUNC(vkEnumerateDeviceExtensionProperties);
    LOAD_INSTANCE_FUNC(vkCreateDevice);
    LOAD_INSTANCE_FUNC(vkDestroySurfaceKHR);
    LOAD_INSTANCE_FUNC(vkCreateDebugUtilsMessengerEXT);
    LOAD_INSTANCE_FUNC(vkDestroyDebugUtilsMessengerEXT);
}

#define LOAD_DEVICE_FUNC(fn) fn = (PFN_##fn)vkGetInstanceProcAddr(gInstance, #fn)

static void loadDeviceFunctions() {
    LOAD_DEVICE_FUNC(vkDestroyDevice);
    LOAD_DEVICE_FUNC(vkGetDeviceQueue);
    LOAD_DEVICE_FUNC(vkCreateSwapchainKHR);
    LOAD_DEVICE_FUNC(vkDestroySwapchainKHR);
    LOAD_DEVICE_FUNC(vkGetSwapchainImagesKHR);
    LOAD_DEVICE_FUNC(vkAcquireNextImageKHR);
    LOAD_DEVICE_FUNC(vkQueuePresentKHR);
    LOAD_DEVICE_FUNC(vkCreateCommandPool);
    LOAD_DEVICE_FUNC(vkDestroyCommandPool);
    LOAD_DEVICE_FUNC(vkAllocateCommandBuffers);
    LOAD_DEVICE_FUNC(vkFreeCommandBuffers);
    LOAD_DEVICE_FUNC(vkBeginCommandBuffer);
    LOAD_DEVICE_FUNC(vkEndCommandBuffer);
    LOAD_DEVICE_FUNC(vkCmdPipelineBarrier);
    LOAD_DEVICE_FUNC(vkCmdCopyImage);
    LOAD_DEVICE_FUNC(vkQueueSubmit);
    LOAD_DEVICE_FUNC(vkQueueWaitIdle);
    LOAD_DEVICE_FUNC(vkDeviceWaitIdle);
    LOAD_DEVICE_FUNC(vkCreateSemaphore);
    LOAD_DEVICE_FUNC(vkDestroySemaphore);
    LOAD_DEVICE_FUNC(vkCreateFence);
    LOAD_DEVICE_FUNC(vkDestroyFence);
    LOAD_DEVICE_FUNC(vkWaitForFences);
    LOAD_DEVICE_FUNC(vkResetFences);
    LOAD_DEVICE_FUNC(vkCreateBuffer);
    LOAD_DEVICE_FUNC(vkDestroyBuffer);
    LOAD_DEVICE_FUNC(vkGetBufferMemoryRequirements);
    LOAD_DEVICE_FUNC(vkAllocateMemory);
    LOAD_DEVICE_FUNC(vkFreeMemory);
    LOAD_DEVICE_FUNC(vkBindBufferMemory);
    LOAD_DEVICE_FUNC(vkMapMemory);
    LOAD_DEVICE_FUNC(vkUnmapMemory);
    LOAD_DEVICE_FUNC(vkCreateImage);
    LOAD_DEVICE_FUNC(vkDestroyImage);
    LOAD_DEVICE_FUNC(vkGetImageMemoryRequirements);
    LOAD_DEVICE_FUNC(vkBindImageMemory);
    LOAD_DEVICE_FUNC(vkCreateImageView);
    LOAD_DEVICE_FUNC(vkDestroyImageView);
    LOAD_DEVICE_FUNC(vkCreateDescriptorSetLayout);
    LOAD_DEVICE_FUNC(vkDestroyDescriptorSetLayout);
    LOAD_DEVICE_FUNC(vkCreateDescriptorPool);
    LOAD_DEVICE_FUNC(vkDestroyDescriptorPool);
    LOAD_DEVICE_FUNC(vkAllocateDescriptorSets);
    LOAD_DEVICE_FUNC(vkUpdateDescriptorSets);
    LOAD_DEVICE_FUNC(vkCreatePipelineLayout);
    LOAD_DEVICE_FUNC(vkDestroyPipelineLayout);
    LOAD_DEVICE_FUNC(vkDestroyPipeline);
    LOAD_DEVICE_FUNC(vkCreateShaderModule);
    LOAD_DEVICE_FUNC(vkDestroyShaderModule);
    LOAD_DEVICE_FUNC(vkCmdBindPipeline);
    LOAD_DEVICE_FUNC(vkCmdBindDescriptorSets);
    LOAD_DEVICE_FUNC(vkFlushMappedMemoryRanges);

    // RT extensions
    _vkGetBufferDeviceAddress = (PFN_vkGetBufferDeviceAddressKHR)vkGetInstanceProcAddr(gInstance, "vkGetBufferDeviceAddressKHR");
    _vkCreateAccelerationStructure = (PFN_vkCreateAccelerationStructureKHR)vkGetInstanceProcAddr(gInstance, "vkCreateAccelerationStructureKHR");
    _vkDestroyAccelerationStructure = (PFN_vkDestroyAccelerationStructureKHR)vkGetInstanceProcAddr(gInstance, "vkDestroyAccelerationStructureKHR");
    _vkGetAccelerationStructureBuildSizes = (PFN_vkGetAccelerationStructureBuildSizesKHR)vkGetInstanceProcAddr(gInstance, "vkGetAccelerationStructureBuildSizesKHR");
    _vkCmdBuildAccelerationStructures = (PFN_vkCmdBuildAccelerationStructuresKHR)vkGetInstanceProcAddr(gInstance, "vkCmdBuildAccelerationStructuresKHR");
    _vkGetAccelerationStructureDeviceAddress = (PFN_vkGetAccelerationStructureDeviceAddressKHR)vkGetInstanceProcAddr(gInstance, "vkGetAccelerationStructureDeviceAddressKHR");
    _vkCreateRayTracingPipelines = (PFN_vkCreateRayTracingPipelinesKHR)vkGetInstanceProcAddr(gInstance, "vkCreateRayTracingPipelinesKHR");
    _vkGetRayTracingShaderGroupHandles = (PFN_vkGetRayTracingShaderGroupHandlesKHR)vkGetInstanceProcAddr(gInstance, "vkGetRayTracingShaderGroupHandlesKHR");
    _vkCmdTraceRays = (PFN_vkCmdTraceRaysKHR)vkGetInstanceProcAddr(gInstance, "vkCmdTraceRaysKHR");
}

// ========================================================================
// Debug callback
// ========================================================================

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        fprintf(stderr, "[Vulkan] %s\n", data->pMessage);
    return VK_FALSE;
}

// ========================================================================
// Helpers
// ========================================================================

static std::vector<char> readFile(const std::string& path) {
    std::ifstream f(path, std::ios::ate | std::ios::binary);
    if (!f.is_open()) { fprintf(stderr, "Failed to open: %s\n", path.c_str()); abort(); }
    size_t sz = f.tellg();
    std::vector<char> buf(sz);
    f.seekg(0);
    f.read(buf.data(), sz);
    return buf;
}

static uint32_t findMemoryType(uint32_t filter, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(gPhysicalDevice, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((filter & (1 << i)) && (mp.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    fprintf(stderr, "Failed to find memory type\n"); abort();
}

static void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props,
                          VkBuffer& buf, VkDeviceMemory& mem) {
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size = size;
    ci.usage = usage;
    VK_CHECK(vkCreateBuffer(gDevice, &ci, nullptr, &buf));

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(gDevice, buf, &req);

    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, props);

    VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    if (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
        flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        ai.pNext = &flags;
    }

    VK_CHECK(vkAllocateMemory(gDevice, &ai, nullptr, &mem));
    vkBindBufferMemory(gDevice, buf, mem, 0);
}

static void uploadBuffer(VkBuffer buf, VkDeviceMemory mem, const void* data, VkDeviceSize size) {
    void* mapped;
    vkMapMemory(gDevice, mem, 0, size, 0, &mapped);
    memcpy(mapped, data, size);
    vkUnmapMemory(gDevice, mem);
}

static VkDeviceAddress getBufferAddress(VkBuffer buf) {
    VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    info.buffer = buf;
    return _vkGetBufferDeviceAddress(gDevice, &info);
}

static VkCommandBuffer beginCmd() {
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = gCmdPool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(gDevice, &ai, &cmd);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    return cmd;
}

static void endCmd(VkCommandBuffer cmd) {
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(gQueue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(gQueue);
    vkFreeCommandBuffers(gDevice, gCmdPool, 1, &cmd);
}

static VkShaderModule createShaderModule(const std::vector<char>& code) {
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = code.size();
    ci.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule mod;
    VK_CHECK(vkCreateShaderModule(gDevice, &ci, nullptr, &mod));
    return mod;
}

static uint32_t alignUp(uint32_t value, uint32_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

// ========================================================================
// Instance + Debug
// ========================================================================

static void initInstance() {
    VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.pApplicationName = "Vulkan RT Sample";
    appInfo.apiVersion = VK_API_VERSION_1_2;

    // Required extensions: GLFW surface + debug
    uint32_t glfwExtCount;
    const char** glfwExts = glfwGetRequiredInstanceExtensions(&glfwExtCount);
    std::vector<const char*> exts(glfwExts, glfwExts + glfwExtCount);
    exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    // Check if validation layer is available
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    std::vector<VkLayerProperties> availLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availLayers.data());
    bool hasValidation = false;
    for (auto& l : availLayers) {
        if (strcmp(l.layerName, VALIDATION_LAYERS[0]) == 0) { hasValidation = true; break; }
    }
    if (!hasValidation) {
        printf("Validation layer not found, running without it.\n");
        // Also skip debug utils extension
        exts.pop_back();
    }

    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &appInfo;
    ci.enabledExtensionCount = (uint32_t)exts.size();
    ci.ppEnabledExtensionNames = exts.data();
    if (hasValidation) {
        ci.enabledLayerCount = VALIDATION_LAYER_COUNT;
        ci.ppEnabledLayerNames = VALIDATION_LAYERS;
    }

    VK_CHECK(vkCreateInstance(&ci, nullptr, &gInstance));
    loadInstanceFunctions();

    // Debug messenger
    VkDebugUtilsMessengerCreateInfoEXT dci{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    dci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                          VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    dci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                      VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
    dci.pfnUserCallback = debugCallback;
    if (vkCreateDebugUtilsMessengerEXT)
        vkCreateDebugUtilsMessengerEXT(gInstance, &dci, nullptr, &gDebugMessenger);
}

// ========================================================================
// Physical device selection
// ========================================================================

static bool checkDeviceExtensions(VkPhysicalDevice pd) {
    uint32_t count;
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> avail(count);
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &count, avail.data());

    for (uint32_t i = 0; i < DEVICE_EXTENSION_COUNT; i++) {
        bool found = false;
        for (auto& ext : avail) {
            if (strcmp(ext.extensionName, DEVICE_EXTENSIONS[i]) == 0) { found = true; break; }
        }
        if (!found) return false;
    }
    return true;
}

static void pickPhysicalDevice() {
    uint32_t count;
    vkEnumeratePhysicalDevices(gInstance, &count, nullptr);
    assert(count > 0 && "No Vulkan-capable GPU found");
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(gInstance, &count, devices.data());

    for (auto& pd : devices) {
        if (!checkDeviceExtensions(pd)) continue;

        // Check RT support
        VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtFeats{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
        VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeats{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
        asFeats.pNext = &rtFeats;
        VkPhysicalDeviceFeatures2 feats2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        feats2.pNext = &asFeats;
        vkGetPhysicalDeviceFeatures2(pd, &feats2);

        if (!rtFeats.rayTracingPipeline || !asFeats.accelerationStructure) continue;

        // Check queue family
        uint32_t qfCount;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfCount, nullptr);
        std::vector<VkQueueFamilyProperties> qfs(qfCount);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfCount, qfs.data());

        for (uint32_t i = 0; i < qfCount; i++) {
            VkBool32 present = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(pd, i, gSurface, &present);
            if ((qfs[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) && present) {
                gPhysicalDevice = pd;
                gQueueFamily = i;

                // Get RT properties
                gRTProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
                VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
                props2.pNext = &gRTProps;
                vkGetPhysicalDeviceProperties2(pd, &props2);

                VkPhysicalDeviceProperties devProps;
                vkGetPhysicalDeviceProperties(pd, &devProps);
                printf("Selected GPU: %s\n", devProps.deviceName);
                printf("RT handle size: %u, alignment: %u, base alignment: %u\n",
                       gRTProps.shaderGroupHandleSize,
                       gRTProps.shaderGroupHandleAlignment,
                       gRTProps.shaderGroupBaseAlignment);
                return;
            }
        }
    }
    fprintf(stderr, "No suitable GPU with RT support found\n");
    abort();
}

// ========================================================================
// Logical device
// ========================================================================

static void createDevice() {
    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = gQueueFamily;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;

    // Enable features via pNext chain
    VkPhysicalDeviceScalarBlockLayoutFeatures scalarFeats{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES};
    scalarFeats.scalarBlockLayout = VK_TRUE;

    VkPhysicalDeviceBufferDeviceAddressFeatures bdaFeats{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
    bdaFeats.bufferDeviceAddress = VK_TRUE;
    bdaFeats.pNext = &scalarFeats;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeats{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    asFeats.accelerationStructure = VK_TRUE;
    asFeats.pNext = &bdaFeats;

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtFeats{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
    rtFeats.rayTracingPipeline = VK_TRUE;
    rtFeats.pNext = &asFeats;

    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &rtFeats;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = DEVICE_EXTENSION_COUNT;
    dci.ppEnabledExtensionNames = DEVICE_EXTENSIONS;

    VK_CHECK(vkCreateDevice(gPhysicalDevice, &dci, nullptr, &gDevice));
    loadDeviceFunctions();
    vkGetDeviceQueue(gDevice, gQueueFamily, 0, &gQueue);
}

// ========================================================================
// Swapchain
// ========================================================================

static void createSwapchain() {
    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(gPhysicalDevice, gSurface, &caps);

    uint32_t fmtCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(gPhysicalDevice, gSurface, &fmtCount, nullptr);
    std::vector<VkSurfaceFormatKHR> fmts(fmtCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(gPhysicalDevice, gSurface, &fmtCount, fmts.data());

    // Prefer B8G8R8A8_SRGB
    gSwapFormat = fmts[0].format;
    VkColorSpaceKHR colorSpace = fmts[0].colorSpace;
    for (auto& f : fmts) {
        if (f.format == VK_FORMAT_B8G8R8A8_SRGB && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            gSwapFormat = f.format;
            colorSpace = f.colorSpace;
            break;
        }
    }

    gSwapExtent = caps.currentExtent;
    if (gSwapExtent.width == UINT32_MAX) {
        gSwapExtent = {WIDTH, HEIGHT};
    }

    uint32_t imgCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && imgCount > caps.maxImageCount)
        imgCount = caps.maxImageCount;

    VkSwapchainCreateInfoKHR sci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    sci.surface = gSurface;
    sci.minImageCount = imgCount;
    sci.imageFormat = gSwapFormat;
    sci.imageColorSpace = colorSpace;
    sci.imageExtent = gSwapExtent;
    sci.imageArrayLayers = 1;
    sci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    sci.preTransform = caps.currentTransform;
    sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    sci.clipped = VK_TRUE;

    VK_CHECK(vkCreateSwapchainKHR(gDevice, &sci, nullptr, &gSwapchain));

    vkGetSwapchainImagesKHR(gDevice, gSwapchain, &imgCount, nullptr);
    gSwapImages.resize(imgCount);
    vkGetSwapchainImagesKHR(gDevice, gSwapchain, &imgCount, gSwapImages.data());
}

// ========================================================================
// Command pool + sync
// ========================================================================

static void createCommandPool() {
    VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    ci.queueFamilyIndex = gQueueFamily;
    VK_CHECK(vkCreateCommandPool(gDevice, &ci, nullptr, &gCmdPool));
}

static void createSyncObjects() {
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    VK_CHECK(vkCreateSemaphore(gDevice, &sci, nullptr, &gImageAvailSem));
    VK_CHECK(vkCreateSemaphore(gDevice, &sci, nullptr, &gRenderDoneSem));
    VK_CHECK(vkCreateFence(gDevice, &fci, nullptr, &gInFlightFence));
}

// ========================================================================
// RT output image
// ========================================================================

static void createOutputImage() {
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R8G8B8A8_UNORM;
    ici.extent = {gSwapExtent.width, gSwapExtent.height, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    VK_CHECK(vkCreateImage(gDevice, &ici, nullptr, &gRTImage));

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(gDevice, gRTImage, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(gDevice, &ai, nullptr, &gRTImageMem));
    vkBindImageMemory(gDevice, gRTImage, gRTImageMem, 0);

    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = gRTImage;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = VK_FORMAT_R8G8B8A8_UNORM;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VK_CHECK(vkCreateImageView(gDevice, &vci, nullptr, &gRTImageView));

    // Transition to GENERAL
    VkCommandBuffer cmd = beginCmd();
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.image = gRTImage;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        0, 0, nullptr, 0, nullptr, 1, &barrier);
    endCmd(cmd);
}

// ========================================================================
// Geometry
// ========================================================================

static GeometryData createGeometryData(const std::vector<Vertex>& verts, const std::vector<uint32_t>& idxs) {
    GeometryData g{};
    g.indexCount = (uint32_t)idxs.size();

    VkDeviceSize vSize = verts.size() * sizeof(Vertex);
    VkDeviceSize iSize = idxs.size() * sizeof(uint32_t);

    VkBufferUsageFlags usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkMemoryPropertyFlags memProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    createBuffer(vSize, usage, memProps, g.vertexBuf, g.vertexMem);
    uploadBuffer(g.vertexBuf, g.vertexMem, verts.data(), vSize);
    g.vertexAddr = getBufferAddress(g.vertexBuf);

    createBuffer(iSize, usage, memProps, g.indexBuf, g.indexMem);
    uploadBuffer(g.indexBuf, g.indexMem, idxs.data(), iSize);
    g.indexAddr = getBufferAddress(g.indexBuf);

    return g;
}

static void createGeometry() {
    // Ground plane: 10x10 quad at y=0
    {
        std::vector<Vertex> verts = {
            {{-10, 0, -10}, {0, 1, 0}},
            {{ 10, 0, -10}, {0, 1, 0}},
            {{ 10, 0,  10}, {0, 1, 0}},
            {{-10, 0,  10}, {0, 1, 0}},
        };
        std::vector<uint32_t> idxs = {0, 1, 2, 0, 2, 3};
        gGeometries.push_back(createGeometryData(verts, idxs));
    }

    // Unit cube centered at origin (-0.5 to 0.5)
    {
        auto face = [](float nx, float ny, float nz,
                       float ax, float ay, float az, float bx, float by, float bz,
                       std::vector<Vertex>& v, std::vector<uint32_t>& idx) {
            uint32_t base = (uint32_t)v.size();
            // center offset along normal
            float cx = nx * 0.5f, cy = ny * 0.5f, cz = nz * 0.5f;
            v.push_back({{cx - ax - bx, cy - ay - by, cz - az - bz}, {nx, ny, nz}});
            v.push_back({{cx + ax - bx, cy + ay - by, cz + az - bz}, {nx, ny, nz}});
            v.push_back({{cx + ax + bx, cy + ay + by, cz + az + bz}, {nx, ny, nz}});
            v.push_back({{cx - ax + bx, cy - ay + by, cz - az + bz}, {nx, ny, nz}});
            idx.insert(idx.end(), {base, base+1, base+2, base, base+2, base+3});
        };
        std::vector<Vertex> verts;
        std::vector<uint32_t> idxs;
        face( 1, 0, 0,  0, 0, 0.5f,  0, 0.5f, 0, verts, idxs); // +X
        face(-1, 0, 0,  0, 0,-0.5f,  0, 0.5f, 0, verts, idxs); // -X
        face( 0, 1, 0,  0.5f, 0, 0,  0, 0, 0.5f, verts, idxs); // +Y
        face( 0,-1, 0,  0.5f, 0, 0,  0, 0,-0.5f, verts, idxs); // -Y
        face( 0, 0, 1,  0.5f, 0, 0,  0, 0.5f, 0, verts, idxs); // +Z
        face( 0, 0,-1, -0.5f, 0, 0,  0, 0.5f, 0, verts, idxs); // -Z
        gGeometries.push_back(createGeometryData(verts, idxs));
    }

    // UV sphere (radius 1, centered at origin)
    {
        const int SLICES = 32, STACKS = 16;
        std::vector<Vertex> verts;
        std::vector<uint32_t> idxs;

        for (int j = 0; j <= STACKS; j++) {
            float phi = 3.14159265f * float(j) / float(STACKS);
            float sp = sinf(phi), cp = cosf(phi);
            for (int i = 0; i <= SLICES; i++) {
                float theta = 2.0f * 3.14159265f * float(i) / float(SLICES);
                float st = sinf(theta), ct = cosf(theta);
                float x = sp * ct, y = cp, z = sp * st;
                verts.push_back({{x, y, z}, {x, y, z}});
            }
        }
        for (int j = 0; j < STACKS; j++) {
            for (int i = 0; i < SLICES; i++) {
                uint32_t a = j * (SLICES + 1) + i;
                uint32_t b = a + SLICES + 1;
                idxs.insert(idxs.end(), {a, b, a + 1});
                idxs.insert(idxs.end(), {a + 1, b, b + 1});
            }
        }
        gGeometries.push_back(createGeometryData(verts, idxs));
    }
}

// ========================================================================
// Bottom-Level Acceleration Structures (BLAS)
// ========================================================================

static AccelStruct buildBLAS(const GeometryData& geom) {
    VkAccelerationStructureGeometryTrianglesDataKHR triData{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
    triData.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    triData.vertexData.deviceAddress = geom.vertexAddr;
    triData.vertexStride = sizeof(Vertex);
    triData.maxVertex = geom.indexCount;  // upper bound
    triData.indexType = VK_INDEX_TYPE_UINT32;
    triData.indexData.deviceAddress = geom.indexAddr;

    VkAccelerationStructureGeometryKHR asGeom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    asGeom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    asGeom.geometry.triangles = triData;
    asGeom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

    uint32_t primCount = geom.indexCount / 3;

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &asGeom;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    _vkGetAccelerationStructureBuildSizes(gDevice,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo, &primCount, &sizeInfo);

    AccelStruct as{};

    // AS buffer
    createBuffer(sizeInfo.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        as.buffer, as.memory);

    VkAccelerationStructureCreateInfoKHR asCI{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    asCI.buffer = as.buffer;
    asCI.size = sizeInfo.accelerationStructureSize;
    asCI.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    VK_CHECK(_vkCreateAccelerationStructure(gDevice, &asCI, nullptr, &as.handle));

    // Scratch buffer
    VkBuffer scratchBuf;
    VkDeviceMemory scratchMem;
    createBuffer(sizeInfo.buildScratchSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        scratchBuf, scratchMem);

    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.dstAccelerationStructure = as.handle;
    buildInfo.scratchData.deviceAddress = getBufferAddress(scratchBuf);

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount = primCount;
    const VkAccelerationStructureBuildRangeInfoKHR* pRange = &rangeInfo;

    VkCommandBuffer cmd = beginCmd();
    _vkCmdBuildAccelerationStructures(cmd, 1, &buildInfo, &pRange);
    endCmd(cmd);

    // Get device address
    VkAccelerationStructureDeviceAddressInfoKHR addrInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
    addrInfo.accelerationStructure = as.handle;
    as.address = _vkGetAccelerationStructureDeviceAddress(gDevice, &addrInfo);

    // Cleanup scratch
    vkDestroyBuffer(gDevice, scratchBuf, nullptr);
    vkFreeMemory(gDevice, scratchMem, nullptr);

    return as;
}

static void createBLAS() {
    for (auto& g : gGeometries) {
        gBLAS.push_back(buildBLAS(g));
    }
}

// ========================================================================
// Top-Level Acceleration Structure (TLAS)
// ========================================================================

static VkTransformMatrixKHR toVkTransform(const glm::mat4& m) {
    VkTransformMatrixKHR t{};
    // GLM is column-major; VkTransformMatrixKHR is row-major 3x4
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 4; c++)
            t.matrix[r][c] = m[c][r];
    return t;
}

static void createTLAS() {
    // 4 instances: ground, cube1 (sand), cube2 (mirror), glass sphere
    glm::mat4 identity(1.0f);
    glm::mat4 cube1Xform = glm::translate(identity, glm::vec3(0.0f, 0.5f, 0.0f));
    glm::mat4 cube2Xform = glm::translate(identity, glm::vec3(-2.0f, 0.5f, 1.0f));
    glm::mat4 sphereXform = glm::translate(identity, glm::vec3(2.0f, 1.0f, -1.0f));

    std::vector<VkAccelerationStructureInstanceKHR> instances(4);

    // Ground (BLAS 0, ObjDesc 0)
    instances[0] = {};
    instances[0].transform = toVkTransform(identity);
    instances[0].instanceCustomIndex = 0;
    instances[0].mask = 0xFF;
    instances[0].instanceShaderBindingTableRecordOffset = 0;
    instances[0].flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    instances[0].accelerationStructureReference = gBLAS[0].address;

    // Cube 1 - sand diffuse (BLAS 1, ObjDesc 1)
    instances[1] = {};
    instances[1].transform = toVkTransform(cube1Xform);
    instances[1].instanceCustomIndex = 1;
    instances[1].mask = 0xFF;
    instances[1].instanceShaderBindingTableRecordOffset = 0;
    instances[1].flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    instances[1].accelerationStructureReference = gBLAS[1].address;

    // Cube 2 - mirror (BLAS 1, ObjDesc 2)
    instances[2] = {};
    instances[2].transform = toVkTransform(cube2Xform);
    instances[2].instanceCustomIndex = 2;
    instances[2].mask = 0xFF;
    instances[2].instanceShaderBindingTableRecordOffset = 0;
    instances[2].flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    instances[2].accelerationStructureReference = gBLAS[1].address;

    // Glass sphere (BLAS 2, ObjDesc 3)
    instances[3] = {};
    instances[3].transform = toVkTransform(sphereXform);
    instances[3].instanceCustomIndex = 3;
    instances[3].mask = 0xFF;
    instances[3].instanceShaderBindingTableRecordOffset = 0;
    instances[3].flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    instances[3].accelerationStructureReference = gBLAS[2].address;

    // Upload instances to buffer
    VkDeviceSize instSize = instances.size() * sizeof(VkAccelerationStructureInstanceKHR);
    VkBuffer instBuf;
    VkDeviceMemory instMem;
    createBuffer(instSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        instBuf, instMem);
    uploadBuffer(instBuf, instMem, instances.data(), instSize);

    VkAccelerationStructureGeometryInstancesDataKHR instData{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR};
    instData.data.deviceAddress = getBufferAddress(instBuf);

    VkAccelerationStructureGeometryKHR asGeom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    asGeom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    asGeom.geometry.instances = instData;

    uint32_t primCount = (uint32_t)instances.size();

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &asGeom;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    _vkGetAccelerationStructureBuildSizes(gDevice,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo, &primCount, &sizeInfo);

    createBuffer(sizeInfo.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        gTLAS.buffer, gTLAS.memory);

    VkAccelerationStructureCreateInfoKHR asCI{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    asCI.buffer = gTLAS.buffer;
    asCI.size = sizeInfo.accelerationStructureSize;
    asCI.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    VK_CHECK(_vkCreateAccelerationStructure(gDevice, &asCI, nullptr, &gTLAS.handle));

    // Scratch
    VkBuffer scratchBuf;
    VkDeviceMemory scratchMem;
    createBuffer(sizeInfo.buildScratchSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        scratchBuf, scratchMem);

    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.dstAccelerationStructure = gTLAS.handle;
    buildInfo.scratchData.deviceAddress = getBufferAddress(scratchBuf);

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount = primCount;
    const VkAccelerationStructureBuildRangeInfoKHR* pRange = &rangeInfo;

    VkCommandBuffer cmd = beginCmd();
    _vkCmdBuildAccelerationStructures(cmd, 1, &buildInfo, &pRange);
    endCmd(cmd);

    VkAccelerationStructureDeviceAddressInfoKHR addrInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
    addrInfo.accelerationStructure = gTLAS.handle;
    gTLAS.address = _vkGetAccelerationStructureDeviceAddress(gDevice, &addrInfo);

    vkDestroyBuffer(gDevice, scratchBuf, nullptr);
    vkFreeMemory(gDevice, scratchMem, nullptr);
    vkDestroyBuffer(gDevice, instBuf, nullptr);
    vkFreeMemory(gDevice, instMem, nullptr);
}

// ========================================================================
// ObjDesc + UBO buffers
// ========================================================================

static void createObjDescBuffer() {
    // 4 instances: ground, cube1 (sand), cube2 (mirror), glass sphere
    std::vector<ObjDesc> descs = {
        {gGeometries[0].vertexAddr, gGeometries[0].indexAddr, {0.7f, 0.7f, 0.7f, MAT_DIFFUSE}},    // ground
        {gGeometries[1].vertexAddr, gGeometries[1].indexAddr, {0.86f, 0.78f, 0.57f, MAT_DIFFUSE}},  // cube1: sand
        {gGeometries[1].vertexAddr, gGeometries[1].indexAddr, {1.0f, 0.78f, 0.34f, MAT_MIRROR}},    // cube2: gold mirror
        {gGeometries[2].vertexAddr, gGeometries[2].indexAddr, {0.95f, 0.95f, 1.0f, MAT_GLASS}},     // glass sphere
    };

    VkDeviceSize size = descs.size() * sizeof(ObjDesc);
    createBuffer(size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        gObjDescBuf, gObjDescMem);
    uploadBuffer(gObjDescBuf, gObjDescMem, descs.data(), size);
}

static void createUBO() {
    createBuffer(sizeof(CameraUBO),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        gUBOBuf, gUBOMem);
    vkMapMemory(gDevice, gUBOMem, 0, sizeof(CameraUBO), 0, &gUBOMapped);
}

// ========================================================================
// Descriptors
// ========================================================================

static void createDescriptors() {
    // Layout
    VkDescriptorSetLayoutBinding bindings[4] = {};
    // 0: TLAS
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    // 1: Output image
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    // 2: Camera UBO
    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    // 3: ObjDesc SSBO
    bindings[3].binding = 3;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

    VkDescriptorSetLayoutCreateInfo layoutCI{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutCI.bindingCount = 4;
    layoutCI.pBindings = bindings;
    VK_CHECK(vkCreateDescriptorSetLayout(gDevice, &layoutCI, nullptr, &gDescLayout));

    // Pool
    VkDescriptorPoolSize poolSizes[] = {
        {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
    };
    VkDescriptorPoolCreateInfo poolCI{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolCI.maxSets = 1;
    poolCI.poolSizeCount = 4;
    poolCI.pPoolSizes = poolSizes;
    VK_CHECK(vkCreateDescriptorPool(gDevice, &poolCI, nullptr, &gDescPool));

    // Allocate set
    VkDescriptorSetAllocateInfo allocCI{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocCI.descriptorPool = gDescPool;
    allocCI.descriptorSetCount = 1;
    allocCI.pSetLayouts = &gDescLayout;
    VK_CHECK(vkAllocateDescriptorSets(gDevice, &allocCI, &gDescSet));

    // Write descriptors
    // 0: TLAS
    VkWriteDescriptorSetAccelerationStructureKHR asWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    asWrite.accelerationStructureCount = 1;
    asWrite.pAccelerationStructures = &gTLAS.handle;

    // 1: Output image
    VkDescriptorImageInfo imgInfo{};
    imgInfo.imageView = gRTImageView;
    imgInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    // 2: UBO
    VkDescriptorBufferInfo uboInfo{};
    uboInfo.buffer = gUBOBuf;
    uboInfo.range = sizeof(CameraUBO);

    // 3: ObjDesc
    VkDescriptorBufferInfo objInfo{};
    objInfo.buffer = gObjDescBuf;
    objInfo.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet writes[4] = {};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].pNext = &asWrite;
    writes[0].dstSet = gDescSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = gDescSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[1].pImageInfo = &imgInfo;

    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = gDescSet;
    writes[2].dstBinding = 2;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[2].pBufferInfo = &uboInfo;

    writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[3].dstSet = gDescSet;
    writes[3].dstBinding = 3;
    writes[3].descriptorCount = 1;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[3].pBufferInfo = &objInfo;

    vkUpdateDescriptorSets(gDevice, 4, writes, 0, nullptr);
}

// ========================================================================
// RT Pipeline
// ========================================================================

static void createRTPipeline() {
    auto rgenCode = readFile("shaders/raygen.rgen.spv");
    auto missCode = readFile("shaders/miss.rmiss.spv");
    auto shadowCode = readFile("shaders/shadow.rmiss.spv");
    auto chitCode = readFile("shaders/closesthit.rchit.spv");

    VkShaderModule rgenMod = createShaderModule(rgenCode);
    VkShaderModule missMod = createShaderModule(missCode);
    VkShaderModule shadowMod = createShaderModule(shadowCode);
    VkShaderModule chitMod = createShaderModule(chitCode);

    // 4 shader stages
    VkPipelineShaderStageCreateInfo stages[4] = {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stages[0].module = rgenMod;
    stages[0].pName = "main";

    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[1].module = missMod;
    stages[1].pName = "main";

    stages[2].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[2].stage = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[2].module = shadowMod;
    stages[2].pName = "main";

    stages[3].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[3].stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    stages[3].module = chitMod;
    stages[3].pName = "main";

    // 4 shader groups
    VkRayTracingShaderGroupCreateInfoKHR groups[4] = {};
    // Group 0: Raygen
    groups[0].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[0].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[0].generalShader = 0;
    groups[0].closestHitShader = VK_SHADER_UNUSED_KHR;
    groups[0].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[0].intersectionShader = VK_SHADER_UNUSED_KHR;

    // Group 1: Primary miss
    groups[1].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[1].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[1].generalShader = 1;
    groups[1].closestHitShader = VK_SHADER_UNUSED_KHR;
    groups[1].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[1].intersectionShader = VK_SHADER_UNUSED_KHR;

    // Group 2: Shadow miss
    groups[2].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[2].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[2].generalShader = 2;
    groups[2].closestHitShader = VK_SHADER_UNUSED_KHR;
    groups[2].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[2].intersectionShader = VK_SHADER_UNUSED_KHR;

    // Group 3: Closest hit
    groups[3].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[3].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    groups[3].generalShader = VK_SHADER_UNUSED_KHR;
    groups[3].closestHitShader = 3;
    groups[3].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[3].intersectionShader = VK_SHADER_UNUSED_KHR;

    // Pipeline layout
    VkPipelineLayoutCreateInfo plCI{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plCI.setLayoutCount = 1;
    plCI.pSetLayouts = &gDescLayout;
    VK_CHECK(vkCreatePipelineLayout(gDevice, &plCI, nullptr, &gPipeLayout));

    // Create RT pipeline
    VkRayTracingPipelineCreateInfoKHR rtCI{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
    rtCI.stageCount = 4;
    rtCI.pStages = stages;
    rtCI.groupCount = 4;
    rtCI.pGroups = groups;
    rtCI.maxPipelineRayRecursionDepth = 8;  // primary + reflection/refraction + shadow
    rtCI.layout = gPipeLayout;
    VK_CHECK(_vkCreateRayTracingPipelines(gDevice, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &rtCI, nullptr, &gRTPipeline));

    vkDestroyShaderModule(gDevice, rgenMod, nullptr);
    vkDestroyShaderModule(gDevice, missMod, nullptr);
    vkDestroyShaderModule(gDevice, shadowMod, nullptr);
    vkDestroyShaderModule(gDevice, chitMod, nullptr);
}

// ========================================================================
// Shader Binding Table (SBT)
// ========================================================================

static void createSBT() {
    uint32_t handleSize = gRTProps.shaderGroupHandleSize;
    uint32_t handleAlignment = gRTProps.shaderGroupHandleAlignment;
    uint32_t baseAlignment = gRTProps.shaderGroupBaseAlignment;
    uint32_t handleSizeAligned = alignUp(handleSize, handleAlignment);

    uint32_t groupCount = 4;
    uint32_t sbtSize = groupCount * handleSizeAligned;

    // Get all group handles
    std::vector<uint8_t> handles(groupCount * handleSize);
    VK_CHECK(_vkGetRayTracingShaderGroupHandles(gDevice, gRTPipeline, 0, groupCount,
                                                 handles.size(), handles.data()));

    // Compute region offsets (each region must be baseAlignment-aligned)
    // Raygen: 1 entry
    uint32_t rgenSize = alignUp(handleSizeAligned, baseAlignment);
    // Miss: 2 entries
    uint32_t missStride = handleSizeAligned;
    uint32_t missSize = alignUp(2 * handleSizeAligned, baseAlignment);
    // Hit: 1 entry
    uint32_t hitStride = handleSizeAligned;
    uint32_t hitSize = alignUp(handleSizeAligned, baseAlignment);

    uint32_t rgenOffset = 0;
    uint32_t missOffset = rgenSize;  // already aligned
    uint32_t hitOffset = missOffset + missSize;

    uint32_t totalSBTSize = hitOffset + hitSize;

    createBuffer(totalSBTSize,
        VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        gSBTBuf, gSBTMem);

    // Write handles into the SBT buffer
    void* mapped;
    vkMapMemory(gDevice, gSBTMem, 0, totalSBTSize, 0, &mapped);
    uint8_t* pSBT = (uint8_t*)mapped;

    // Group 0 → raygen region
    memcpy(pSBT + rgenOffset, handles.data() + 0 * handleSize, handleSize);
    // Group 1 → miss region entry 0
    memcpy(pSBT + missOffset + 0 * missStride, handles.data() + 1 * handleSize, handleSize);
    // Group 2 → miss region entry 1
    memcpy(pSBT + missOffset + 1 * missStride, handles.data() + 2 * handleSize, handleSize);
    // Group 3 → hit region entry 0
    memcpy(pSBT + hitOffset + 0 * hitStride, handles.data() + 3 * handleSize, handleSize);

    vkUnmapMemory(gDevice, gSBTMem);

    VkDeviceAddress sbtAddr = getBufferAddress(gSBTBuf);

    gRgenRegion.deviceAddress = sbtAddr + rgenOffset;
    gRgenRegion.stride = handleSizeAligned;
    gRgenRegion.size = rgenSize;

    gMissRegion.deviceAddress = sbtAddr + missOffset;
    gMissRegion.stride = missStride;
    gMissRegion.size = missSize;

    gHitRegion.deviceAddress = sbtAddr + hitOffset;
    gHitRegion.stride = hitStride;
    gHitRegion.size = hitSize;

    gCallRegion = {};  // unused
}

// ========================================================================
// Update camera UBO
// ========================================================================

static void updateCamera(float time) {
    float angle = time * 0.5f;
    float dist = 5.0f;
    glm::vec3 eye(sinf(angle) * dist, 2.5f, cosf(angle) * dist);
    glm::vec3 target(0.0f, 0.5f, 0.0f);
    glm::vec3 up(0.0f, 1.0f, 0.0f);

    glm::mat4 view = glm::lookAt(eye, target, up);
    glm::mat4 proj = glm::perspective(glm::radians(60.0f),
                                       (float)gSwapExtent.width / (float)gSwapExtent.height,
                                       0.1f, 100.0f);

    CameraUBO ubo;
    ubo.viewInverse = glm::inverse(view);
    ubo.projInverse = glm::inverse(proj);
    memcpy(gUBOMapped, &ubo, sizeof(ubo));
}

// ========================================================================
// Render frame
// ========================================================================

static void drawFrame(float time) {
    vkWaitForFences(gDevice, 1, &gInFlightFence, VK_TRUE, UINT64_MAX);
    vkResetFences(gDevice, 1, &gInFlightFence);

    uint32_t imageIndex;
    VK_CHECK(vkAcquireNextImageKHR(gDevice, gSwapchain, UINT64_MAX, gImageAvailSem, VK_NULL_HANDLE, &imageIndex));

    updateCamera(time);

    // Record command buffer
    VkCommandBuffer cmd = beginCmd();

    // Bind RT pipeline and descriptors
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, gRTPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                            gPipeLayout, 0, 1, &gDescSet, 0, nullptr);

    // Trace rays
    _vkCmdTraceRays(cmd,
                    &gRgenRegion, &gMissRegion, &gHitRegion, &gCallRegion,
                    gSwapExtent.width, gSwapExtent.height, 1);

    // Transition RT image: GENERAL → TRANSFER_SRC
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.image = gRTImage;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    // Transition swapchain image: UNDEFINED → TRANSFER_DST
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.image = gSwapImages[imageIndex];
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    // Copy RT image → swapchain image
    VkImageCopy copyRegion{};
    copyRegion.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copyRegion.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copyRegion.extent = {gSwapExtent.width, gSwapExtent.height, 1};
    vkCmdCopyImage(cmd,
        gRTImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        gSwapImages[imageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1, &copyRegion);

    // Transition swapchain image: TRANSFER_DST → PRESENT_SRC
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    barrier.image = gSwapImages[imageIndex];
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = 0;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    // Transition RT image back: TRANSFER_SRC → GENERAL
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.image = gRTImage;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    vkEndCommandBuffer(cmd);

    // Submit
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = &gImageAvailSem;
    si.pWaitDstStageMask = &waitStage;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &gRenderDoneSem;
    VK_CHECK(vkQueueSubmit(gQueue, 1, &si, gInFlightFence));

    // Present
    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &gRenderDoneSem;
    pi.swapchainCount = 1;
    pi.pSwapchains = &gSwapchain;
    pi.pImageIndices = &imageIndex;
    vkQueuePresentKHR(gQueue, &pi);
}

// ========================================================================
// Cleanup
// ========================================================================

static void cleanup() {
    vkDeviceWaitIdle(gDevice);

    vkDestroyBuffer(gDevice, gSBTBuf, nullptr);
    vkFreeMemory(gDevice, gSBTMem, nullptr);
    vkDestroyPipeline(gDevice, gRTPipeline, nullptr);
    vkDestroyPipelineLayout(gDevice, gPipeLayout, nullptr);
    vkDestroyDescriptorPool(gDevice, gDescPool, nullptr);
    vkDestroyDescriptorSetLayout(gDevice, gDescLayout, nullptr);

    vkUnmapMemory(gDevice, gUBOMem);
    vkDestroyBuffer(gDevice, gUBOBuf, nullptr);
    vkFreeMemory(gDevice, gUBOMem, nullptr);
    vkDestroyBuffer(gDevice, gObjDescBuf, nullptr);
    vkFreeMemory(gDevice, gObjDescMem, nullptr);

    _vkDestroyAccelerationStructure(gDevice, gTLAS.handle, nullptr);
    vkDestroyBuffer(gDevice, gTLAS.buffer, nullptr);
    vkFreeMemory(gDevice, gTLAS.memory, nullptr);

    for (auto& b : gBLAS) {
        _vkDestroyAccelerationStructure(gDevice, b.handle, nullptr);
        vkDestroyBuffer(gDevice, b.buffer, nullptr);
        vkFreeMemory(gDevice, b.memory, nullptr);
    }

    for (auto& g : gGeometries) {
        vkDestroyBuffer(gDevice, g.vertexBuf, nullptr);
        vkFreeMemory(gDevice, g.vertexMem, nullptr);
        vkDestroyBuffer(gDevice, g.indexBuf, nullptr);
        vkFreeMemory(gDevice, g.indexMem, nullptr);
    }

    vkDestroyImageView(gDevice, gRTImageView, nullptr);
    vkDestroyImage(gDevice, gRTImage, nullptr);
    vkFreeMemory(gDevice, gRTImageMem, nullptr);

    vkDestroySemaphore(gDevice, gImageAvailSem, nullptr);
    vkDestroySemaphore(gDevice, gRenderDoneSem, nullptr);
    vkDestroyFence(gDevice, gInFlightFence, nullptr);
    vkDestroyCommandPool(gDevice, gCmdPool, nullptr);
    vkDestroySwapchainKHR(gDevice, gSwapchain, nullptr);
    vkDestroyDevice(gDevice, nullptr);
    vkDestroySurfaceKHR(gInstance, gSurface, nullptr);
    if (gDebugMessenger && vkDestroyDebugUtilsMessengerEXT)
        vkDestroyDebugUtilsMessengerEXT(gInstance, gDebugMessenger, nullptr);
    vkDestroyInstance(gInstance, nullptr);
    glfwDestroyWindow(gWindow);
    glfwTerminate();
}

// ========================================================================
// Main
// ========================================================================

int main() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    gWindow = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan RT Sample", nullptr, nullptr);
    if (!gWindow) { fprintf(stderr, "Failed to create window\n"); return 1; }

    loadGlobalFunctions();
    initInstance();

    // Create surface
    VK_CHECK(glfwCreateWindowSurface(gInstance, gWindow, nullptr, &gSurface));

    pickPhysicalDevice();
    createDevice();
    createSwapchain();
    createCommandPool();
    createSyncObjects();
    createOutputImage();
    createGeometry();
    createBLAS();
    createTLAS();
    createObjDescBuffer();
    createUBO();
    createDescriptors();
    createRTPipeline();
    createSBT();

    printf("Vulkan RT initialized. Rendering...\n");

    auto startTime = std::chrono::high_resolution_clock::now();

    while (!glfwWindowShouldClose(gWindow)) {
        glfwPollEvents();

        auto now = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float>(now - startTime).count();

        drawFrame(time);
    }

    cleanup();
    return 0;
}
