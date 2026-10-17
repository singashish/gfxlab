#include "renderer/vulkan/vk_context.h"

#include <algorithm>
#include <cstring>
#include <vector>

namespace gfx {
namespace {

constexpr const char* kValidationLayer =
    "VK_LAYER_KHRONOS_validation";

#ifdef NDEBUG
constexpr bool kRequestValidation = false;
#else
constexpr bool kRequestValidation = true;
#endif

VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
    void*) {

    const char* message =
        callback_data && callback_data->pMessage
            ? callback_data->pMessage
            : "Unknown Vulkan validation message";

    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        LOG_ERROR("[Vulkan validation] %s", message);
    } else {
        LOG_WARN("[Vulkan validation] %s", message);
    }

    return VK_FALSE;
}

VkDebugUtilsMessengerCreateInfoEXT make_debug_info() {
    VkDebugUtilsMessengerCreateInfoEXT info{};
    info.sType =
        VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;

    info.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;

    info.messageType =
        VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;

    info.pfnUserCallback = debug_callback;
    return info;
}

bool has_instance_layer(const char* requested) {
    std::uint32_t count = 0;

    if (vkEnumerateInstanceLayerProperties(&count, nullptr)
        != VK_SUCCESS) {
        LOG_ERROR("failed to enumerate Vulkan instance layers");
        return false;
    }

    std::vector<VkLayerProperties> layers(count);

    if (vkEnumerateInstanceLayerProperties(&count, layers.data())
        != VK_SUCCESS) {
        LOG_ERROR("failed to read Vulkan instance layers");
        return false;
    }

    return std::any_of(
        layers.begin(), layers.end(),
        [requested](const VkLayerProperties& layer) {
            return std::strcmp(layer.layerName, requested) == 0;
        });
}

bool has_instance_extension(const char* requested) {
    std::uint32_t count = 0;

    if (vkEnumerateInstanceExtensionProperties(
            nullptr, &count, nullptr) != VK_SUCCESS) {
        LOG_ERROR("failed to enumerate Vulkan instance extensions");
        return false;
    }

    std::vector<VkExtensionProperties> extensions(count);

    if (vkEnumerateInstanceExtensionProperties(
            nullptr, &count, extensions.data()) != VK_SUCCESS) {
        LOG_ERROR("failed to read Vulkan instance extensions");
        return false;
    }

    return std::any_of(
        extensions.begin(), extensions.end(),
        [requested](const VkExtensionProperties& extension) {
            return std::strcmp(extension.extensionName, requested) == 0;
        });
}

bool has_device_extension(VkPhysicalDevice device, const char* requested) {

    std::uint32_t count = 0;

    if (vkEnumerateDeviceExtensionProperties(
            device, nullptr, &count, nullptr) != VK_SUCCESS) {
        return false;
    }

    std::vector<VkExtensionProperties> extensions(count);

    if (vkEnumerateDeviceExtensionProperties(
            device, nullptr, &count, extensions.data()) != VK_SUCCESS) {
        return false;
    }

    return std::any_of(
        extensions.begin(), extensions.end(),
        [requested](const VkExtensionProperties& extension) {
            return std::strcmp(extension.extensionName, requested) == 0;
        });
}

const char* device_type_name(VkPhysicalDeviceType type) {
    switch (type) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
        return "discrete";
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
        return "integrated";
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
        return "virtual";
    case VK_PHYSICAL_DEVICE_TYPE_CPU:
        return "CPU";
    default:
        return "other";
    }
}

bool find_graphics_present_queue(VkPhysicalDevice device,
            VkSurfaceKHR surface, std::uint32_t& selected_family) {

    std::uint32_t count = 0;

    vkGetPhysicalDeviceQueueFamilyProperties(
        device, &count, nullptr);
        
    if (count == 0) {
        return false;
    }

    std::vector<VkQueueFamilyProperties> families(count);

    vkGetPhysicalDeviceQueueFamilyProperties(
        device, &count, families.data());

    for (std::uint32_t i = 0; i < count; ++i) {
        if (families[i].queueCount == 0 ||
            (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0) {
            continue;
        }

        VkBool32 supports_present = VK_FALSE;

        if (vkGetPhysicalDeviceSurfaceSupportKHR(
                device, i, surface, &supports_present) != VK_SUCCESS) {
            continue;
        }

        if (supports_present == VK_TRUE) {
            selected_family = i;
            return true;
        }
    }

    return false;
}

} // namespace

VulkanContext::~VulkanContext() {
    destroy();
}

void VulkanContext::destroy() {
    // 1. The logical device must be destroyed before the instance.
    if (m_device != VK_NULL_HANDLE) {
        vkDestroyDevice(m_device, nullptr);
        m_device = VK_NULL_HANDLE;
    }

    m_queue = VK_NULL_HANDLE;
    m_physical_device = VK_NULL_HANDLE;
    m_queue_family = 0;

    // 2. The surface belongs to the instance.
    if (m_surface != VK_NULL_HANDLE &&
        m_instance != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
        m_surface = VK_NULL_HANDLE;
    }

    // 3. Destroy the debug messenger while the instance still exists.
    if (m_debug_messenger != VK_NULL_HANDLE &&
        m_instance != VK_NULL_HANDLE) {
        if (m_destroy_debug_messenger != nullptr) {
            m_destroy_debug_messenger(
                m_instance, m_debug_messenger, nullptr);
        } else {
            LOG_ERROR(
                "cannot destroy Vulkan debug messenger: "
                "destroy function unavailable");
        }

        m_debug_messenger = VK_NULL_HANDLE;
        m_destroy_debug_messenger = nullptr;
    }

    // 4. Destroy the instance last.
    if (m_instance != VK_NULL_HANDLE) {
        vkDestroyInstance(m_instance, nullptr);
        m_instance = VK_NULL_HANDLE;
    }

    m_validation_enabled = false;
}

bool VulkanContext::create(void* native_window) {
    destroy();

    if (native_window == nullptr) {
        LOG_ERROR("Vulkan initialization failed: HWND is null");
        return false;
    }

    // Step 1: Check the Vulkan loader version.
    std::uint32_t loader_version = VK_API_VERSION_1_0;

    if (!vk_ok(vkEnumerateInstanceVersion(&loader_version),
               "query Vulkan loader version")) {
        destroy();
        return false;
    }

    LOG_INFO("Vulkan loader version: %u.%u.%u",
        VK_API_VERSION_MAJOR(loader_version),
        VK_API_VERSION_MINOR(loader_version),
        VK_API_VERSION_PATCH(loader_version));

    if (loader_version < VK_API_VERSION_1_3) {
        LOG_ERROR("Vulkan 1.3 or newer loader is required");
        destroy();
        return false;
    }

    // Step 2: Enable validation in debug builds when available.
    if (kRequestValidation) {
        m_validation_enabled = has_instance_layer(kValidationLayer);

        if (m_validation_enabled) {
            LOG_INFO("Vulkan validation layer enabled");
        } else {
            LOG_WARN(
                "VK_LAYER_KHRONOS_validation not found; "
                "continuing without validation");
        }
    }

    // Step 3: Check the required instance extensions.
    std::vector<const char*> extensions = {
        VK_KHR_SURFACE_EXTENSION_NAME,
        VK_KHR_WIN32_SURFACE_EXTENSION_NAME
    };

    if (m_validation_enabled) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    for (const char* extension : extensions) {
        if (!has_instance_extension(extension)) {
            LOG_ERROR(
                "required Vulkan instance extension missing: %s",
                extension);
            destroy();
            return false;
        }
    }

    std::vector<const char*> layers;

    if (m_validation_enabled) {
        layers.push_back(kValidationLayer);
    }

    // Step 4: Create the instance.
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "gfxlab";
    app_info.applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
    app_info.pEngineName = "gfxlab";
    app_info.engineVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_3;

    VkDebugUtilsMessengerCreateInfoEXT debug_info =
        make_debug_info();

    VkInstanceCreateInfo instance_info{};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &app_info;
    instance_info.enabledExtensionCount =
        static_cast<std::uint32_t>(extensions.size());
    instance_info.ppEnabledExtensionNames = extensions.data();
    instance_info.enabledLayerCount =
        static_cast<std::uint32_t>(layers.size());
    instance_info.ppEnabledLayerNames =
        layers.empty() ? nullptr : layers.data();

    // Also capture applicable messages during instance creation.
    if (m_validation_enabled) {
        instance_info.pNext = &debug_info;
    }

    if (!vk_ok(vkCreateInstance(
                   &instance_info, nullptr, &m_instance),
               "create Vulkan instance")) {
        destroy();
        return false;
    }

    // Step 5: Load and create the debug messenger.
    if (m_validation_enabled) {
        auto create_debug =
            reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(
                    m_instance, "vkCreateDebugUtilsMessengerEXT"));

        m_destroy_debug_messenger =
            reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(
                    m_instance, "vkDestroyDebugUtilsMessengerEXT"));

        if (create_debug == nullptr ||
            m_destroy_debug_messenger == nullptr) {
            LOG_ERROR(
                "failed to load Vulkan debug messenger functions");
            destroy();
            return false;
        }

        if (!vk_ok(create_debug(
                       m_instance, &debug_info, nullptr,
                       &m_debug_messenger),
                   "create Vulkan debug messenger")) {
            destroy();
            return false;
        }
    }

    // Step 6: Create a surface for the existing Win32 HWND.
    VkWin32SurfaceCreateInfoKHR surface_info{};
    surface_info.sType =
        VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    surface_info.hinstance = GetModuleHandleW(nullptr);
    surface_info.hwnd = static_cast<HWND>(native_window);

    if (!vk_ok(vkCreateWin32SurfaceKHR(
                   m_instance, &surface_info, nullptr, &m_surface),
               "create Win32 surface")) {
        destroy();
        return false;
    }

    // Step 7: Enumerate and score physical devices.
    std::uint32_t device_count = 0;

    if (!vk_ok(vkEnumeratePhysicalDevices(
                   m_instance, &device_count, nullptr),
               "enumerate physical devices")) {
        destroy();
        return false;
    }

    if (device_count == 0) {
        LOG_ERROR("no Vulkan physical devices found");
        destroy();
        return false;
    }

    std::vector<VkPhysicalDevice> devices(device_count);

    if (!vk_ok(vkEnumeratePhysicalDevices(
                   m_instance, &device_count, devices.data()),
               "read physical devices")) {
        destroy();
        return false;
    }

    VkPhysicalDevice best_device = VK_NULL_HANDLE;
    std::uint32_t best_queue_family = 0;
    int best_score = -1;

    for (VkPhysicalDevice candidate : devices) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(candidate, &properties);

        LOG_INFO(
            "Vulkan device: %s | type=%s | API=%u.%u.%u",
            properties.deviceName,
            device_type_name(properties.deviceType),
            VK_API_VERSION_MAJOR(properties.apiVersion),
            VK_API_VERSION_MINOR(properties.apiVersion),
            VK_API_VERSION_PATCH(properties.apiVersion));

        if (properties.apiVersion < VK_API_VERSION_1_3) {
            LOG_WARN("  rejected: Vulkan 1.3 is required");
            continue;
        }

        if (!has_device_extension(
                candidate, VK_KHR_SWAPCHAIN_EXTENSION_NAME)) {
            LOG_WARN("  rejected: VK_KHR_swapchain is unavailable");
            continue;
        }

        VkPhysicalDeviceVulkan13Features features13{};
        features13.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;

        VkPhysicalDeviceFeatures2 features2{};
        features2.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2.pNext = &features13;

        vkGetPhysicalDeviceFeatures2(candidate, &features2);

        if (!features13.dynamicRendering ||
            !features13.synchronization2) {
            LOG_WARN(
                "  rejected: dynamicRendering or synchronization2 "
                "is unavailable");
            continue;
        }

        std::uint32_t queue_family = 0;

        if (!find_graphics_present_queue(
                candidate, m_surface, queue_family)) {
            LOG_WARN(
                "  rejected: no queue family supports graphics "
                "and presentation");
            continue;
        }

        int score = 0;

        if (properties.deviceType ==
            VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            score += 1000;
        } else if (properties.deviceType ==
                   VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
            score += 100;
        }

        LOG_INFO("  suitable: score=%d, queue family=%u",
                 score, queue_family);

        if (score > best_score) {
            best_score = score;
            best_device = candidate;
            best_queue_family = queue_family;
        }
    }

    if (best_device == VK_NULL_HANDLE) {
        LOG_ERROR(
            "no GPU satisfies the Vulkan 1.3 requirements");
        destroy();
        return false;
    }

    m_physical_device = best_device;
    m_queue_family = best_queue_family;

    VkPhysicalDeviceProperties selected_properties{};
    vkGetPhysicalDeviceProperties(
        m_physical_device, &selected_properties);

    LOG_INFO("selected Vulkan GPU: %s",
             selected_properties.deviceName);

    // Step 8: Create the logical device.
    constexpr float queue_priority = 1.0f;

    VkDeviceQueueCreateInfo queue_info{};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = m_queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &queue_priority;

    VkPhysicalDeviceVulkan13Features enabled_features{};
    enabled_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    enabled_features.dynamicRendering = VK_TRUE;
    enabled_features.synchronization2 = VK_TRUE;

    const char* device_extensions[] = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

    VkDeviceCreateInfo device_info{};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.pNext = &enabled_features;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.enabledExtensionCount = 1;
    device_info.ppEnabledExtensionNames = device_extensions;
    device_info.pEnabledFeatures = nullptr;

    if (!vk_ok(vkCreateDevice(
                   m_physical_device, &device_info, nullptr, &m_device),
               "create logical device")) {
        destroy();
        return false;
    }

    vkGetDeviceQueue(m_device, m_queue_family, 0, &m_queue);

    if (m_queue == VK_NULL_HANDLE) {
        LOG_ERROR("vkGetDeviceQueue returned a null queue");
        destroy();
        return false;
    }

    LOG_INFO("Vulkan context initialized successfully");
    return true;
}

} // namespace gfx
