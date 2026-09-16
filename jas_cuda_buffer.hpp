#ifndef __JAS_CUDA_BUFFER_HPP__
#define __JAS_CUDA_BUFFER_HPP__

/**
 * CUDA 运行时基础设施：错误检查、设备查询、设备缓冲区。
 *
 * 这个头**只在 CUDA 构建里用**（它直接依赖 CUDA 运行时 API）。
 * 想让表达式模板本身可用于设备，看 jas_cuda_leaf.hpp —— 那个头不依赖运行时，
 * 纯 CPU 构建也能 include。
 */

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

namespace jasmine {
namespace cuda {

/** CUDA 调用失败。带上出错的调用点，省得每次都去翻 cudaGetErrorString。 */
class cuda_error : public std::runtime_error
{
public:
    cuda_error(cudaError_t code, const char* expr, const char* file, int line)
        : std::runtime_error(build_message(code, expr, file, line)), m_code(code)
    {
    }

    cudaError_t code() const noexcept { return m_code; }

private:
    static std::string build_message(cudaError_t code, const char* expr, const char* file, int line)
    {
        std::string msg = "CUDA 调用失败: ";
        msg += cudaGetErrorName(code);
        msg += " (";
        msg += cudaGetErrorString(code);
        msg += ")\n  调用: ";
        msg += expr;
        msg += "\n  位置: ";
        msg += file;
        msg += ":";
        msg += std::to_string(line);
        return msg;
    }

    cudaError_t m_code;
};

} // namespace cuda
} // namespace jasmine

/** 检查一次 CUDA 调用的返回值，失败即抛。 */
#define JAS_CUDA_CHECK(expr)                                                       \
    do {                                                                           \
        ::cudaError_t _jas_err = (expr);                                           \
        if (_jas_err != ::cudaSuccess)                                             \
            throw ::jasmine::cuda::cuda_error(_jas_err, #expr, __FILE__, __LINE__); \
    } while (0)

namespace jasmine {
namespace cuda {

/** 设备概要。只查一次就缓存，避免反复问驱动。 */
struct device_info_t
{
    std::string name;
    int major = 0;
    int minor = 0;
    std::size_t total_mem = 0;
    int sm_count = 0;
    int max_threads_per_block = 0;

    int compute_capability() const { return major * 10 + minor; }

    std::string to_string() const
    {
        std::string s = name + " (sm_" + std::to_string(compute_capability()) + ", "
                        + std::to_string(sm_count) + " SMs, "
                        + std::to_string(total_mem / (1024 * 1024)) + " MiB)";
        return s;
    }
};

inline const device_info_t& device_info()
{
    static device_info_t info = [] {
        int count = 0;
        JAS_CUDA_CHECK(cudaGetDeviceCount(&count));
        if (count <= 0)
            throw std::runtime_error("没有可见的 CUDA 设备");

        cudaDeviceProp prop{};
        JAS_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

        device_info_t d;
        d.name = prop.name;
        d.major = prop.major;
        d.minor = prop.minor;
        d.total_mem = static_cast<std::size_t>(prop.totalGlobalMem);
        d.sm_count = prop.multiProcessorCount;
        d.max_threads_per_block = prop.maxThreadsPerBlock;
        return d;
    }();
    return info;
}

/**
 * 设备缓冲区：RAII 持有显存，只可移动不可拷贝。
 *
 * 刻意做成「零初始化」的裸数组而不是 `mat_t` 的镜像 —— 设备叶子 `dev_mat_t`
 * 只是个薄壳（指针 + 维度），真正管内存的是这里的 dev_buf_t。
 * 两者分开之后，表达式树里存的就只是薄壳，可以随便拷贝、可以按值塞进 kernel。
 */
template <typename T>
class dev_buf_t
{
public:
    dev_buf_t() noexcept = default;

    explicit dev_buf_t(std::size_t n)
    {
        allocate(n);
    }

    ~dev_buf_t() noexcept
    {
        release();
    }

    dev_buf_t(const dev_buf_t&) = delete;
    dev_buf_t& operator=(const dev_buf_t&) = delete;

    dev_buf_t(dev_buf_t&& other) noexcept
        : m_data(other.m_data), m_size(other.m_size)
    {
        other.m_data = nullptr;
        other.m_size = 0;
    }

    dev_buf_t& operator=(dev_buf_t&& other) noexcept
    {
        if (this != &other)
        {
            release();
            m_data = other.m_data;
            m_size = other.m_size;
            other.m_data = nullptr;
            other.m_size = 0;
        }
        return *this;
    }

    void allocate(std::size_t n)
    {
        if (n == m_size && m_data != nullptr)
            return;
        release();
        if (n == 0)
            return;
        JAS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_data), n * sizeof(T)));
        m_size = n;
    }

    void release() noexcept
    {
        if (m_data != nullptr)
        {
            // 析构路径不方便抛异常，失败最多是泄漏一点点显存，仍然报出来
            cudaError_t err = cudaFree(m_data);
            if (err != cudaSuccess)
                std::fprintf(stderr, "警告: cudaFree 失败: %s\n", cudaGetErrorString(err));
            m_data = nullptr;
            m_size = 0;
        }
    }

    T* data() noexcept { return m_data; }
    const T* data() const noexcept { return m_data; }
    std::size_t size() const noexcept { return m_size; }
    bool empty() const noexcept { return m_size == 0; }
    std::size_t bytes() const noexcept { return m_size * sizeof(T); }

    /** 主机 → 设备 */
    void upload(const T* host, std::size_t n)
    {
        if (n > m_size)
            throw std::out_of_range("dev_buf_t::upload: 源数据比缓冲区还大");
        JAS_CUDA_CHECK(cudaMemcpy(m_data, host, n * sizeof(T), cudaMemcpyHostToDevice));
    }

    /** 设备 → 主机 */
    void download(T* host, std::size_t n) const
    {
        if (n > m_size)
            throw std::out_of_range("dev_buf_t::download: 目标比缓冲区还大");
        JAS_CUDA_CHECK(cudaMemcpy(host, m_data, n * sizeof(T), cudaMemcpyDeviceToHost));
    }

    void zero()
    {
        if (m_data != nullptr)
            JAS_CUDA_CHECK(cudaMemset(m_data, 0, bytes()));
    }

private:
    T* m_data = nullptr;
    std::size_t m_size = 0;
};

/** 等设备把已提交的工作做完。计时、取值、以及和主机对拍之前都用它兜底。 */
inline void sync()
{
    JAS_CUDA_CHECK(cudaDeviceSynchronize());
}

/** 分配主机端「钉住的」（page-locked）内存，用于高频 H2D/D2H 时避免分页开销。 */
template <typename T>
class pinned_buf_t
{
public:
    explicit pinned_buf_t(std::size_t n)
        : m_size(n)
    {
        JAS_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&m_data), n * sizeof(T),
                                     cudaHostAllocDefault));
    }

    ~pinned_buf_t() noexcept
    {
        if (m_data != nullptr)
            cudaFreeHost(m_data);
    }

    pinned_buf_t(const pinned_buf_t&) = delete;
    pinned_buf_t& operator=(const pinned_buf_t&) = delete;

    T* data() noexcept { return m_data; }
    const T* data() const noexcept { return m_data; }
    std::size_t size() const noexcept { return m_size; }

private:
    T* m_data = nullptr;
    std::size_t m_size = 0;
};

} // namespace cuda
} // namespace jasmine

#endif
