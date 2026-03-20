# Jetson Orin 本地环境安装教程（无 Docker）

根据 `easy_deploy_tool/docker/jetson_tensorrt_trt10_u2204.dockerfile` 整理，目标是在 **Jetson Orin（aarch64, Ubuntu 22.04 / jammy）** 上复现相同依赖环境。

---

## 一、适用范围与目标版本

- **平台**：Jetson Orin（AGX Orin / Orin NX / Orin Nano）
- **架构**：`aarch64`
- **系统**：Ubuntu 22.04（jammy）
- **CUDA 基线**：`l4t-cuda:12.2.12-devel`（Dockerfile 基础镜像）
- **TensorRT**：`10.7.0.23`（l4t aarch64, cuda12.6 tar 包）
- **CV-CUDA**：`0.12.0-beta`（aarch64, cuda12）
- **CMake**：`3.22.3`（aarch64 tar 包）
- **glog**：`0.7.0`（源码编译）

---

## 二、环境自检

```bash
# 系统与架构
cat /etc/os-release
uname -m

# Jetson/L4T 信息（存在时）
cat /etc/nv_tegra_release 2>/dev/null || true

# CUDA 是否可用
nvcc --version

# TensorRT 运行库（若已装）
ldconfig -p | rg -i nvinfer || true
```

期望：

- `uname -m` 为 `aarch64`
- Ubuntu 为 `22.04 (jammy)`

---

## 三、切换 apt 源到 ubuntu-ports（与 Dockerfile 一致）

Dockerfile 使用 USTC 的 `ubuntu-ports` 源。宿主机也可对齐（可按需替换为你常用镜像）。

```bash
sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak.$(date +%Y%m%d_%H%M%S)

sudo tee /etc/apt/sources.list >/dev/null <<'EOF'
deb https://mirrors.ustc.edu.cn/ubuntu-ports/ jammy main restricted universe multiverse
deb https://mirrors.ustc.edu.cn/ubuntu-ports/ jammy-security main restricted universe multiverse
deb https://mirrors.ustc.edu.cn/ubuntu-ports/ jammy-updates main restricted universe multiverse
deb https://mirrors.ustc.edu.cn/ubuntu-ports/ jammy-backports main restricted universe multiverse
EOF

sudo apt-get update
```

---

## 四、安装基础依赖（逐项对齐 Dockerfile）

```bash
sudo apt-get install -y \
  build-essential \
  manpages-dev \
  wget \
  zlib1g \
  software-properties-common \
  git \
  libssl-dev \
  zlib1g-dev \
  libbz2-dev \
  libreadline-dev \
  libsqlite3-dev \
  ca-certificates \
  curl \
  llvm \
  libncurses5-dev \
  xz-utils \
  tk-dev \
  libxml2-dev \
  libxmlsec1-dev \
  libffi-dev \
  liblzma-dev \
  mecab-ipadic-utf8 \
  libopencv-dev \
  libeigen3-dev \
  libgtest-dev \
  libassimp-dev
```

---

## 五、安装 CMake 3.22.3（aarch64）

Dockerfile 使用 aarch64 预编译包并软链接到 `/usr/local/bin/cmake`。

```bash
cd /tmp
wget https://gp.zz990099.cn/https://github.com/Kitware/CMake/releases/download/v3.22.3/cmake-3.22.3-linux-aarch64.tar.gz
tar -xzvf cmake-3.22.3-linux-aarch64.tar.gz
sudo mv cmake-3.22.3-linux-aarch64 /opt/cmake-3.22.3
sudo ln -sf /opt/cmake-3.22.3/bin/cmake /usr/local/bin/cmake

cmake --version
```

---

## 六、安装 glog 0.7.0（源码编译）

```bash
cd /tmp
wget https://gp.zz990099.cn/https://github.com/google/glog/archive/refs/tags/v0.7.0.tar.gz
tar -xzvf v0.7.0.tar.gz
cd glog-0.7.0
mkdir -p build && cd build
cmake ..
make -j"$(nproc)"
sudo make install
sudo ldconfig
```

---

## 七、安装 CV-CUDA（aarch64, cuda12）

```bash
cd /tmp
wget https://gp.zz990099.cn/https://github.com/CVCUDA/CV-CUDA/releases/download/v0.12.0-beta/cvcuda-lib-0.12.0_beta-cuda12-aarch64-linux.deb
wget https://gp.zz990099.cn/https://github.com/CVCUDA/CV-CUDA/releases/download/v0.12.0-beta/cvcuda-dev-0.12.0_beta-cuda12-aarch64-linux.deb

sudo dpkg -i cvcuda-lib-0.12.0_beta-cuda12-aarch64-linux.deb
sudo dpkg -i cvcuda-dev-0.12.0_beta-cuda12-aarch64-linux.deb
sudo ldconfig
```

---

## 八、安装 TensorRT 10.7.0.23（L4T aarch64 tar）

Dockerfile 采用 tar 安装并把库/头文件拷贝到系统路径。

```bash
cd /tmp
wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.7.0/tars/TensorRT-10.7.0.23.l4t.aarch64-gnu.cuda-12.6.tar.gz
tar -xzvf TensorRT-10.7.0.23.l4t.aarch64-gnu.cuda-12.6.tar.gz
sudo mv TensorRT-10.7.0.23 /usr/src/tensorrt

# 复制运行库和头文件到系统目录（与 Dockerfile 一致）
sudo cp /usr/src/tensorrt/lib/*.so* /usr/lib/aarch64-linux-gnu/
sudo cp /usr/src/tensorrt/include/* /usr/include/aarch64-linux-gnu/
sudo ldconfig
```

可选：编译 samples（Dockerfile 里有）

```bash
cd /usr/src/tensorrt/samples
sudo make -j"$(nproc)"
```

---

## 九、环境变量（建议）

多数 Jetson 环境已内置 CUDA 路径；仍建议显式补充：

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

---

## 十、验证

```bash
# CUDA
nvcc --version

# TensorRT 头文件/库
ls /usr/include/aarch64-linux-gnu/NvInfer*.h
ldconfig -p | rg -i "nvinfer|nvonnx"

# CV-CUDA（存在即可）
ldconfig -p | rg -i cvcuda || true

# CMake / glog
cmake --version
ldconfig -p | rg -i glog || true
```

---

## 十一、常见问题

- **TensorRT 与 JetPack 自带版本冲突**  
  如果系统已有另一版 TensorRT，优先保证运行时实际加载的是 `/usr/lib/aarch64-linux-gnu` 下你刚复制的版本（可用 `ldconfig -p` 和 `ldd` 检查）。

- **下载慢或失败**  
  文档已优先使用 Dockerfile 同款加速前缀 `https://gp.zz990099.cn/https://github.com/...`。若不可用，改回 GitHub 原始链接重试。

- **系统非 jammy**  
  本文与 Dockerfile 对齐为 jammy。若是其他版本，建议先切到 Ubuntu 22.04（JetPack 对应版本）再执行。

---

按以上步骤可在 Jetson Orin 宿主机上复现 `jetson_tensorrt_trt10_u2204.dockerfile` 的核心依赖环境。
