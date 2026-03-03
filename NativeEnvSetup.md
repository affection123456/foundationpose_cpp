# 本地环境安装教程（无 Docker）

根据 Dockerfile 要求，在宿主机上安装 CUDA、cuDNN、TensorRT 及依赖，适用于 **Ubuntu 20.04 / 22.04 / 24.04**（x86_64）。

---

## 一、当前环境检测结果（参考）

在项目根目录或任意终端执行以下命令可自检：

```bash
# 系统与架构
cat /etc/os-release
uname -m   # 需为 x86_64

# 显卡与驱动
nvidia-smi
lspci | grep -i nvidia

# 是否已装 CUDA Toolkit / cuDNN
ls -la /usr/local/cuda
nvcc --version
ldconfig -p | grep -i cudnn
```

**典型结论：**

- **Ubuntu 版本**：决定后面使用的 repo 代号（focal=20.04, jammy=22.04, noble=24.04）。
- **NVIDIA 驱动**：需已安装且版本满足对应 CUDA 要求（见下方版本表）。
- **CUDA / cuDNN**：若未安装，按本教程安装。

---

## 二、版本对应关系（与 Dockerfile 一致）

| 方案 | 系统 (示例) | CUDA | cuDNN | TensorRT | 说明 |
|------|-------------|------|-------|----------|------|
| **TRT10** | Ubuntu 22.04 | 12.3.2 | 9 | 10.7.0 | 需 CV-CUDA、assimp，见 TRT10 Dockerfile |
| **TRT8**  | Ubuntu 20.04 | 11.7.1 | 8 | 8.6.1 | 与 TRT8 focal Dockerfile 一致 |
| **TRT8**  | Ubuntu 22.04 | 11.7.1 | 8 | 8.6.1 | 与 TRT8 jammy Dockerfile 一致 |

**Ubuntu 24.04**：官方已有 `ubuntu2404` 的 CUDA/cuDNN 源，优先用 **CUDA 12.x + cuDNN 9 + TensorRT 10** 方案；若要用 TRT8，需确认 NVIDIA 是否提供 ubuntu2404 的 TensorRT 8 包，否则可尝试用 ubuntu2204 源或 Tar 安装。

---

## 三、确定 Ubuntu 代号与 CUDA 源

```bash
. /etc/os-release && echo "UBUNTU_CODENAME=$VERSION_CODENAME"
```

记下 `VERSION_CODENAME`（如 `noble`），下面用 `$UBUNTU` 表示，并选择对应 CUDA 源：

- Ubuntu 20.04 → `focal`，源路径：`ubuntu2004`
- Ubuntu 22.04 → `jammy`，源路径：`ubuntu2204`
- Ubuntu 24.04 → `noble`，源路径：`ubuntu2404`

---

## 四、安装步骤概览

1. 安装/确认 NVIDIA 驱动（已满足可跳过）
2. 添加 NVIDIA CUDA 官方 apt 源并安装 CUDA Toolkit + cuDNN
3. 通过同一源或 Tar 安装 TensorRT
4. 安装 Dockerfile 中的通用 apt 依赖
5. （仅 TRT10 方案）安装 libassimp、CV-CUDA

---

## 五、安装 NVIDIA 驱动（未安装时）

若 `nvidia-smi` 已正常显示，可跳过。

```bash
# 添加 NVIDIA 驱动 PPA（Ubuntu 22.04/24.04 常用）
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo ubuntu-drivers devices
sudo apt install -y nvidia-driver-550   # 或推荐版本，按 ubuntu-drivers 建议
# 安装后重启
sudo reboot
```

---

## 六、添加 NVIDIA CUDA 源并安装 CUDA + cuDNN

以下 `$UBUNTU_REPO` 替换为：`ubuntu2004` / `ubuntu2204` / `ubuntu2404` 之一。

### 6.1 添加 GPG 与源

```bash
# 以 Ubuntu 22.04 为例；24.04 改为 ubuntu2404，20.04 改为 ubuntu2004
export UBUNTU_REPO=ubuntu2204

wget https://developer.download.nvidia.com/compute/cuda/repos/$UBUNTU_REPO/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
rm -f cuda-keyring_1.1-1_all.deb

# 若系统仍使用 apt-key（旧方式）：
# sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$UBUNTU_REPO/x86_64/3bf863cc.pub
# sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/$UBUNTU_REPO/x86_64/ /"

sudo apt update
```

### 6.2 安装 CUDA Toolkit（按方案二选一）

**方案 A：TRT10 / CUDA 12（推荐 Ubuntu 22.04/24.04）**

```bash
# CUDA 12.3 或 12.6 等（与 Dockerfile 12.3.2 接近即可）
sudo apt install -y cuda-toolkit-12-6
# 或指定 12.3：sudo apt install -y cuda-toolkit-12-3
```

**方案 B：TRT8 / CUDA 11**

```bash
sudo apt install -y cuda-toolkit-11-8
# 或 11.7：sudo apt install -y cuda-toolkit-11-7
```

### 6.3 安装 cuDNN（与 CUDA 版本匹配）

**CUDA 12 + cuDNN 9：**

```bash
sudo apt install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12
sudo ldconfig
ldconfig -p | grep cudnn
```

**CUDA 11 + cuDNN 8：**

```bash
sudo apt install -y libcudnn8 libcudnn8-dev
```

### 6.4 配置环境变量

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

若 CUDA 通过 apt 装到 `/usr` 而非 `/usr/local/cuda`，可先创建符号链接或改用实际路径：

```bash
# 仅当 /usr/local/cuda 不存在且希望统一路径时
sudo ln -s /usr/local/cuda-12.6 /usr/local/cuda   # 版本号按实际安装
```

验证：

```bash
nvcc --version
ldconfig -p | grep cudnn
```

---

## 七、安装 TensorRT

Dockerfile 中 TensorRT 来自 **同一 CUDA 源**（即 `developer.download.nvidia.com/compute/cuda/repos/...`）。  
若你的 `UBUNTU_REPO` 中有对应 TensorRT 包，可直接 apt 安装。

### 7.1 通过 apt（源内已有 TensorRT 时）

**TensorRT 10（与 TRT10 Dockerfile 一致）：**

```bash
# 版本号按源内实际包名微调，例如 10.7.0.23-1+cuda12.6
TENSORRT_PKG=10.7.0.23-1+cuda12.6
sudo apt install -y \
  libnvinfer10=${TENSORRT_PKG} \
  libnvinfer-plugin10=${TENSORRT_PKG} \
  libnvinfer-dev=${TENSORRT_PKG} \
  libnvinfer-headers-dev=${TENSORRT_PKG} \
  libnvinfer-headers-plugin-dev=${TENSORRT_PKG} \
  libnvinfer-plugin-dev=${TENSORRT_PKG} \
  libnvonnxparsers10=${TENSORRT_PKG} \
  libnvonnxparsers-dev=${TENSORRT_PKG} \
  libnvinfer-lean10=${TENSORRT_PKG} \
  libnvinfer-lean-dev=${TENSORRT_PKG} \
  libnvinfer-dispatch10=${TENSORRT_PKG} \
  libnvinfer-dispatch-dev=${TENSORRT_PKG} \
  libnvinfer-vc-plugin10=${TENSORRT_PKG} \
  libnvinfer-vc-plugin-dev=${TENSORRT_PKG} \
  libnvinfer-samples=${TENSORRT_PKG}
```

**TensorRT 8（与 TRT8 Dockerfile 一致）：**

```bash
TENSORRT_PKG=8.6.1.6-1+cuda11.8
sudo apt install -y \
  libnvinfer8=${TENSORRT_PKG} \
  libnvinfer-plugin8=${TENSORRT_PKG} \
  libnvinfer-dev=${TENSORRT_PKG} \
  libnvinfer-headers-dev=${TENSORRT_PKG} \
  libnvinfer-headers-plugin-dev=${TENSORRT_PKG} \
  libnvinfer-plugin-dev=${TENSORRT_PKG} \
  libnvonnxparsers8=${TENSORRT_PKG} \
  libnvonnxparsers-dev=${TENSORRT_PKG} \
  libnvparsers8=${TENSORRT_PKG} \
  libnvparsers-dev=${TENSORRT_PKG} \
  libnvinfer-lean8=${TENSORRT_PKG} \
  libnvinfer-lean-dev=${TENSORRT_PKG} \
  libnvinfer-dispatch8=${TENSORRT_PKG} \
  libnvinfer-dispatch-dev=${TENSORRT_PKG} \
  libnvinfer-vc-plugin8=${TENSORRT_PKG} \
  libnvinfer-vc-plugin-dev=${TENSORRT_PKG} \
  libnvinfer-samples=${TENSORRT_PKG}
```

若包名或版本在 `ubuntu2404` 中不同，可先查询：

```bash
apt-cache search libnvinfer
apt-cache policy libnvinfer-dev
```

### 7.2 使用 Tar 包安装（apt 源无对应 TRT 时）

从 [NVIDIA TensorRT 下载页](https://developer.nvidia.com/tensorrt-download) 下载与 CUDA 版本匹配的 Tar 包，解压后设置 `TENSORRT_ROOT` 并加入 `LD_LIBRARY_PATH` 和 `CMAKE_PREFIX_PATH`，编译时指定 `-DTENSORRT_ROOT=...`。

---

## 八、安装 Dockerfile 中的通用依赖

与三个 Dockerfile 中 `apt install` 列表一致：

```bash
sudo apt update
sudo apt install -y \
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
  cmake \
  libopencv-dev \
  libeigen3-dev \
  libgoogle-glog-dev \
  libgtest-dev
```

---

## 九、仅 TRT10 方案：assimp + CV-CUDA

与 `nvidia_gpu_tensorrt_trt10_u2204.dockerfile` 一致。

```bash
sudo apt install -y libassimp-dev
```

CV-CUDA（0.12.0-beta，CUDA 12）：

```bash
cd /tmp
wget https://github.com/CVCUDA/CV-CUDA/releases/download/v0.12.0-beta/cvcuda-lib-0.12.0_beta-cuda12-x86_64-linux.deb
wget https://github.com/CVCUDA/CV-CUDA/releases/download/v0.12.0-beta/cvcuda-dev-0.12.0_beta-cuda12-x86_64-linux.deb
sudo dpkg -i cvcuda-lib-0.12.0_beta-cuda12-x86_64-linux.deb
sudo dpkg -i cvcuda-dev-0.12.0_beta-cuda12-x86_64-linux.deb
rm -f cvcuda-lib-0.12.0_beta-cuda12-x86_64-linux.deb cvcuda-dev-0.12.0_beta-cuda12-x86_64-linux.deb
```

若 GitHub 较慢，Dockerfile 中使用的国内镜像为：  
`https://gp.zz990099.cn/https://github.com/...`（同上文件名）。

---

## 十、验证

```bash
# 驱动与 GPU
nvidia-smi

# CUDA
nvcc --version

# cuDNN（应有 libcudnn）
ldconfig -p | grep cudnn

# TensorRT（示例：检查头文件与库）
ls /usr/include/x86_64-linux-gnu/NvInfer*.h 2>/dev/null || ls /usr/include/NvInfer*.h 2>/dev/null
pkg-config --modversion nvinfer 2>/dev/null || true
```

编译 FoundationPose 或 easy_deploy_tool 时，确保 CMake 能找到 CUDA、cuDNN、TensorRT（若 TRT 用 Tar 安装，需设置 `TENSORRT_ROOT`）。

---

## 十一、针对你当前机器（Ubuntu 24.04 + RTX 5060）的推荐

- **系统**：Ubuntu 24.04 (noble)，`UBUNTU_REPO=ubuntu2404`。
- **驱动**：已安装 590.48.01，支持 CUDA 13.1，无需重装。
- **建议方案**：**CUDA 12.x + cuDNN 9 + TensorRT 10**（与 TRT10 Dockerfile 对齐）。
- **步骤顺序**：  
  第六节添加 `ubuntu2404` 源 → 安装 `cuda-toolkit-12-6`（或 12.3）→ `libcudnn9`/`libcudnn9-dev` → 第七节安装 TensorRT 10（若 ubuntu2404 源有对应包则用 apt，否则用 Tar）→ 第八节通用依赖 → 第九节 assimp + CV-CUDA。

按上述顺序执行即可在无 Docker 环境下复现 Dockerfile 的 CUDA、cuDNN、TensorRT 及依赖。
