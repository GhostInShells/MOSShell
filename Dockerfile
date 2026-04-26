# 使用 Ubuntu 22.04 作为基础
FROM ubuntu:22.04

# 1. 设置环境变量，防止安装交互式提示
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV PYTHONUNBUFFERED=1

# 2. 安装基础环境和必要的编译工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    curl \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. 建立 python 链接，确保 uv 能够识别
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# 4. 安装 uv (极速包管理)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# 5. 安装依赖层 (利用缓存，pyproject.toml 不变时不重新安装)
COPY pyproject.toml uv.lock ./
RUN uv pip install --system -r pyproject.toml

# 6. 复制项目源码 (这一层之后修改代码才会触发重新构建)
COPY src/ ./src/
COPY tests/ ./tests/
COPY README.md ./

# 7. 启动时默认为 bash，方便调试，或者你可以改成 CMD ["python", "-m", "pytest"]
CMD ["/bin/bash"]