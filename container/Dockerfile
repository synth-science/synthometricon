# syntax=docker/dockerfile:1
# CUDA llama-server + Python pipeline. Used by the devcontainer and, via
# scripts/build_sif.sh, as the cluster .sif. See docs/cluster.md.

# Stage 1: build llama-server with CUDA.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS llama-build

# Commit, tag or branch. 1acee6b is the build that produced the corpus (`llama-server --version`
# in the cluster image printed "version: 1 (1acee6b)").
ARG LLAMA_CPP_REF=1acee6b
# V100 = SM 70; e.g. "70;75;80" for several.
ARG CUDA_ARCHITECTURES=70

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake build-essential ninja-build \
        libcurl4-openssl-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
# Blob-less clone + checkout: works for commit hashes (which `--branch` rejects) as well as tags/branches.
RUN git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git \
    && git -C llama.cpp checkout --quiet "${LLAMA_CPP_REF}" \
    && echo "llama.cpp at $(git -C llama.cpp rev-parse HEAD)"

WORKDIR /opt/llama.cpp
# Link against the CUDA driver stub (libcuda.so.1 symlink); the real driver is
# injected at runtime by --nv / --gpus.
ENV CUDA_STUBS=/usr/local/cuda/lib64/stubs
RUN ln -sf "${CUDA_STUBS}/libcuda.so" "${CUDA_STUBS}/libcuda.so.1" \
    && cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
        `# No -march=native: clara CPUs lack AVX512 (SIGILL). See docs/cluster.md.` \
        -DGGML_NATIVE=OFF \
        -DGGML_AVX2=ON \
        -DGGML_FMA=ON \
        -DGGML_AVX512=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DCMAKE_EXE_LINKER_FLAGS="-L${CUDA_STUBS} -Wl,-rpath-link,${CUDA_STUBS}" \
        -DCMAKE_SHARED_LINKER_FLAGS="-L${CUDA_STUBS} -Wl,-rpath-link,${CUDA_STUBS}" \
    && cmake --build build --config Release --target llama-server -j \
    && mkdir -p /opt/llama-install/bin /opt/llama-install/lib \
    && cp build/bin/llama-server /opt/llama-install/bin/ \
    && cp -a build/bin/*.so* /opt/llama-install/lib/
# Only llama-server is built, so `cmake --install` fails; copy files manually.

# Stage 2: runtime.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=2.1.3 \
    POETRY_VIRTUALENVS_CREATE=false \
    PIP_NO_CACHE_DIR=1

# Python 3.12 (deadsnakes) + curl runtime for llama-server's `-hf`/health.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        libcurl4-openssl-dev \
        libgomp1 \

    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && python3.12 -m pip install "poetry==${POETRY_VERSION}" \
    && rm -rf /var/lib/apt/lists/*

COPY --from=llama-build /opt/llama-install/ /usr/local/
RUN ldconfig

WORKDIR /app

# Deps first for layer caching; package-mode is off, so --no-root.
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --no-interaction --no-ansi

# On the cluster the live repo is bind-mounted over /app.
COPY . .

# Don't execute it here: the CUDA driver exists only at runtime.
RUN command -v llama-server

CMD ["bash"]
