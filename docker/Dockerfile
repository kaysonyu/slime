# Supply a CUDA/PyTorch/TransformerEngine image matching tts-runtime-versions.json.
# Omni and Megatron remain independently maintained source mounts.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
WORKDIR /workspace/slime
COPY . /workspace/slime
RUN python -m pip install -e .
ENV PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    CUDA_DEVICE_MAX_CONNECTIONS=1 \
    NCCL_CUMEM_ENABLE=0 \
    NCCL_NVLS_ENABLE=0
