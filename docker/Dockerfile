# TensorFold on NGC PyTorch for GB10 (sm_121, aarch64).
# The CUDA kernels JIT-compile at first start (build.py asks the GPU for its arch), not here;
# run.sh mounts a host cache so only the first boot pays for it.
ARG BASE=nvcr.io/nvidia/pytorch:26.07-py3@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c
FROM ${BASE}
ARG TF_REPO=https://github.com/ashhart/TensorFold.git
ARG TF_SHA=56e2e3ec55bc0ae1d7d5158c4fa2c79a3567ab21
# Optional patch on top of TF_SHA, a file under docker/patches/ (`none` = upstream as tagged).
# patches/pr141-on-0.6.2.patch: upstream PR #141 (--parallel N on two ranks) ported onto 0.6.2.
ARG TF_PATCH=none
ARG TF_PATCH_SHA=none
# response_format / guided_* need xgrammar on BOTH ranks: rank 1 recompiles each grammar, and a missing
# xgrammar there raises out of its follow loop and leaves rank 0 hanging in NCCL.
ARG XGRAMMAR=xgrammar==0.2.8
COPY patches/ /opt/tensorfold-patches/
# torch and triton come from the base image; fail the build if pip replaces them.
RUN pip freeze | grep -iE '^(torch|triton) @ ' > /opt/base-pins.txt \
 && git clone --quiet --filter=blob:none "${TF_REPO}" /opt/tensorfold-src \
 && git -C /opt/tensorfold-src checkout --quiet "${TF_SHA}" \
 && if [ "${TF_PATCH}" != none ]; then \
      echo "${TF_PATCH_SHA}  /opt/tensorfold-patches/${TF_PATCH#patches/}" | sha256sum -c - \
      && git -C /opt/tensorfold-src apply --index "/opt/tensorfold-patches/${TF_PATCH#patches/}"; fi \
 && env -u HF_TOKEN pip install --no-cache-dir /opt/tensorfold-src "${XGRAMMAR}" \
 && pip freeze | grep -iE '^(torch|triton) @ ' | diff - /opt/base-pins.txt \
 && python -c "import sys, tensorfold; assert sys.version_info >= (3, 11); print('tensorfold', tensorfold.__version__)" \
 && python -c "import xgrammar, transformers; from transformers import PreTrainedTokenizerFast" \
 && command -v ninja nvcc && test -e /usr/lib/aarch64-linux-gnu/libnccl.so.2 \
 && { echo "${TF_SHA}"; echo "${TF_PATCH} ${TF_PATCH_SHA}"; } > /opt/tensorfold-sha
LABEL tensorfold.sha=${TF_SHA} tensorfold.repo=${TF_REPO} tensorfold.patch=${TF_PATCH} tensorfold.patch_sha=${TF_PATCH_SHA}
ENV PYTHONUNBUFFERED=1 TENSORFOLD_NO_UPDATE_CHECK=1 HF_HUB_OFFLINE=1 HF_HOME=/cache/huggingface
