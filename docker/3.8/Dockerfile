# bring in the micromamba image so we can copy files from it
FROM mambaorg/micromamba:1.4.1 as micromamba

# the actual base image
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04
# FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel
ENV NVIDIA_DRIVER_CAPABILITIES all

# Install os-level packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    libxcursor-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxi-dev \
    mesa-common-dev \
    zip \
    unzip \
    wget \
    tmux \
    curl \
    make \
    mesa-vulkan-drivers \
    pigz \
    git \
    libegl1 \
    git-lfs
# gcc-8 \
# g++-8 \
# vulkan-utils \

# COPY nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
# COPY nvidia_layers.json /etc/vulkan/implicit_layer.d/nvidia_layers.json

USER root

# if your image defaults to a non-root user, then you may want to make the
# next 3 ARG commands match the values in your image. You can get the values
# by running: docker run --rm -it my/image id -a
ARG MAMBA_USER=mamba
ARG MAMBA_USER_ID=1001
ARG MAMBA_USER_GID=1001
ENV MAMBA_USER=$MAMBA_USER
ENV MAMBA_ROOT_PREFIX="/opt/conda"
ENV MAMBA_EXE="/bin/micromamba"

COPY --from=micromamba "$MAMBA_EXE" "$MAMBA_EXE"
COPY --from=micromamba /usr/local/bin/_activate_current_env.sh /usr/local/bin/_activate_current_env.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_shell.sh /usr/local/bin/_dockerfile_shell.sh
COPY --from=micromamba /usr/local/bin/_entrypoint.sh /usr/local/bin/_entrypoint.sh
COPY --from=micromamba /usr/local/bin/_activate_current_env.sh /usr/local/bin/_activate_current_env.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_initialize_user_accounts.sh /usr/local/bin/_dockerfile_initialize_user_accounts.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_setup_root_prefix.sh /usr/local/bin/_dockerfile_setup_root_prefix.sh

RUN /usr/local/bin/_dockerfile_initialize_user_accounts.sh && \
    /usr/local/bin/_dockerfile_setup_root_prefix.sh


USER $MAMBA_USER

SHELL ["/usr/local/bin/_dockerfile_shell.sh"]

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]

# You can modify the CMD statement as needed....
CMD ["/bin/bash"]

# Optional: you can now populate a conda environment:
RUN micromamba install --yes --name base --channel conda-forge \
    jq python=3.8 setuptools= && \
    micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1  # (otherwise python will not be found)

USER root

RUN mkdir /work
WORKDIR /work

ARG CACHE_DATE=not_a_date

RUN pip3 install --no-cache-dir torch
# torchvision torchaudio
RUN wget https://developer.nvidia.com/isaac-gym-preview-4 -O isaac.tar.gz \
    && tar -xzf isaac.tar.gz -C . \
    && pip install --no-cache-dir -e isaacgym/python
RUN pip --no-cache-dir install \
    "omegaconf" \
    "dacite" \
    "wandb" \
    "tensorboard" \
    "tensorboardX" \
    "neptune" \
    "shortuuid" \
    "tqdm" \
    "gymnasium" \
    "opencv-python" \
    "imageio[ffmpeg]" \
    "mujoco==3.2.3" \
    "psutil" \
    "scikit-learn" \
    "ipdb" \
    "jupyterlab" \
    "ipython" \
    "black" \
    "isort" \
    "pre-commit"

RUN apt-get install -y --no-install-recommends libglib2.0-0
