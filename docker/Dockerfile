ARG CONDADIR=/opt/deeprteenv

FROM mambaorg/micromamba:latest AS conda
ARG CONDADIR

SHELL ["/bin/bash", "-c"]
USER root

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN CONDA_OVERRIDE_CUDA="11.2" micromamba create -p $CONDADIR -c conda-forge \
    python=3.10 \
    jax \
    nvidia::cuda-nvcc \
    dm-haiku \
    optax \
    ml-collections \
    pip \
    && micromamba clean -ya

RUN $CONDADIR/bin/pip install --upgrade --no-cache-dir pip \
    && $CONDADIR/bin/pip install --no-cache-dir \
    tensorflow-cpu \
    tensorflow-datasets \
    git+https://github.com/mazhengcn/jaxline 

FROM ubuntu:latest
ARG CONDADIR

COPY --from=conda $CONDADIR $CONDADIR

ENV PATH=$CONDADIR/bin:$PATH \
    LC_ALL=C.UTF-8