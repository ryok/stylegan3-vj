FROM nvcr.io/nvidia/pytorch:21.08-py3

ENV DEBIAN_FRONTEND noninteractive
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt update
RUN apt-get install -y libgl1-mesa-dev
RUN apt-get install -y libsndfile1
RUN apt-get install -y ffmpeg

WORKDIR /

RUN git clone https://github.com/NVlabs/stylegan3.git

WORKDIR /workspace

COPY pyproject.toml .

RUN apt-get update \
    && apt-get install --no-install-recommends -y curl git build-essential \
    && pip install poetry \ 
    && poetry config virtualenvs.create false \
    && poetry install \
    && rm pyproject.toml

RUN pip install opencv-python