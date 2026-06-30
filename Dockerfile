FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    git git-lfs ffmpeg libsm6 libxext6 cmake rsync libgl1 curl bzip2 \
    && rm -rf /var/lib/apt/lists/*

RUN rm -rf /opt/conda && \
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
ENV PATH="/opt/conda/bin:$PATH"

RUN conda install -y --override-channels -c conda-forge pykep && conda clean -afy

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app
COPY . .

EXPOSE 7860

CMD ["python", "ui.py"]
