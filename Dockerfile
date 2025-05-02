FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-devel

# apt install 과정에서 생기는 사용자 상호작용 방지
ENV DEBIAN_FRONTEND=noninteractive

# 서드파티 apt 저장소 설정 모두 제거
RUN rm -f /etc/apt/sources.list.d/*.list

# 기본으로 설치할 프로그램들
RUN apt-get update && apt-get install -y \
    wget \
    vim \
    curl \
    ssh \
    tree \
    sudo \
    git \
    libglvnd-dev \
    libglib2.0-0 \
    python3 \
    python3-pip \
    python3-dev \
    zip && \
    apt-get clean

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "run.py"]