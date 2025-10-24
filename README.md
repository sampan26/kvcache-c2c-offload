# kvcache-c2c-offload

## Setup
- docker run --rm -it --ipc host --gpus all --ulimit memlock=-1 --ulimit stack=67108864 -p 8000:8000 nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc1
- git clone https://github.com/sampan26/kvcache-c2c-offload.git
