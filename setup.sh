# Install the packages in r1-v .
cd src/r1-v 
pip install -e ".[dev]"

python -m pip install --upgrade pip
pip install torch==2.1.0+cu121 torchvision==0.12.0+cu121 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121


# Addtional modules
pip install wandb==0.18.3
pip install tensorboardx
pip install qwen_vl_utils torchvision
#pip install flash-attn --no-build-isolation
pip install flash_attn==2.7.4.post1


# vLLM support 
#pip install vllm==0.7.2
pip3 install vllm==0.7.3

# fix transformers version
#pip install git+https://github.com/huggingface/transformers.git@336dc69d63d56f232a183a3e7f52790429b871ef
pip install git+https://github.com/huggingface/transformers.git@a40f1ac602fe900281722254c52ce3773f28eb0e
