#
# Start tensorrt-llm docker
#

# install tensorrt-llm
# cd /dd/jiuchen/src/tensorrt_llm/ && pip install -e .

#
# Model conversion to HF weights
#
: <<'COMMENT'
# 70B to hf
CUDA_VISIBLE_DEVICES=GPU-b930159a-a609-47c4-a508-a534e7da9016 \
python /usr/local/lib/python3.10/dist-packages/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /nvme0-mnt/model/llama/data/models/llama-2 \
    --model_size 70B \
    --output_dir /nvme0-mnt/model/llama/data/models/llama-2/70B-hf


# 7b to hf
CUDA_VISIBLE_DEVICES=GPU-b930159a-a609-47c4-a508-a534e7da9016 \
python /dd/fhu/github/transformers/src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /nvme0-mnt/model/llama/data/models/llama-2 \
    --model_size 7B \
    --output_dir /nvme0-mnt/model/llama/data/models/llama-2/7B-hf
COMMENT

: <<'COMMENT'
#
# Quantization
#
# QPTQ 4 bit
git clone https://github.com/qwopqwop200/GPTQ-for-LLaMa.git
cd GPTQ-for-LLaMa/
pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=GPU-b930159a-a609-47c4-a508-a534e7da9016 \
python3 llama.py /nvme0-mnt/model/llama/data/models/llama-2/7B-hf/ c4 \
    --wbits 4 --true-sequential --groupsize 128 \
    --save_safetensors /nvme0-mnt/model/llama/data/models/llama-2/7B-hf/llama-7b-4bit-gs128.safetensors


CUDA_VISIBLE_DEVICES=GPU-b930159a-a609-47c4-a508-a534e7da9016 \
python3 llama.py /nvme0-mnt/model/llama/data/models/llama-2/70B-hf/ c4 \
    --wbits 4 --true-sequential --groupsize 128 \
    --save_safetensors /nvme0-mnt/model/llama/data/models/llama-2/70B-hf/llama-70b-4bit-gs128.safetensors

cd GPTQ-for-LLaMa/
CUDA_VISIBLE_DEVICES=GPU-b930159a-a609-47c4-a508-a534e7da9016 \
python3 llama.py /nvme2-mnt/Llama-2-70b-instruct c4 \
    --wbits 4 --true-sequential --groupsize 128 \
    --save_safetensors /nvme2-mnt/Llama-2-70b-instruct/llama-70b-4bit-gs128.safetensors
COMMENT

: <<'COMMENT'
# Smooth Q to int 8
python3 hf_llama_convert.py \
          -i /nvme0-mnt/model/llama/data/models/llama-2/7B-hf/ \
          -o /nvme0-mnt/model/llama/data/models/llama-2/7B-hf/smoothq0.5/ \
          -sq 0.5 \
          --calibrate-kv-cache \
          -t fp16
COMMENT

#
# A100 benchmark
: <<'COMMENT'
#
# 7B model
# build trt engine
CUDA_VISIBLE_DEVICES=GPU-b930159a-a609-47c4-a508-a534e7da9016 \
python build.py --model_dir /nvme0-mnt/model/llama/data/models/llama-2/7B-hf/ \
                --quant_ckpt_path /nvme0-mnt/model/llama/data/models/llama-2/7B-hf/llama-7b-4bit-gs128.safetensors \
                --dtype float16 \
                --use_gpt_attention_plugin float16 \
                --use_gemm_plugin float16 \
                --use_weight_only \
                --weight_only_precision int4_gptq \
                --per_group \
                --enable_context_fmha \
                --remove_input_padding \
                --max_batch_size 45 \
                --max_input_len 2048 \
                --max_output_len 256 \
                --output_dir /nvme0-mnt/model/llama/data/models/llama-2/7B-hf/trt_engines/int4_GPTQ_batch45_r0.5/1-gpu/

#batch_size: 45, min: 5858.132, max: 10420.846, p50: 9740.724, p95: 10384.242, p99: 10420.846
#Throughput = 1222 tokens/s
tokenizer_dir=/nvme0-mnt/model/llama/data/models/llama-2/7B-hf/
engine_dir=/nvme0-mnt/model/llama/data/models/llama-2/7B-hf/trt_engines/int4_GPTQ_batch45_r0.5/1-gpu/
## fp16 model
#engine_dir=/dd/abo/abo-models1/team/adsbrain/llama-2/m/llama-2/2/7B/
CUDA_VISIBLE_DEVICES=GPU-b930159a-a609-47c4-a508-a534e7da9016 \
python3 run.py --max_output_len=256 \
               --tokenizer_dir $tokenizer_dir \
               --engine_dir $engine_dir \
                --input_text "" \
                --batch_size 45 \
                --num_samples 1024
COMMENT

#
# interactive mode
#
: <<'COMMENT'
CUDA_VISIBLE_DEVICES=GPU-b930159a-a609-47c4-a508-a534e7da9016 \
python3 run.py --max_output_len=256 \
               --tokenizer_dir /nvme0-mnt/model/llama/data/models/llama-2/7B-hf/ \
               --engine_dir=/nvme0-mnt/model/llama/data/models/llama-2/7B-hf/trt_engines/int4_GPTQ_batch45c/1-gpu/


COMMENT

#
# upstage/Llama-2-70b-instruct
#
#export CUDA_VISIBLE_DEVICES=GPU-b930159a-a609-47c4-a508-a534e7da9016,GPU-edd78cc7-6a37-5a0d-f2f8-89698bbb7b82
export CUDA_VISIBLE_DEVICES=GPU-b930159a-a609-47c4-a508-a534e7da9016
tokenizer_dir=/nvme2-mnt/Llama-2-70b-instruct
hf_model_dir=$tokenizer_dir
engine_dir=$hf_model_dir/trt_engines/int4_GPTQ_batch16_r0.5_tp1_pp1/
: <<'COMMENT'
#
# interactive mode, HF model
python3 llama_hf.py \
    --model_dir $hf_model_dir \
    --tokenizer_dir $hf_model_dir \
    --interactive
#
# HF speed
# with 2 A100-80GB, bsz 4
python3 llama_hf.py \
    --model_dir $hf_model_dir \
    --tokenizer_dir $hf_model_dir \
    --batch_size 4 \
    --num_samples 20
COMMENT

: <<'COMMENT'
# build trt engine
# RuntimeError: Sizes of tensors must match except in dimension 0. 
# Expected size 8192 but got size 1024 for tensor number 1 in the list.
quant_ckpt_path=/nvme2-mnt/Llama-2-70b-instruct/llama-70b-4bit-gs128.safetensors
python build.py --model_dir $hf_model_dir \
                --quant_ckpt_path $quant_ckpt_path \
                --dtype float16 \
                --use_gpt_attention_plugin float16 \
                --use_gemm_plugin float16 \
                --use_weight_only \
                --weight_only_precision int4_gptq \
                --per_group \
                --enable_context_fmha \
                --remove_input_padding \
                --max_batch_size 16 \
                --max_input_len 2048 \
                --max_output_len 256 \
                --output_dir $engine_dir \
                --world_size 1 \
                --tp_size 1 \
                --pp_size 1


# gptq 4 bit model benchmark
python3 run.py --max_output_len=256 \
               --tokenizer_dir $tokenizer_dir \
               --engine_dir $engine_dir \
                --input_text "" \
                --batch_size 16 \
                --num_samples 64
COMMENT


: <<'COMMENT'
#
# paged attention, very slow, 80 tokens/s
BSZ=45
# A100 paged kv, having OOM problem
CUDA_VISIBLE_DEVICES=GPU-b930159a-a609-47c4-a508-a534e7da9016 \
python build.py --model_dir /nvme0-mnt/model/llama/data/models/llama-2/7B-hf/ \
                --quant_ckpt_path /nvme0-mnt/model/llama/data/models/llama-2/7B-hf/llama-7b-4bit-gs128.safetensors \
                --dtype float16 \
                --use_gpt_attention_plugin float16 \
                --use_gemm_plugin float16 \
                --use_weight_only \
                --weight_only_precision int4 \
                --per_group \
                --use_rmsnorm_plugin float16 \
                --enable_context_fmha \
                --remove_input_padding \
                --max_batch_size $BSZ \
                --max_input_len 2048 \
                --max_output_len 256 \
                --output_dir /nvme0-mnt/model/llama/data/models/llama-2/7B-hf/trt_engines/int4_GPTQ_batch${BSZ}_paged/1-gpu/ \
                --paged_kv_cache



CUDA_VISIBLE_DEVICES=GPU-b930159a-a609-47c4-a508-a534e7da9016 \
python3 run_ads.py --max_output_len=256 \
               --tokenizer_dir /nvme0-mnt/model/llama/data/models/llama-2/7B-hf/ \
               --engine_dir=/nvme0-mnt/model/llama/data/models/llama-2/7B-hf/trt_engines/int4_GPTQ_batch${BSZ}_paged/1-gpu/ \
               --batch_size $BSZ \
               --num_samples 256

COMMENT
