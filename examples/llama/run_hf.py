from transformers import LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, AutoTokenizer, TextStreamer
import time
import torch
import argparse
from run import print_benchmark

EOS_TOKEN = 2
PAD_TOKEN = 2

def parse_arguments():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--max_output_len', type=int, required=True)
    #parser.add_argument('--log_level', type=str, default='error')
    parser.add_argument('--model_dir', type=str, default='llama_outputs')
    parser.add_argument('--tokenizer_dir',
                        type=str,
                        default=".",
                        help="Directory containing the tokenizer.model.")
    parser.add_argument('--interactive', action='store_true', help='Whether to run in interactive mode.')
    #parser.add_argument('--input_text',
    #                    type=str,
    #                    default='Born in north-east France, Soyer trained as a')
    #parser.add_argument(
    #    '--input_tokens',
    #    dest='input_file',
    #    type=str,
    #    help=
    #    'CSV or Numpy file containing tokenized input. Alternative to text input.',
    #    default=None)
    #parser.add_argument('--output_csv',
    #                    type=str,
    #                    help='CSV file where the tokenized output is stored.',
    #                    default=None)
    #parser.add_argument('--output_npy',
    #                    type=str,
    #                    help='Numpy file where the tokenized output is stored.',
    #                    default=None)
    #parser.add_argument('--num_beams',
    #                    type=int,
    #                    help="Use beam search if num_beams >1",
    #                    default=1)
    parser.add_argument('--batch_size',
                        type=int,
                        default=1)
    parser.add_argument('--num_samples',
                        type=int,
                        default=128)
    return parser.parse_args()

def generate(
    model_dir: str,
    tokenizer_dir: str,
    interactive: bool,
    batch_size: int,
    num_samples: int):
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
  model = AutoModelForCausalLM.from_pretrained(
      model_dir,
      device_map="auto",
      torch_dtype=torch.float16,
      #load_in_8bit=True,
      #rope_scaling={"type": "dynamic", "factor": 2} # allows handling of longer inputs
  )

  #tokenizer = LlamaTokenizer.from_pretrained(tokenizer_dir, legacy=False)
  #model = LlamaForCausalLM.from_pretrained(model_dir).to('cuda')

  tokenizer.pad_token = PAD_TOKEN
  model.half()
  model.eval()
  print(f"Done model loading")
  
  if interactive:
    while True:
      prompt = input("\nInput an one line prompt below:\n")
      inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
      #del inputs["token_type_ids"]
      streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
      output = model.generate(**inputs, streamer=streamer, use_cache=True, max_new_tokens=float('inf'))
      output_text = tokenizer.decode(output[0], skip_special_tokens=True)
      print(output_text)

  else:
    ads_data = "/dd/fhu/github/tensorrt_llm/examples/llama/sample_data_0808_0830_10k.output.AdsLLM.train.prompt.tsv"
    with open(ads_data, "r") as infile:
        input_texts = infile.readlines()
    # adjust the number of samples to be a multiple of batch_size
    num_samples = num_samples // batch_size * batch_size
    input_texts = input_texts[:num_samples]
    latencies = []
    total_generated_token_num = 0
    for i in range(0, len(input_texts), batch_size):
      batch_texts = input_texts[i : i + batch_size]
      tokenized_inputs = tokenizer(
        batch_texts,
        return_tensors='pt',
        padding=True,
        ).to('cuda')
      print(f"Batch {1 + i // batch_size}/{num_samples // batch_size} - "
            f"Input shape: {tokenized_inputs['input_ids'].shape}")
      input_id_num = (tokenized_inputs['input_ids'] != PAD_TOKEN).sum().item()
      start = time.time()
      responses = model.generate(**tokenized_inputs, max_length=256 + tokenized_inputs['input_ids'].size(1))
      torch.cuda.synchronize()
      end = time.time()
      latencies.append((end - start) * 1000)
      print(f"\t --> Output shape: {responses.shape}, {latencies[-1]:.3f} ms")
      output_token_num = (responses != PAD_TOKEN).sum().item()
      total_generated_token_num += (output_token_num - input_id_num)
      responses = tokenizer.batch_decode(responses, skip_special_tokens = True)
    #print(responses)
    print_benchmark(latencies, batch_size)
    print(f"Throughput: {int(total_generated_token_num / sum(latencies) * 1000 + 0.5)} tokens/s")

if __name__ == "__main__":
    args = parse_arguments()
    generate(**vars(args))
