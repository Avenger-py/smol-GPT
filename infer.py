import tiktoken
from tqdm import tqdm
import torch
import torch.nn.functional as F
import argparse
from model import GPT
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Infer parameters")
    parser.add_argument("--model_path", default="checkpoints/ckpt_v4_ep10000.pt", help="Path to the model to infer with")
    parser.add_argument("--prompt", type=str, default="The Amazon rainforest is", help="Input text to the model")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use 'cuda' or 'cpu'")
    parser.add_argument("--stream_text", action="store_true", help="Stream text")

    args = parser.parse_args()

    return args

@torch.no_grad()
def generate_text(text, model, max_new_tokens=100, device="cuda", stream_text=True):
    if stream_text:
        sys.stdout.write(text)
        sys.stdout.flush()
    
    encd = tiktoken.get_encoding("gpt2")
    enc = encd.encode_ordinary(text)
    input_tokens = enc
    output_tokens = []

    for i in range(max_new_tokens):
        input_tokens = input_tokens if len(input_tokens)<=config.context_length else input_tokens[- config.context_length:]
        with torch.no_grad():
            model.eval()
        logits, _ = model(torch.tensor(input_tokens, dtype=torch.int64, device=device).view(1, -1))
        
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        out_token = torch.multinomial(probs, num_samples=1)
        
        if stream_text:
            sys.stdout.write(encd.decode([out_token.item()]))
            sys.stdout.flush()
        
        input_tokens.append(out_token.item())
        output_tokens.append(out_token.item())

    if not stream_text:
        print(encd.decode(enc+output_tokens))


args = parse_args()

checkpoint = torch.load(args.model_path, map_location=args.device)

config = checkpoint['config']
# print(config)
config.device = args.device
model = GPT(config).to(args.device)
model.load_state_dict(checkpoint['model'])
model.eval()

print(f"Model weights loaded from path: {args.model_path}")

generate_text(text=args.prompt, 
              model=model, 
              max_new_tokens=args.max_new_tokens, 
              device=args.device, 
              stream_text=args.stream_text)
