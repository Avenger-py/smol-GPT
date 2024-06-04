import torch
from tqdm import tqdm
import argparse
import pickle
from utils import load_embedded_data, generate_batches
from model import GPT
from sklearn.metrics import precision_score, recall_score, f1_score
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser(description="Training parameters")
    parser.add_argument("--model_path", default="checkpoints/ckpt_v4_ep10000.pt", help="Path to the model to infer with")
    parser.add_argument("--test_data_path", default="wikitext103/test_embd.bin", help="Path to the test dataset")
    parser.add_argument("--eval_epochs", type=int, default=100, help="Number of epochs to evaluate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use 'cuda' or 'cpu'")

    args = parser.parse_args()
    return args


def get_test_loss(model, test_data, eval_epochs):
    model.eval()
    test_losses = torch.zeros(eval_epochs)

    for iter in tqdm(range(eval_epochs)):
        X, Y = generate_batches(test_data, config)
        _, loss = model(X, Y)
        test_losses[iter] = loss.item()

    loss_mean = test_losses.mean()
    model.train()
    return loss_mean


def perplexity(loss):
    return torch.exp(loss)

def sample(logits):
    logits = logits.view(-1, logits.shape[-1])
    probs = F.softmax(logits, dim=-1)
    out_tokens = torch.multinomial(probs, num_samples=1)
    return out_tokens

def get_precision_recall_f1(model, test_data, eval_epochs):
    model.eval()
    precision_scores = []
    recall_scores = []
    f1_scores = []
    for iter in tqdm(range(eval_epochs)):
        X, Y = generate_batches(test_data, config)
        logits, _ = model(X, Y)
        logits = sample(logits).view(config.batch_size, -1)

        prcn = precision_score(Y.view(-1).detach().cpu().numpy(), logits.view(-1).detach().cpu().numpy(), average='weighted', zero_division=0)
        precision_scores.append(prcn)

        rcll = recall_score(Y.view(-1).detach().cpu().numpy(), logits.view(-1).detach().cpu().numpy(), average='weighted', zero_division=0)
        recall_scores.append(rcll)

        f1 = f1_score(Y.view(-1).detach().cpu().numpy(), logits.view(-1).detach().cpu().numpy(), average='weighted', zero_division=0)
        f1_scores.append(f1)
    
    avg_precision_score = torch.tensor(precision_scores).mean()
    avg_recall_score = torch.tensor(recall_scores).mean()
    avg_f1_score = torch.tensor(f1_scores).mean()
    return avg_precision_score, avg_recall_score, avg_f1_score


def get_accuracy(model, test_data, eval_epochs):
    model.eval()
    accuracy_scores = []
    for iter in tqdm(range(eval_epochs)):
        X, Y = generate_batches(test_data, config)
        logits, _ = model(X, Y)
        logits = sample(logits).view(config.batch_size, -1)
        acc = (logits.view(-1).detach().cpu().numpy() == Y.view(-1).detach().cpu().numpy()).sum()/len(Y)
        accuracy_scores.append(acc)
    
    avg_accuracy_score = torch.tensor(accuracy_scores).mean()
    return avg_accuracy_score


args = parse_args()

test_enc = load_embedded_data(args.test_data_path)
test_enc = torch.tensor(test_enc)

checkpoint = torch.load(args.model_path, map_location=args.device)
print(f"Model weights loaded from path: {args.model_path}")
config = checkpoint['config']

# If cuda out of memory, reduce batch size
# config.batch_size = 8

config.device = args.device
model = GPT(config).to(args.device)
model.load_state_dict(checkpoint['model'])
model.eval()

loss_mean = get_test_loss(model, test_enc, args.eval_epochs)
perplexity_score = perplexity(loss_mean)
precision_mean, recall_mean, f1_mean = get_precision_recall_f1(model, test_enc, args.eval_epochs)
accuracy = get_accuracy(model, test_enc, args.eval_epochs)

print(f"\nTest loss: {loss_mean:.4f}")
print(f"Test perplexity: {perplexity_score:.4f}")
print(f"Test precision: {precision_mean:.4f}")
print(f"Test recall: {recall_mean:.4f}")
print(f"Test f1: {f1_mean:.4f}")
print(f"Test accuracy: {accuracy:.4f}")