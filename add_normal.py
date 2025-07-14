import tqdm
import wandb
import torch as t
from torch import nn
import pandas as pd
import numpy as np

from models import GPT2, ModelConfig, TrainingConfig
from utils import *

@t.inference_mode()
def benchmark_addition_normal(model: GPT2, dataset: pd.DataFrame):
    model.eval()
    q_toks = t.tensor(np.stack(dataset['question_toks']))
    ans_toks = t.tensor(dataset['answer_tok'].to_numpy())

    logits = model(q_toks)  # [seq_len, vocab]
    logprobs = t.log_softmax(logits[:, -1], dim=-1)
    
    pred_logprobs = logprobs[t.arange(ans_toks.shape[0]), ans_toks]
    mean_logprob = pred_logprobs.mean().item()
    pred_guesses = logprobs.argmax(dim=-1)
    accuracy = (pred_guesses == ans_toks).float().mean().item()
    return mean_logprob, accuracy

def train(model: GPT2, cfg: TrainingConfig, dataset: pd.DataFrame, testset: pd.DataFrame):
    opt = t.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay)

    input_max = dataset.attrs["input_max"]
    num_adds = dataset.attrs["num_adds"]

    batch_indices = t.arange(cfg.batch_size, requires_grad=False)

    wandb.init(project="recycler", name=f"normal_{input_max}x{num_adds}_ns", config=cfg)
    wandb.config.update(cfg.to_dict())

    acc = 0.0

    for b in (tr:=tqdm.trange(0, len(dataset) - len(dataset)%cfg.batch_size, cfg.batch_size, ncols=100)):
        q_toks = t.tensor(np.stack(dataset.iloc[b:b+cfg.batch_size]['question_toks']))
        ans_toks = t.tensor(dataset.iloc[b:b+cfg.batch_size]['answer_tok'].to_numpy())

        logits = model.forward(q_toks).squeeze()
        logprobs = t.log_softmax(logits, dim=-1)

        pred_logprobs = logprobs[batch_indices, -1, ans_toks]
        loss = -pred_logprobs.mean()
        loss.backward()
        opt.step()
        opt.zero_grad()

        wandb.log({"loss": loss.detach().item()}, step=b)
        tr.set_description(f"{magenta}loss: {loss.detach().item():.3f}, test acc: {acc:.4f}")

        if b*cfg.batch_size % 64_000 == 0:
            _, acc = benchmark_addition_normal(model, testset)
            wandb.log({"test_acc": acc}, step=b)


INPUT_MAX = 100
NUM_EXAMPLES = 1_000_000
NUM_ADDS = 2

if __name__ == "__main__":
    t.set_default_device(t.device("cuda"))
    t.manual_seed(42)
    random.seed(42)

    model_cfg = ModelConfig(
        d_model=32,
        seq_len=3,
        d_mlp=256,
        d_head=16,
        n_heads=4,
        n_layers=2,
        d_vocab=INPUT_MAX
    )
    model = GPT2(model_cfg)
    
    training_cfg = TrainingConfig(
        batch_size=128,
        lr=6e-4,
        weight_decay=1e-9,
        adam_beta1=0.9,
        adam_beta2=0.95
    )

    trainset, testset = makeAdditionDataset(INPUT_MAX, NUM_ADDS, NUM_EXAMPLES, train_split=0.999)

    train(model, training_cfg, trainset, testset)