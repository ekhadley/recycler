import tqdm
import wandb
import torch as t
from torch import nn
import pandas as pd
import numpy as np

from models import Recycler, RecycleModelConfig, TrainingConfig
from utils import *

@t.inference_mode()
def benchmark_addition_recycler(model: Recycler, dataset: pd.DataFrame):
    model.eval()
    q_toks = t.tensor(np.stack(dataset['question_toks']))
    ans_toks = t.tensor(dataset['answer_tok'].to_numpy())

    n_examples = q_toks.shape[0]
    q_len = dataset.attrs["num_adds"]
    d_model = model.cfg.d_model

    ctx = t.zeros((n_examples, q_len, d_model))
    for s in range(q_len):
        toks = q_toks[:, s].reshape(-1, 1)
        new_ctx, logits = model.forward(toks, ctx[:, :s] if s != 0 else None)
        ctx[:, s, :] = new_ctx

    logprobs = t.log_softmax(logits, dim=-1)
    
    pred_logprobs = logprobs[t.arange(ans_toks.shape[0]), ans_toks]
    mean_logprob = pred_logprobs.mean().item()
    pred_guesses = logprobs.argmax(dim=-1)
    accuracy = (pred_guesses == ans_toks).float().mean().item()
    return mean_logprob, accuracy

def train(model: Recycler, cfg: TrainingConfig, dataset: pd.DataFrame, testset: pd.DataFrame):
    opt = t.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay)

    input_max = dataset.attrs["input_max"]
    q_len = dataset.attrs["num_adds"]
    batch_size = cfg.batch_size
    d_model = model.cfg.d_model

    batch_indices = t.arange(cfg.batch_size, requires_grad=False)

    wandb.init(project="recycler", name=f"recycler_{input_max}x{q_len}_ns", config=cfg)
    wandb.config.update(cfg.to_dict())

    acc = 0.0

    for b in (tr:=tqdm.trange(0, len(dataset) - len(dataset)%cfg.batch_size, cfg.batch_size, ncols=100)):
        q_toks = t.tensor(np.stack(dataset.iloc[b:b+cfg.batch_size]['question_toks']))
        ans_toks = t.tensor(dataset.iloc[b:b+cfg.batch_size]['answer_tok'].to_numpy())

        ctx = t.zeros((batch_size, q_len, d_model)) # preaallocate context instead of cating
        for s in range(q_len):
            toks = q_toks[:, s].reshape(-1, 1) # (batch, 1)
            new_ctx, logits = model.forward(toks, ctx[:, :s] if s != 0 else None) # process the next token with the current context
            ctx[:, s, :] = new_ctx # update the context with the new context vector

        #print()
        #print(red, q_toks.shape, blue, ans_toks.shape, endc)
        #print(purple, ctx.shape, lime, new_ctx.shape, green, logits.shape, endc)
        logprobs = t.log_softmax(logits, dim=-1)
        loss = -logprobs[batch_indices, ans_toks].mean()

        loss.backward()
        opt.step()
        opt.zero_grad()

        wandb.log({"loss": loss.detach().item()}, step=b)
        tr.set_description(f"{magenta}loss: {loss.detach().item():.3f}, test acc: {acc:.4f}")

        if b*cfg.batch_size % 64_000 == 0:
            _, acc = benchmark_addition_recycler(model, testset)
            wandb.log({"test_acc": acc}, step=b)


INPUT_MAX = 100
NUM_EXAMPLES = 1_000_000
NUM_ADDS = 2

if __name__ == "__main__":
    t.set_default_device(t.device("cuda"))
    t.manual_seed(42)
    random.seed(42)

    model_cfg = RecycleModelConfig(
        d_model=32,
        seq_len=3,
        d_mlp=256,
        d_head=16,
        n_heads=4,
        n_layers=4,
        d_vocab=INPUT_MAX,
        recycle_layer=3
    )
    model = Recycler(model_cfg)
    
    training_cfg = TrainingConfig(
        batch_size=4,
        lr=6e-4,
        weight_decay=1e-9,
        adam_beta1=0.9,
        adam_beta2=0.95
    )

    trainset, testset = makeAdditionDataset(INPUT_MAX, NUM_ADDS, NUM_EXAMPLES, train_split=0.9)

    train(model, training_cfg, trainset, testset)