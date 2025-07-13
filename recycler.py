import eindex
import wandb
import tqdm
import datasets

from utils import *
from models import Recycler, TrainingConfig, RecycleModelConfig

def completion(model: Recycler, prompt: str, max_length: int = 50) -> str:
    tokens = model.tokenizer(prompt, return_tensors='pt').input_ids.squeeze()
    out = tokens.tolist()
    with t.no_grad():
        ctx = model.process_seq(tokens[:-1])  # Get initial context for the prompt
        next_token = tokens[-1].unsqueeze(0)  # The last token of the prompt
        for _ in range(max_length):
            new_ctx, logits = model.forward(next_token, ctx) # Process the next token with the current context
            probs = F.softmax(logits, dim=-1)
            next_token  = t.multinomial(probs, num_samples=1)
            out.append(next_token.item())
            ctx = t.cat((ctx, new_ctx.unsqueeze(1)), dim=1)  # Append the new context
    return model.tokenizer.decode(out)

def train(model: Recycler, cfg: TrainingConfig, dataset: datasets.Dataset):
    optimizer = t.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay)

    model.train()

    wandb.init(project="recycler", name="recycler", config=cfg)
    wandb.config.update(cfg.to_dict())

    sample_completion = completion(model, "George Washington was")
    print(yellow, sample_completion, endc)
    table_data = [[sample_completion]]
    table = wandb.Table(data=table_data, columns=['completion'])
    wandb.log({"sample_completion": table})

    seq_len = model.cfg.seq_len
    batch_size = cfg.batch_size
    d_model = model.cfg.d_model
    d_vocab = model.cfg.d_vocab

    dl = t.utils.data.DataLoader(dataset, batch_size=cfg.batch_size)
    for i, batch in enumerate((tr:=tqdm.tqdm(dl, ncols=100))):
        tokens = batch['input_ids']
        print()
        print(red, tokens.shape, endc)

        ctx = t.zeros((batch_size, seq_len, d_model)) # preaallocate context instead of cating
        logits = t.zeros((batch_size, seq_len, d_vocab)) # preaallocate context instead of cating
        new_ctx, new_logits = model.forward(tokens[:, 0].reshape(-1, 1)) # get initial context by feeding  first token
        ctx[:, 0, :] = new_ctx
        logits[:, 0, :] = new_logits
        for s in range(1, model.cfg.seq_len):
            toks = tokens[:, s].reshape(-1, 1) # (batch, 1)
            new_ctx, new_logits = model.forward(toks, ctx[:, :s]) # process the next token with the current context
            ctx[:, s, :] = new_ctx # update the context with the new context vector
            logits[:, s-1, :] = new_logits
       
        logprobs = t.log_softmax(logits, dim=-1)
        loss = -eindex.eindex(logprobs[:, :-1], tokens[:, 1:], "batch seq [batch seq]").sum()

        loss.backward()
        optimizer.step()
        
        exit()

        wandb.log({"loss": loss.item()})
        tr.set_description(f"{magenta}loss: {loss.item():.3f}")

        if i%1_000 == 0:
            sample_completion = completion(model, "George Washington was")
            print(yellow, sample_completion, endc)
            table_data.append([sample_completion])
            table = wandb.Table(data=table_data, columns=['completion'])
            wandb.log({"sample_completion": table})

            t.save(model.state_dict(), f"saves/normal{i}.pth")

if __name__ == "__main__":
    t.set_default_device(t.device("cuda"))
    t.manual_seed(42)
    random.seed(42)

    #model_cfg = RecycleModelConfig(d_model=32, seq_len=128, d_mlp=128, d_head=16, n_heads=2, n_layers=4, recycle_layer_pos=3, d_vocab=50257)
    model_cfg = RecycleModelConfig(d_model=512, seq_len=256, d_mlp=2048, d_head=64, n_heads=8, n_layers=4, d_vocab=50257)
    model = Recycler(model_cfg)
    training_cfg = TrainingConfig(
        batch_size=64,
        lr=1e-3,
        weight_decay=1e-6,
        adam_beta1=0.9,
        adam_beta2=0.95
    )

    #dataset = tokenizeAndSaveDataset(model.tokenizer, model_cfg, "HuggingFaceFW/fineweb-edu", "sample-10BT", f"fineweb-edu-tokenized-512", 0.07, pad=False)
    dataset = loadTokenizedDataset("fineweb-edu-tokenized-128")
    
    train(model, training_cfg, dataset)