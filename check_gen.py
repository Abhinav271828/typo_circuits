import torch
import os
import pickle as pkl
import argparse

from model import GPT
from dgp import get_dataloader


def load_dataset(path):
    state_dict = torch.load(os.path.join(path, "latest_ckpt.pt"), map_location="cpu")
    cfg = state_dict["config"]
    dataloader = get_dataloader(
        language=cfg.data.language,
        config=cfg.data.config,
        alpha=cfg.data.alpha,
        prior_type=cfg.data.prior_type,
        num_iters=cfg.data.num_iters * cfg.data.batch_size,
        max_sample_length=cfg.data.max_sample_length,
        seed=cfg.seed,
        batch_size=cfg.data.batch_size,
        num_workers=0,
    )
    return dataloader.dataset


def load_model(path):
    # path = "results/scratch/uhpr1wa0"
    state_dict = torch.load(os.path.join(path, "latest_ckpt.pt"), map_location="cpu")
    cfg = state_dict["config"]

    with open(os.path.join(path, "grammar/PCFG.pkl"), "rb") as f:
        pcfg = pkl.load(f)
    model = GPT(cfg.model, pcfg.vocab_size)
    model.load_state_dict(state_dict["net"])
    model.eval()
    return model, cfg


def print_samples(path, n):
    model, cfg = load_model(path)
    dataset = load_dataset(path)

    inputs = dataset.template.repeat(n, 1)
    samples, _ = model.sample(
        inputs=inputs,
        max_new_tokens=cfg.data.max_sample_length - 10,
        retrieve_llhoods="tokens",
    )

    # Transfer to CPU and detokenize
    samples = samples.cpu().numpy()
    samples = [dataset.PCFG.detokenize_sentence(s).split("<eos>")[0] for s in samples]

    for sample in samples:
        print(sample[33:])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate samples from a model.")
    parser.add_argument(
        "--path",
        type=str,
        default="",
        help="Path to the model directory",
    )
    parser.add_argument(
        "--num_samples", type=int, default=10, help="Number of samples to generate"
    )
    args = parser.parse_args()
    # Use the provided path and number of samples

    print_samples("results/scratch/" + args.path, args.num_samples)
