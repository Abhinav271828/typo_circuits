import torch
import os
import pickle as pkl
import argparse
import random
import time

from model import GPT
from dgp import get_dataloader


def load_dataloader(path):
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
    return dataloader


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


def print_samples(path, n, use_model, error):
    model, cfg = load_model(path)
    dataloader = load_dataloader(path)
    dataset = dataloader.dataset

    if use_model:
        inputs = dataset.template.repeat(n, 1)
        samples, _ = model.sample(
            inputs=inputs,
            max_new_tokens=cfg.data.max_sample_length - 10,
            retrieve_llhoods="tokens",
        )

    else:
        # Use the dataloader
        random.seed(int(time.time()))
        datapoints = random.sample(range(len(dataset)), n)
        data = [dataset[i] for i in datapoints]
        samples = [point[0] for point in data]
        indices = [point[2] for point in data]

    samples = [
        dataset.PCFG.detokenize_sentence(s).split("<eos>")[0][33:] for s in samples
    ]

    for sample, idx in zip(samples, indices):
        if error and idx != -1:
            print(" ".join(sample.split()[:idx]), end=" ")
            print("\033[91m" + sample.split()[idx] + "\033[0m", end=" ")
            print(" ".join(sample.split()[idx + 1 :]))
        else:
            print(sample)


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
    parser.add_argument(
        "--use_model",
        action="store_true",
        default=False,
        help="Use the model to generate samples",
    )
    parser.add_argument(
        "--error",
        action="store_true",
        default=False,
        help="Show where the typos are, if any (doesn't work with model)",
    )
    args = parser.parse_args()
    # Use the provided path and number of samples

    print_samples(
        "results/scratch/" + args.path, args.num_samples, args.use_model, args.error
    )
