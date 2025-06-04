import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
from dgp import get_dataloader
from check_gen import load_dataset, load_model


def plot_logits(path: str, seq: str, probs: bool = False):
    model, cfg = load_model(path)
    dataset = load_dataset(path)

    instr = torch.tensor(
        dataset.PCFG.tokenize_sentence(
            dataset.instruction_decorator.format(task_token="Task0", ops="<null>")
        )
    )

    sequence = torch.tensor(dataset.PCFG.tokenize_sentence(seq))
    sequence = torch.cat((instr, sequence, torch.tensor([dataset.PCFG.vocab["<eos>"]])))
    sequence = sequence.unsqueeze(0)  # Add batch dimension

    logits = model(sequence).squeeze(0)
    if args.probs:
        logits = torch.nn.functional.softmax(logits, dim=-1)

    logits = logits.detach().cpu().numpy()[8:, :14].transpose()
    # Cut off instruction prompt and special token logits

    plt.figure(figsize=(logits.shape[0] * 0.9, logits.shape[1] * 0.4))
    sns.heatmap(
        logits,
        annot=True,
        fmt="0.1f",
        linewidths=0.5,
        cmap="rocket",
        xticklabels=seq.split(),
        yticklabels=[
            "dig",
            "un",
            "bin",
            "tern",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
        ],
    )
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot logits for a sequence.")
    parser.add_argument(
        "--path",
        type=str,
        default="results/scratch/aqx6xdv8",
        help="Path to the model directory containing the latest checkpoint.",
    )
    parser.add_argument(
        "--seq",
        type=str,
        required=True,
        help="The sequence to plot logits for.",
    )
    parser.add_argument(
        "--probs",
        type=bool,
        default=False,
        help="Whether to plot probabilities instead of logits.",
    )
    args = parser.parse_args()

    plot_logits(args.path, args.seq, args.probs)
