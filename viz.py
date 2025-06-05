import matplotlib.pyplot as plt
import argparse
import seaborn as sns
import torch
import os
from dgp import get_dataloader
from check_gen import load_dataloader, load_model


def plot_logits(path: str, sfile: str, probs: bool = False):
    path = "results/scratch/" + path
    model, cfg = load_model(path)
    dataset = load_dataloader(path).dataset

    instr = torch.tensor(
        dataset.PCFG.tokenize_sentence(
            dataset.instruction_decorator.format(task_token="Task0", ops="<null>")
        )
    )

    with open(sfile, "r") as f:
        for line in f:
            seq = line.strip()
            if not seq:
                return

            sequence = torch.tensor(dataset.PCFG.tokenize_sentence(seq))
            sequence = torch.cat(
                (instr, sequence, torch.tensor([dataset.PCFG.vocab["<eos>"]]))
            )
            sequence = sequence.unsqueeze(0)  # Add batch dimension

            logits = model(sequence).squeeze(0)  # [S, V]
            if args.probs:
                logits = torch.nn.functional.softmax(logits, dim=-1)

            logits = torch.cat(
                [
                    logits[:, : len(dataset.PCFG.vocab) - 9],
                    logits[:, dataset.PCFG.vocab["<eos>"]].unsqueeze(1),
                ],
                dim=1,
            )
            logits = logits.detach().cpu().numpy()[7:-1, :].transpose()
            # Cut off instruction prompt and special token logits

            plt.figure(figsize=(logits.shape[0] * 0.9, 8))
            sns.heatmap(
                logits,
                annot=True,
                fmt="0.1f",
                linewidths=0.5,
                cmap="rocket",
                xticklabels=dataset.PCFG.detokenize_sentence(sequence[0, 7:-1]).split(),
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
                    "<eos>",
                ],
            )
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot logits for a sequence.")
    parser.add_argument(
        "--path",
        type=str,
        default="",
        help="Path to the model directory containing the latest checkpoint.",
    )
    parser.add_argument(
        "--sfile",
        type=str,
        required=True,
        help="The file with a list of sequences to plot logits for.",
    )
    parser.add_argument(
        "--probs",
        action="store_true",
        default=False,
        help="Whether to plot probabilities instead of logits.",
    )
    args = parser.parse_args()

    plot_logits(args.path, args.sfile, args.probs)
