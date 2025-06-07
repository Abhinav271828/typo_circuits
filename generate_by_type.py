import argparse
from dgp import get_dataloader
from omegaconf import OmegaConf
import random


def load_dataset_from_config_file(config):
    cfg = OmegaConf.load(config)
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


def generate_by_type(type, orig, dataset):
    """
    Generate samples from the dataset based on the specified error type.
    Args:
        type (int): The type of error to filter samples by.
            0: No error
            1: Error is clear
            2: Error has no effect
            3: Error is not clear
        dataset (Dataset): The dataset from which to generate samples.
    """
    sample = random.choice(dataset)
    sample = dataset.PCFG.detokenize_sentence(sample[0]).split("<eos>")[0][33:]
    sample = sample.split()
    if orig:
        print(" ".join(sample))
    match type:
        case 0:
            pass
        case 1:
            r = random.random()
            if r < 0.33:  # word errors
                indices = [
                    i
                    for i, word in enumerate(sample)
                    if word in ["dig", "un", "bin", "tern"]
                ]
                idx = random.choice(indices)
                sample[idx] = random.choice(
                    ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
                )
            else:  # operator index errors
                indices = [
                    i + 1
                    for i, word in enumerate(sample)
                    if word in ["un", "bin", "tern"]
                ]
                if indices != []:
                    idx = random.choice(indices)
                    sample[idx] = random.choice(
                        ["dig", "un", "bin", "tern", "4", "5", "6", "7", "8", "9"]
                    )
                if r < 0.66 or indices == []:  # digit errors
                    indices = [
                        i
                        for i, word in enumerate(sample)
                        if word in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
                        and i not in indices
                    ]
                    idx = random.choice(indices)
                    sample[idx] = random.choice(["dig", "un", "bin", "tern"])
            pass
        case 2:
            indices = [
                i
                for i, word in enumerate(sample)
                if word not in ["dig", "un", "bin", "tern"]
            ]
            idx = random.choice(indices)
            # Must be an operator index
            if sample[idx - 1] in [
                "un",
                "bin",
                "tern",
            ]:
                i = ["0", "1", "2"]
                i.remove(sample[idx])
                sample[idx] = random.choice(i)
            # Must be a digit
            else:
                i = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
                i.remove(sample[idx])
                sample[idx] = random.choice(i)
        case 3:
            indices = [
                i
                for i, word in enumerate(sample)
                if word in ["dig", "un", "bin", "tern"]
            ]
            idx = random.choice(indices)
            words = ["dig", "un", "bin", "tern"]
            words.remove(sample[idx])
            sample[idx] = random.choice(words)

    if type != 0:
        sample[idx] = f"\033[91m{sample[idx]}\033[0m"  # Highlight the error
    print(" ".join(sample))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate samples by error type")
    parser.add_argument(
        "--config", type=str, default="config/conf.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--n", type=int, default=10, help="Number of samples to generate"
    )
    parser.add_argument(
        "--type",
        type=int,
        choices=[0, 1, 2, 3],
        default=0,
        help="Type of error to generate (0: no error, 1: clear error, 2: no effect, 3: unclear error)",
    )
    parser.add_argument(
        "--original",
        action="store_true",
        help="If set, show original samples without errors. Only works if type is not 0.",
    )
    args = parser.parse_args()

    dataset = load_dataset_from_config_file(args.config)
    for _ in range(args.n):
        generate_by_type(args.type, args.original, dataset)
