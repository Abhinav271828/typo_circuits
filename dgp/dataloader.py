import torch
from torch.utils.data import DataLoader
import numpy as np
import random
import time
import os
from .PCFG import PCFG
import pickle as pkl


def get_dataloader(
    language: str = "",  # in ['english', 'expr', 'dyck']
    config: dict = {},  # config for PCFG. see below for other languages.
    alpha: float = 1e5,
    prior_type: str = "dirichlet",
    num_iters: int = 1e6,
    max_sample_length: int = 128,
    seed: int = 42,
    batch_size: int = 32,
    num_workers: int = 4,
):
    """Define the PCFG dataloader.

    Args:
        language: The language of the PCFG. One of ['english', 'expr', 'dyck1', 'dyck2'].
        config: The configuration of the PCFG. The keys depend on the language.
        alpha (float, optional): The concentration parameter for the Dirichlet distribution. Defaults to 1e5.
        prior_type (str, optional): The type of prior distribution. Defaults to 'dirichlet'.
        num_iters (int, optional): The number of iterations to make in the training loop per epoch. Defaults to 1e6.
        max_sample_length (int, optional): The maximum length of a sequence. Defaults to 128.
        seed (int, optional): The random seed. Defaults to 42.
        batch_size (int, optional): The batch size. Defaults to 32.
        num_workers (int, optional): The number of workers. Defaults to 4.

    Returns:
        DataLoader: A pytorch compatible, PCFG dataloader.
    """

    # Create a dataset
    dataset = PCFGDataset(
        language=language,
        config=config,
        alpha=alpha,
        prior_type=prior_type,
        num_iters=num_iters,
        max_sample_length=max_sample_length,
        seed=seed,
    )

    # Create a dataloader
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        pin_memory=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return dataloader


class PCFGDataset:
    def __init__(
        self,
        language: str = "",  # in ['english', 'expr', 'dyck']
        config: dict = {},  # config for PCFG. see below for other languages.
        alpha: float = 1e5,
        prior_type: str = "dirichlet",
        num_iters: int = 1e6,
        max_sample_length: int = 128,
        seed: int = 42,
    ):
        """Define the PCFG dataset.

        Args:
            language: The language of the PCFG. One of ['english', 'expr', 'dyck1', 'dyck2'].
            config: The configuration of the PCFG. The keys depend on the language.
            alpha (float, optional): The concentration parameter for the Dirichlet distribution. Defaults to 1e5.
            prior_type (str, optional): The type of prior distribution. Defaults to 'dirichlet'.
            num_iters (int, optional): The number of iterations to make in the training loop per epoch. Defaults to 1e6.
            max_sample_length (int, optional): The maximum length of a sequence. Defaults to 128.
            seed (int, optional): The random seed. Defaults to 42.

        Returns:
            PCFGDataset: A PCFG dataset.
        """

        # Some setup details
        self.num_iters = int(num_iters)
        self.max_sample_length = max_sample_length

        # Instructions / tasks (NOTE: this is where the code will be changed to add tasks)
        self.tasks_dict = {}
        for n_task, task in enumerate(["freegen"]):
            self.tasks_dict[f"Task{n_task}"] = task
        self.task_tokens = list(self.tasks_dict.keys())
        self.prior_over_tasks = [1.0]

        # Define the PCFG
        self.PCFG = PCFG(
            language=language,
            config=config,
            alpha=alpha,
            prior_type=prior_type,
            seed=seed,
            tasks=self.tasks_dict,
        )

        # Instruction decorator
        self.instruction_decorator = "Task: {task_token} \n Ops: {ops} \n Out:"
        self.decorator_length = len(
            self.PCFG.tokenize_sentence(
                self.instruction_decorator.format(task_token="Task0", ops="<null>")
            )
        )

        ## Special tokens
        # Pad token
        self.pad_token = "<pad>"
        self.pad_token_id = self.PCFG.vocab["<pad>"]

        # Tokenize the task tokens
        self.task_token_idx = {
            t: self.PCFG.tokenize_sentence(t)[0] for t in self.task_tokens
        }

        # Define the PCFG generator
        self.generator = self.PCFG.sentence_generator(num_of_samples=self.num_iters)

        # Generation input template
        self.template = torch.tensor(
            self.PCFG.tokenize_sentence("Task: Task0 \n Ops: <null> \n Out:")
        )

    def save_grammar(self, path_to_results: str):
        """
        Save the grammar underlying the dataset
        """
        base_dir = os.path.join(path_to_results, "grammar")
        os.makedirs(base_dir, exist_ok=True)
        with open(os.path.join(base_dir, "PCFG.pkl"), "wb") as f:
            pkl.dump(self.PCFG, f)

    def load_grammar(self, path_to_results: str):
        """
        Load and override grammar of the dataset
        """
        base_dir = os.path.join(path_to_results, "grammar")
        with open(os.path.join(base_dir, "PCFG.pkl"), "rb") as f:
            self.PCFG = pkl.load(f)

    def __len__(self):
        """
        Return the number of iterations made in the training loop per epoch.
        """
        return self.num_iters

    def __getitem__(self, index):
        """
        Get the next sequence from the PCFG generator.
        """

        while True:

            # Generate a sequence from the PCFG
            sequence = self.generator.__next__()

            # Tokenize the sequence
            sequence = torch.tensor(self.PCFG.tokenize_sentence(sequence))
            seq_length = float(sequence.size(0))

            idx = -1
            err = -1
            if random.random() < self.PCFG.error_rate:
                match self.PCFG.error_type:
                    case "swp":
                        # Swap a random token with its neighbour
                        if sequence.size(0) > 2:
                            idx = np.random.randint(0, sequence.size(0) - 1)
                            sequence[idx], sequence[idx + 1] = (
                                sequence[idx + 1],
                                sequence[idx],
                            )
                        err = 0
                    case "del":
                        # Delete a random token
                        if sequence.size(0) > 1:
                            idx = np.random.randint(0, sequence.size(0))
                            sequence = torch.cat((sequence[:idx], sequence[idx + 1 :]))
                        err = 1
                    case "ins":
                        # Insert a random token
                        idx = np.random.randint(0, sequence.size(0) + 1)
                        tok = np.random.randint(0, len(self.PCFG.vocab) - 8)
                        sequence = torch.cat(
                            (sequence[:idx], torch.tensor([tok]), sequence[idx:])
                        )
                        err = 2
                    case "sub":
                        # Substitute a random token with another token
                        idx = np.random.randint(0, sequence.size(0))
                        tok = np.random.randint(0, len(self.PCFG.vocab) - 8)
                        sequence[idx] = tok
                        err = 3

            # Define instruction (NOTE: kept null for now, but the code can be changed here to implement tasks)
            task_token = np.random.choice(self.task_tokens, p=self.prior_over_tasks)
            if task_token == "Task0":
                ops = "<null>"
            else:
                raise ValueError(f"Invalid task token: {task_token}")
            instr = torch.tensor(
                self.PCFG.tokenize_sentence(
                    self.instruction_decorator.format(task_token=task_token, ops=ops)
                )
            )

            # Concatenate the instruction to the sequence
            sequence = torch.cat(
                (instr, sequence, torch.tensor([self.PCFG.vocab["<eos>"]]))
            )

            # Truncate the sequence if it is longer than the max sequence length
            if sequence.size(0) > self.max_sample_length - 10:
                pass

            # Pad the sequence to the max sequence length with <pad>
            else:
                sequence = torch.cat(
                    (
                        sequence,
                        torch.tensor(
                            [self.pad_token_id]
                            * (self.max_sample_length - len(sequence))
                        ),
                    )
                )
                break

        return sequence, seq_length, idx, err
