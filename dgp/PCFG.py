from typing import Iterator, List, Tuple, Union
import random
import numpy as np
import nltk  # type: ignore
from nltk.grammar import ProbabilisticProduction  # type: ignore
from nltk.grammar import Nonterminal  # type: ignore
from .utils import define_prior

Symbol = Union[str, Nonterminal]


class ProbabilisticGenerator(nltk.grammar.PCFG):
    def generate(self, n: int = 1) -> Iterator[str]:
        """Probabilistically, recursively reduce the start symbol `n` times,
        yielding a valid sentence each time.

        Args:
            n: The number of sentences to generate.

        Yields:
            The next generated sentence.
        """
        for _ in range(n):
            x = self._generate_derivation(self.start())
            yield x

    def _generate_derivation(self, nonterminal: Nonterminal) -> str:
        """Probabilistically, recursively reduce `nonterminal` to generate a
        derivation of `nonterminal`.

        Args:
            nonterminal: The non-terminal nonterminal to reduce.

        Returns:
            The derived sentence.
        """
        sentence: List[str] = []
        symbol: Symbol
        derivation: str

        for symbol in self._reduce_once(nonterminal):
            if isinstance(symbol, str):
                derivation = symbol
            else:
                derivation = self._generate_derivation(symbol)

            if derivation != "":
                sentence.append(derivation)

        return " ".join(sentence)

    def _reduce_once(self, nonterminal: Nonterminal) -> Tuple[Symbol]:
        """Probabilistically choose a production to reduce `nonterminal`, then
        return the right-hand side.

        Args:
            nonterminal: The non-terminal symbol to derive.

        Returns:
            The right-hand side of the chosen production.
        """
        return self._choose_production_reducing(nonterminal).rhs()

    def _choose_production_reducing(
        self, nonterminal: Nonterminal
    ) -> ProbabilisticProduction:
        """Probabilistically choose a production that reduces `nonterminal`.

        Args:
            nonterminal: The non-terminal symbol for which to choose a production.

        Returns:
            The chosen production.
        """
        productions: List[ProbabilisticProduction] = self._lhs_index[nonterminal]
        probabilities: List[float] = [production.prob() for production in productions]
        return random.choices(productions, weights=probabilities)[0]


class PCFG:

    def __init__(
        self,
        language: str = "",  # in ['english', 'expr', 'dyck']
        config: dict = {},  # config depends on the language; see below
        alpha: float = 1e5,
        prior_type: str = "dirichlet",
        tasks: dict = None,
        seed: int = 42,
    ):
        """Define the PCFG object.

        Args:
            language: The language of the PCFG. One of ['english', 'expr', 'dyck1', 'dyck2'].
            config: The configuration of the PCFG. The keys depend on the language.
            alpha: The concentration parameter for the Dirichlet distribution.
            prior_type: The type of prior distribution.
            tasks: The tasks to perform.
            seed: The random seed.

        Returns:
            PCFG: A PCFG object.
        """

        # Set the random seed
        random.seed(seed)
        np.random.seed(seed)

        self.language = language
        self.alpha = alpha
        self.prior_type = prior_type

        # Grammar
        self.production_rules = None
        self.lexical_symbolic_rules = None

        # Concept classes object
        if language == "expr":
            self.n_digits = config["n_digits"]
            self.n_ops = config["n_ops"]
            self.grammar = self.create_grammar_expr(
                n_digits=self.n_digits,
                n_ops=self.n_ops,
            )

        else:
            raise ValueError(
                f"Language {language} not supported. Options are ['english', 'expr', 'dyck1', 'dyck2']."
            )

        # Tasks
        self.tasks = tasks

        # Set the vocabulary
        self.vocab, self.id_to_token_map, self.vocab_size = self.gather_vocabulary()

        # Parser
        self.parser = nltk.ViterbiParser(self.grammar)

    def create_grammar_expr(
        self,
        n_digits: int,
        n_ops: int,
    ):
        """Define the PCFG grammar.

        Args:
            n_digits: The number of digits in the vocabulary.
            n_ops: The number of operations in the vocabulary.

        Returns:
            The PCFG grammar.
        """

        # Define production rules
        self.production_rules = """
                S -> Expr [1.0]
                Expr -> OpExpr [0.40] | Digit [0.60]
                OpExpr -> UnOp Expr [0.33] | BinOp Expr Expr [0.33] | TernOp Expr Expr Expr [0.34]
                """

        self.lexical_symbolic_rules = ""

        ## Define lexical rules
        symbol_types = ["Digit", "UnOp", "BinOp", "TernOp"]
        n_symbol_to_tokens = [n_digits, n_ops, n_ops, n_ops]
        token_prefix = ["dig ", "un ", "bin ", "tern "]

        for symbol_type, n_symbol_to_token, prefix in zip(
            symbol_types, n_symbol_to_tokens, token_prefix
        ):
            prior_over_symbol = define_prior(
                n_symbol_to_token, alpha=self.alpha, prior_type=self.prior_type
            )
            rhs_symbol = ""
            for i in range(n_symbol_to_token):
                rhs_symbol += f"'{prefix}{i}' [{prior_over_symbol[i]}] | "
            rhs_symbol = rhs_symbol[:-3]
            self.lexical_symbolic_rules += f"{symbol_type} -> {rhs_symbol} \n"

        # Create the grammar
        return ProbabilisticGenerator.fromstring(
            self.production_rules + self.lexical_symbolic_rules
        )

    def gather_vocabulary(self):
        """Gather the vocabulary from the concept classes.

        Returns:
            The vocabulary.
        """

        # Gather concept classes' vocabulary
        vocab = {}
        vocab_size = 0
        if self.language == "expr":
            for token in ["dig", "un", "bin", "tern"] + list(range(10)):
                vocab[str(token)] = vocab_size
                vocab_size += 1
        vocab_size = len(vocab)

        # Add special tokens to be used for defining sequences in dataloader
        for special_token in [
            "<pad>",
            "Task:",
            "<null>",
            "Ops:",
            "Out:",
            "\n",
            "<eos>",
            "<sep>",
        ]:
            vocab[special_token] = vocab_size
            vocab_size += 1

        # Add task tokens
        for task_token in self.tasks:
            vocab[task_token] = vocab_size
            vocab_size += 1

        # Create an inverse vocabulary
        id_to_token_map = {v: k for k, v in vocab.items()}

        return vocab, id_to_token_map, vocab_size

    def tokenize_sentence(self, sentence: str) -> List[int]:
        """Tokenize a sentence.

        Args:
            sentence: The sentence to tokenize.

        Returns:
            The tokenized sentence.
        """

        # Tokenize the sentence
        tokens = sentence.split(" ")

        # Convert the tokens to indices
        token_indices = []
        for token in tokens:
            if token == "" or token == " ":
                continue
            else:
                token_indices.append(self.vocab[token])

        return token_indices

    def detokenize_sentence(self, token_indices) -> str:
        """Detokenize a sentence.

        Args:
            token_indices: The token indices to detokenize.

        Returns:
            The detokenized sentence.
        """

        # Convert the indices to tokens
        tokens = [self.id_to_token_map[token] for token in np.array(token_indices)]

        # Detokenize the tokens
        sentence = " ".join(tokens)

        return sentence

    def sentence_generator(
        self,
        num_of_samples: int,
    ) -> Iterator[str]:
        """
        1. Generate a sentence from the grammar
        2. Fill the sentence with values from the concept classes
        """

        # An iterator that dynamically generates symbolic sentences from the underlying PCFG
        symbolic_sentences = self.grammar.generate(num_of_samples)

        # Fill the sentences with values from the concept classes
        for s in symbolic_sentences:
            yield s

    def check_grammaticality(self, sentence: str) -> bool:
        """Check if a sentence is in the grammar.

        Args:
            sentence: The sentence to check.

        Returns:
            Whether the sentence is in the grammar.
        """

        # Remove instruction decorator and pad tokens
        if "Out:" in sentence:
            sentence = sentence.split("Out: ")
            sentence = sentence[1] if len(sentence) > 1 else sentence[0]
        if "<pad>" in sentence:
            sentence = sentence.split(" <pad>")
            sentence = sentence[0] if len(sentence) > 1 else sentence[0]

        # Tokenize the sentence
        tokens = sentence.split(" ")
        if "" in tokens:
            tokens.remove("")

        # Run parser
        try:
            parser_output = self.parser.parse(tokens).__next__()
            logprobs, height = parser_output.logprob(), parser_output.height()
            return (True, logprobs, height, None), len(tokens)
        except:
            failure = " ".join(tokens)
            return (False, None, None, failure), len(tokens)
