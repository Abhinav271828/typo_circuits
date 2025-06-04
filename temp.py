from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
trainer = BpeTrainer(vocab_size=30, special_tokens=["[UNK]"])

tokenizer.pre_tokenizer = Whitespace()
tokenizer.train(["ten.txt"], trainer=trainer)

with open("ten.txt", "r") as f:
    for line in f:
        output = tokenizer.encode(line.strip())
        print(output.tokens)

tokenizer.save("ten.json")
tokenizer = Tokenizer.from_file("ten.json")

with open("ten.txt", "r") as f:
    for line in f:
        output = tokenizer.encode(line.strip())
        print(output.tokens)
