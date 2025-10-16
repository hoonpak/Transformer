import sys, os
sys.path.append("../src")
import info

from tokenizers import Tokenizer, normalizers
from tokenizers.models import BPE
from tokenizers.normalizers import NFD, StripAccents
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.decoders import BPEDecoder
from tokenizers.processors import TemplateProcessing

files_path = ["../data/training/training_de.txt", "../data/training/training_en.txt"]

tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.normalizer = normalizers.Sequence([NFD(), StripAccents()])
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(vocab_size=info.ende_vocab_size, special_tokens=info.special_tokens, end_of_word_suffix="</w>", show_progress=True)
tokenizer.train(files=files_path, trainer=trainer)

tokenizer.post_processor = TemplateProcessing(
    single="<s> $A </s>",
    pair="<s> $A </s> $B:1 </s>:1",
    special_tokens=[
        ("<s>", tokenizer.token_to_id("<s>")),
        ("</s>", tokenizer.token_to_id("</s>")),
    ],
)
tokenizer.decoder = BPEDecoder()
# tokenizer.enable_padding(pad_token="<pad>", length=128)

tokenizer.save("../data/ende_WMT14_Tokenizer.json")

output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
print(output.tokens)
print(tokenizer.decode(output.ids))
