import sys, os
sys.path.append("../src")
import info

from tokenizers import Tokenizer, normalizers
from tokenizers.models import WordPiece as WordPieceModel
from tokenizers.normalizers import NFD, StripAccents
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.decoders import WordPiece as WordPieceDecoder
from tokenizers.processors import TemplateProcessing

files_path = ["../data/training/training_enfr_en.txt", "../data/training/training_enfr_fr.txt"]

tokenizer = Tokenizer(WordPieceModel(unk_token="<unk>"))
tokenizer.normalizer = normalizers.Sequence([NFD(), StripAccents()])
tokenizer.pre_tokenizer = Whitespace()

trainer = WordPieceTrainer(vocab_size=info.enfr_vocab_size, special_tokens=info.special_tokens, show_progress=True)
tokenizer.train(files=files_path, trainer=trainer)

tokenizer.post_processor = TemplateProcessing(
    single="<s> $A </s>",
    pair="<s> $A </s> $B:1 </s>:1",
    special_tokens=[
        ("<s>", tokenizer.token_to_id("<s>")),
        ("</s>", tokenizer.token_to_id("</s>")),
    ],
)
tokenizer.decoder = WordPieceDecoder()

tokenizer.save("../data/enfr_WMT14_Tokenizer.json")

output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
print(output.tokens)
print(tokenizer.decode(output.ids))