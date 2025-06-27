import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import ByteLevelBPETokenizer
import nltk
from nltk.corpus import gutenberg
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer

nltk.download("gutenberg")

def load_sentences(max_sentences=2000):
    text = gutenberg.raw("austen-emma.txt")
    trainer = PunktTrainer()
    trainer.train(text)
    tokenizer = PunktSentenceTokenizer(trainer.get_params())
    return tokenizer.tokenize(text)[:max_sentences]

def save_corpus(sentences, path="corpus.txt"):
    with open(path, "w", encoding="utf-8") as f:
        for s in sentences:
            f.write(s + "\n")

def train_tokenizer(path="corpus.txt"):
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=path, vocab_size=1000, min_frequency=2, special_tokens=["<PAD>", "<EOS>"])
    return tokenizer

def encode_sentences(sentences, tokenizer):
    return [tokenizer.encode(s).ids + [tokenizer.token_to_id("<EOS>")] for s in sentences]


class LanguageDataset(Dataset):
    def __init__(self, sequences, context_length=10):
        self.samples = []
        for seq in sequences:
            for i in range(1, len(seq)):
                context = seq[max(0, i - context_length):i]
                context = [0] * (context_length - len(context)) + context
                self.samples.append((context, seq[i]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


def train_model(model, loader, optimizer, loss_fn, epochs=2):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            out = model(x)
            loss = loss_fn(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.2f}")


def generate_sequence(model, tokenizer, prompt, strategy="greedy", context_length=10, max_tokens=20):
    model.eval()
    ids = tokenizer.encode(prompt).ids
    for _ in range(max_tokens):
        context = ids[-context_length:]
        context = [0] * (context_length - len(context)) + context
        input_tensor = torch.tensor([context])
        with torch.no_grad():
            logits = model(input_tensor)
            probs = F.softmax(logits[0], dim=0)
            if strategy == "greedy":
                next_id = torch.argmax(probs).item()
            else:
                next_id = torch.multinomial(probs, 1).item()
        if next_id == tokenizer.token_to_id("<EOS>"):
            break
        ids.append(next_id)
    return tokenizer.decode(ids, skip_special_tokens=True)

class MarkovChain:
    def __init__(self):
        self.transitions = {}

    def train(self, sequences):
        from collections import defaultdict, Counter
        self.transitions = defaultdict(Counter)
        for seq in sequences:
            for a, b in zip(seq[:-1], seq[1:]):
                self.transitions[a][b] += 1

    def generate(self, start_token, tokenizer, max_tokens=20):
        result = [start_token]
        for _ in range(max_tokens):
            if result[-1] not in self.transitions:
                break
            next_tokens = list(self.transitions[result[-1]].elements())
            if not next_tokens:
                break
            result.append(random.choice(next_tokens))
        return tokenizer.decode(result, skip_special_tokens=True)

def main():
    sentences = load_sentences()
    save_corpus(sentences)
    tokenizer = train_tokenizer()
    encoded = encode_sentences(sentences, tokenizer)

    dataset = LanguageDataset(encoded)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = LanguageModel(len(tokenizer.get_vocab()))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.CrossEntropyLoss()
    train_model(model, loader, optimizer, loss_fn, epochs=4)
    num_examples = 5

    print("\nGenerated (Greedy):")
    for s in sentences[:num_examples]:
        print(generate_sequence(model, tokenizer, s[:50], strategy="greedy"))

    print("\nGenerated (Sampling):")
    for s in sentences[:num_examples]:
        print(generate_sequence(model, tokenizer, s[:50], strategy="sample"))

    print("\nMarkov Chain:")
    markov = MarkovChain()
    markov.train(encoded)
    for s in sentences[:num_examples]:
        start_ids = tokenizer.encode(s[:20]).ids
        if start_ids:
            print(markov.generate(start_ids[-1], tokenizer))

if __name__ == "__main__":
    main()
