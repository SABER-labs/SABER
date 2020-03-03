import string

class Vocab(object):

    def __init__(self, blank_id=0, vocab=string.ascii_lowercase + string.whitespace[0]):
        super().__init__()
        self.vocab = vocab

    def encode(self, text):
        return [self.vocab.index(c) + 1 for c in text.lower() if c in self.vocab]

    def decode(self, tokens):
        return ''.join([self.vocab[token-1] for token in tokens if token > 0])

    def __len__(self):
        return len(self.vocab) + 1

if __name__ == "__main__":
    vc = Vocab()
    tensor = vc.encode("My, awesome daylight!")
    print(tensor)
    print(vc.decode(tensor))
    print(len(vc))