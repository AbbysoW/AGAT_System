from collections import defaultdict
from heapq import heappush, heappop
import re
import json
import gzip
import csv
import sys
from datetime import datetime

SPECIAL_TOKENS = {
    0: "/unc",
    1: "/eos",
    2: "/hint",
    3: "/sys",
    4: "/cv",
    5: "/user",
    6: "/model"
}

# Регулярка компилируется один раз
_WORD_RE = re.compile(r"[А-Яа-яЁёA-Za-z0-9]+")
_SPECIAL_RE = re.compile("|".join(re.escape(t) for t in SPECIAL_TOKENS.values()))


class IncrementalBPE:

    def __init__(self, num_merges: int = 1000, min_freq: int = 2):
        self.num_merges = num_merges
        self.min_freq = min_freq          # пары реже min_freq не сливаем
        self.vocab: dict[tuple, int] = {} # word_tuple -> freq
        self.pair_freq: dict[tuple, int] = defaultdict(int)
        self.merges: list[tuple[str, str]] = []

    # ── предобработка ──────────────────────────────────────────────────────
    @staticmethod
    def _pretokenize(text: str) -> list[tuple]:
        text = _SPECIAL_RE.sub(" ", text)
        return [tuple(list(w) + ["</w>"]) for w in _WORD_RE.findall(text)]

    @staticmethod
    def _word_pairs(word: tuple):
        return zip(word, word[1:])

    # ── обновление vocab + pair_freq ───────────────────────────────────────
    def feed(self, text: str):
        chunk_freq: dict[tuple, int] = defaultdict(int)
        for w in self._pretokenize(text):
            chunk_freq[w] += 1

        for word, freq in chunk_freq.items():
            if word in self.vocab:
                self.vocab[word] += freq
            else:
                self.vocab[word] = freq
            for p in self._word_pairs(word):
                self.pair_freq[p] += freq

    # ── применить одно слияние ─────────────────────────────────────────────
    def _apply_merge(self, pair: tuple[str, str]):
        """Только слова содержащие pair — O(affected) вместо O(vocab)."""
        merged = "".join(pair)
        a, b = pair

        # Индекс: какие слова содержат пару (строим лениво при первом вызове)
        # Проще — просто фильтруем vocab, это уже быстрее чем раньше
        # т.к. убрали лишние операции внутри цикла
        affected = {w: f for w, f in self.vocab.items()
                    if a in w and any(w[i] == a and w[i+1] == b
                                      for i in range(len(w)-1))}

        for word, freq in affected.items():
            # удаляем старые пары этого слова из pair_freq
            for p in self._word_pairs(word):
                self.pair_freq[p] -= freq

            # строим новое слово
            new_word: list[str] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i+1] == b:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_t = tuple(new_word)

            # обновляем vocab
            del self.vocab[word]
            self.vocab[new_word_t] = self.vocab.get(new_word_t, 0) + freq

            # добавляем новые пары
            for p in self._word_pairs(new_word_t):
                self.pair_freq[p] += freq

    # ── финализация с heap ─────────────────────────────────────────────────
    def finalize(self, merges_path="vocabulary/bpe_merges.txt.gz"):
        # Heap: (-freq, a, b) — max-heap через отрицание
        heap = []
        for (a, b), f in self.pair_freq.items():
            if f >= self.min_freq:
                heappush(heap, (-f, a, b))

        done = 0
        with gzip.open(merges_path, "wt", encoding="utf-8") as fout:
            fout.write("#version: 1.0\n")
            while heap and done < self.num_merges:
                neg_f, a, b = heappop(heap)
                pair = (a, b)
                # Проверяем актуальность записи из heap
                actual = self.pair_freq.get(pair, 0)
                if actual < self.min_freq:
                    continue
                if -neg_f != actual:
                    # устаревшая запись — кладём актуальную обратно
                    if actual >= self.min_freq:
                        heappush(heap, (-actual, a, b))
                    continue

                self._apply_merge(pair)
                self.merges.append(pair)
                fout.write(f"{a} {b}\n")
                done += 1

                if done % 100 == 0:
                    print(f"  merge {done}/{self.num_merges} | vocab={len(self.vocab)} time:{datetime.now()}")

                # Добавляем новую объединённую пару в heap если она появилась
                new_token = a + b
                for (x, y), f in self.pair_freq.items():
                    if (x == new_token or y == new_token) and f >= self.min_freq:
                        heappush(heap, (-f, x, y))

        print(f"Готово: {done} слияний, vocab={len(self.vocab)} time: {datetime.now()}")

    # ── сохранить словарь ──────────────────────────────────────────────────
    def save_vocab(self, path="vocabulary/bpe_vocabulary.json"):
        subwords: set[str] = set()
        for word in self.vocab:
            for token in word:
                subwords.add(token)

        full_vocab: dict[str, int] = {t: i for i, t in SPECIAL_TOKENS.items()}
        next_id = len(SPECIAL_TOKENS)
        for sw in sorted(subwords):
            if sw not in full_vocab:
                full_vocab[sw] = next_id
                next_id += 1

        with open(path, "w", encoding="utf-8") as f:
            json.dump(full_vocab, f, ensure_ascii=False)

        print(f"Словарь сохранён: {next_id} токенов → {path}")
        return full_vocab


# ── запуск ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from torch.utils.data import IterableDataset

    DIR_PATH = "/mnt/windows/Programs/Datasets/row_text/"
    FILES = ["habr.csv", "lenta_ru_news_v1.csv", "poems.csv", "ria_novosti.csv"]

    csv.field_size_limit(sys.maxsize)

    class StreamDataset(IterableDataset):
        def __init__(self, dir_path, file_names, text_column="text"):
            self.paths = [dir_path + f for f in file_names]
            self.text_column = text_column

        def __iter__(self):
            for path in self.paths:
                with open(path, newline="", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        yield row[self.text_column]

    trainer = IncrementalBPE(num_merges=10_000, min_freq=2)

    print(f"Begun feeding:{datetime.now()}")

    for i, text in enumerate(StreamDataset(DIR_PATH, FILES)):
        trainer.feed(text)
        if i % 50_000 == 0:
            print(f"  fed {i} rows | vocab={len(trainer.vocab)}")

    print(f"Done feeding: {datetime.now()}")

    trainer.finalize("vocabulary/bpe_merges.txt.gz")
    trainer.save_vocab("vocabulary/bpe_vocabulary.json")