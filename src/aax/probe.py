"""Rol ifadesi probe'u — spec Bölüm 8, Sapma 2.

Makale her yanıtı LLM hakeme sorar. Bizim gateway bütçemiz buna yetmiyor
(16.000 rollout ≈ 1600 çağrı, aşama bütçesi 300). Bunun yerine 2.000 yanıtı
hakeme sorup etiketleriyle bge-m3 embedding'leri üzerine lojistik regresyon
oturtuyoruz.

Geri çekilme kuralı: held-out uyum %85'in altındaysa probe atılır ve rol
düzeyinde kaba bir tut/at filtresine dönülür. Bu karar raporlanır.
"""
from __future__ import annotations

import random
from collections import defaultdict

import numpy as np

HOLDOUT_FRACTION = 0.2
FALLBACK_THRESHOLD = 0.85


def stratified_sample(records: list[dict], n: int, *, seed: int) -> list[int]:
    """Rol başına mümkün olduğunca dengeli n indeks seç."""
    if n > len(records):
        raise ValueError(f"örnek sayısı popülasyondan büyük: {n} > {len(records)}")

    by_role: dict[str | None, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_role[record.get("role")].append(index)

    rng = random.Random(seed)
    for indices in by_role.values():
        rng.shuffle(indices)

    roles = sorted(by_role, key=lambda r: (r is None, r))
    chosen: list[int] = []
    cursor = 0
    while len(chosen) < n:
        progressed = False
        for role in roles:
            bucket = by_role[role]
            if cursor < len(bucket):
                chosen.append(bucket[cursor])
                progressed = True
                if len(chosen) == n:
                    break
        if not progressed:
            break
        cursor += 1
    return sorted(chosen)


def embed_answers(answers: list[str], *, model_id: str = "BAAI/bge-m3") -> np.ndarray:
    """bge-m3 ile yanıt embedding'leri. L2 normalize."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id)
    return model.encode(
        answers, batch_size=16, normalize_embeddings=True, show_progress_bar=True
    )


class RoleExpressionProbe:
    """bge-m3 embedding'leri üzerine lojistik regresyon."""

    def __init__(self, *, seed: int = 0) -> None:
        self.seed = seed
        self._model = None
        self.holdout_agreement: float | None = None

    def fit(self, embeddings: np.ndarray, labels: list[str]) -> None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split

        x_train, x_test, y_train, y_test = train_test_split(
            embeddings,
            labels,
            test_size=HOLDOUT_FRACTION,
            random_state=self.seed,
            stratify=labels if len(set(labels)) > 1 else None,
        )
        model = LogisticRegression(max_iter=2000, random_state=self.seed)
        model.fit(x_train, y_train)
        self._model = model
        self.holdout_agreement = float(model.score(x_test, y_test))

    def predict(self, embeddings: np.ndarray) -> list[str]:
        if self._model is None:
            raise RuntimeError("probe eğitilmedi — önce fit() çağır")
        return list(self._model.predict(embeddings))

    @property
    def is_trustworthy(self) -> bool:
        return (
            self.holdout_agreement is not None
            and self.holdout_agreement >= FALLBACK_THRESHOLD
        )
