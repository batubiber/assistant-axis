"""Rollout promptlarının kurulması.

Makalenin kurulumu (Bölüm 2.1.1-2.1.2): her rol için birden çok sistem
promptu × ORTAK soru seti. Sorular roller arasında değişmez — değişen tek
şey sistem promptudur, çünkü ölçtüğümüz şey tam olarak sistem promptunun
aktivasyonlara etkisi.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Default Assistant davranışını yakalayan nötr promptlar (spec Aşama 1).
# None = hiç sistem promptu yok.
DEFAULT_SYSTEM_PROMPTS: tuple[str | None, ...] = (
    None,
    "You are a large language model.",
    "Respond as yourself.",
    "You are a helpful AI assistant.",
)


@dataclass(frozen=True)
class RolloutSpec:
    kind: str  # "role" | "default"
    role: str | None
    system_prompt: str | None
    question: str
    sample_index: int


def load_role_catalog(path: str | Path) -> list[dict]:
    """Kanonik rol kataloğunu yükle ve gerçekten kanonik olduğunu doğrula.

    Fail-closed: kısmi, pilot veya eksik bir katalog sessizce kabul edilirse
    tüm aşağı akış ölçümü yanlış rol kümesi üzerinde yapılır.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))

    if not payload.get("complete"):
        raise ValueError(
            f"{path}: complete=False — kısmi katalog. Aşama 0'ı tamamla."
        )
    if payload.get("limit") is not None:
        raise ValueError(
            f"{path}: limit={payload['limit']} — bu bir pilot artifact'i, katalog değil."
        )
    requested = payload.get("requested")
    catalog_size = payload.get("catalog_size")
    if requested != catalog_size:
        raise ValueError(
            f"{path}: katalog eksik — requested={requested}, catalog_size={catalog_size}"
        )
    # Üstteki üç doğrulama `.get()` kullanırken bu satır `payload["roles"]`
    # ile doğrudan indeksliyordu. Bozuk/eksik bir `roles.json` için sonuç
    # ÇIPLAK bir `KeyError`'dı ve `KeyError` bir `ValueError` DEĞİLDİR:
    # `06_label_and_train_probe.py`'nin `except ValueError` sarmalayıcısını
    # atlayıp yorumlayıcıyı çıkış 1 ile döndürüyordu — o script'te 1
    # "probe güvenilmez, geri çekilme kuralı" demek. Yani BOZUK BİR KATALOG,
    # PROBE HAKKINDA BİR BULGU olarak raporlanıyordu.
    roles = payload.get("roles")
    if not isinstance(roles, list):
        raise ValueError(
            f"{path}: 'roles' anahtarı yok ya da liste değil "
            f"({type(roles).__name__}). Dosya bozuk — Aşama 0'ı "
            "(scripts/00_generate_role_data.py) tekrar çalıştırın."
        )
    return roles


def build_role_specs(catalog: list[dict], questions: list[str]) -> list[RolloutSpec]:
    """Her rol × her sistem promptu × her ortak soru."""
    specs: list[RolloutSpec] = []
    for record in catalog:
        for instruction in record["instructions"]:
            for question in questions:
                specs.append(
                    RolloutSpec(
                        kind="role",
                        role=record["role"],
                        system_prompt=instruction,
                        question=question,
                        sample_index=0,
                    )
                )
    return specs


def build_default_specs(
    questions: list[str], *, samples_per_prompt: int = 10
) -> list[RolloutSpec]:
    """Default Assistant rollout'ları: nötr prompt × soru × tekrar."""
    specs: list[RolloutSpec] = []
    for system_prompt in DEFAULT_SYSTEM_PROMPTS:
        for question in questions:
            for sample_index in range(samples_per_prompt):
                specs.append(
                    RolloutSpec(
                        kind="default",
                        role=None,
                        system_prompt=system_prompt,
                        question=question,
                        sample_index=sample_index,
                    )
                )
    return specs


def to_chat_messages(spec: RolloutSpec) -> list[dict]:
    """Spec'i OpenAI/HF chat formatına çevir."""
    messages: list[dict] = []
    if spec.system_prompt is not None:
        messages.append({"role": "system", "content": spec.system_prompt})
    messages.append({"role": "user", "content": spec.question})
    return messages
