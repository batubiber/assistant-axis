import numpy as np
import pytest

from aax.probe import RoleExpressionProbe, stratified_sample


def make_records(n_roles=4, per_role=50):
    return [
        {"kind": "role", "role": f"rol{r}", "answer": f"yanit {r}-{i}"}
        for r in range(n_roles)
        for i in range(per_role)
    ]


def test_stratified_sample_is_balanced_across_roles():
    records = make_records(n_roles=4, per_role=50)
    idx = stratified_sample(records, n=40, seed=1)
    roles = [records[i]["role"] for i in idx]
    counts = {r: roles.count(r) for r in set(roles)}
    assert len(idx) == 40
    assert max(counts.values()) - min(counts.values()) <= 1


def test_stratified_sample_is_deterministic():
    records = make_records()
    assert stratified_sample(records, n=20, seed=7) == stratified_sample(records, n=20, seed=7)


def test_stratified_sample_differs_with_seed():
    records = make_records()
    assert stratified_sample(records, n=20, seed=7) != stratified_sample(records, n=20, seed=8)


def test_stratified_sample_rejects_n_larger_than_population():
    records = make_records(n_roles=2, per_role=3)
    with pytest.raises(ValueError, match="örnek"):
        stratified_sample(records, n=100, seed=1)


def test_probe_learns_a_separable_signal():
    """Doğrusal ayrılabilir sentetik veride probe neredeyse mükemmel olmalı.

    Bu testin amacı probe'un ÇALIŞTIĞINI göstermek, gerçek veride
    başarılı olacağını değil."""
    rng = np.random.default_rng(0)
    n = 200
    fully = rng.normal(loc=+3.0, scale=1.0, size=(n, 8))
    no = rng.normal(loc=-3.0, scale=1.0, size=(n, 8))
    embeddings = np.vstack([fully, no])
    labels = ["fully"] * n + ["no"] * n

    probe = RoleExpressionProbe(seed=0)
    probe.fit(embeddings, labels)
    assert probe.holdout_agreement > 0.95


def test_probe_reports_low_agreement_on_pure_noise():
    """Sinyal yoksa probe bunu saklamamalı — geri çekilme kuralı buna bakar."""
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(400, 8))
    labels = ["fully" if i % 2 == 0 else "no" for i in range(400)]

    probe = RoleExpressionProbe(seed=0)
    probe.fit(embeddings, labels)
    assert probe.holdout_agreement < 0.75


def test_probe_handles_all_three_production_categories():
    """Üretimde etiketler HER ZAMAN üç kategoridir (fully/somewhat/no).

    Yukarıdaki testlerin tamamı ikili etiketle çalışıyor — bu, gerçek bir
    çok sınıflı (multiclass) bozulmayı (ör. sklearn çağrısının sessizce
    ikili davranışa geri düşmesi ya da üçüncü sınıfın kaybolması) saklayacak
    en olası boşluktu.
    """
    rng = np.random.default_rng(0)
    n = 150
    fully = rng.normal(loc=+5.0, scale=0.5, size=(n, 8))
    somewhat = rng.normal(loc=0.0, scale=0.5, size=(n, 8))
    no = rng.normal(loc=-5.0, scale=0.5, size=(n, 8))
    embeddings = np.vstack([fully, somewhat, no])
    labels = ["fully"] * n + ["somewhat"] * n + ["no"] * n

    probe = RoleExpressionProbe(seed=0)
    probe.fit(embeddings, labels)
    assert probe.holdout_agreement > 0.9

    fresh_fully = rng.normal(loc=+5.0, scale=0.5, size=(5, 8))
    fresh_somewhat = rng.normal(loc=0.0, scale=0.5, size=(5, 8))
    fresh_no = rng.normal(loc=-5.0, scale=0.5, size=(5, 8))

    assert probe.predict(fresh_fully) == ["fully"] * 5
    assert probe.predict(fresh_somewhat) == ["somewhat"] * 5
    assert probe.predict(fresh_no) == ["no"] * 5


def test_probe_predict_returns_one_label_per_row():
    rng = np.random.default_rng(0)
    embeddings = np.vstack([
        rng.normal(loc=+3.0, size=(50, 8)),
        rng.normal(loc=-3.0, size=(50, 8)),
    ])
    labels = ["fully"] * 50 + ["no"] * 50
    probe = RoleExpressionProbe(seed=0)
    probe.fit(embeddings, labels)
    out = probe.predict(rng.normal(size=(7, 8)))
    assert len(out) == 7
    assert set(out) <= {"fully", "somewhat", "no"}


def test_probe_refuses_to_predict_before_fit():
    probe = RoleExpressionProbe(seed=0)
    with pytest.raises(RuntimeError, match="eğitilmedi"):
        probe.predict(np.zeros((2, 8)))
