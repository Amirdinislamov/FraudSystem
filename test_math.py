"""
Минимальный автономный тест математических компонентов БЕЗ FastAPI/Pydantic.
Проверяет формулы, не зависящие от фреймворков:

    1. Welford online: m2 / (n - 1) даёт несмещённую дисперсию.
    2. Haversine: реалистичная дистанция между Ташкентом и Лондоном.
    3. Benford expected: суммирование даёт 1.0.
    4. Poisson PMF корректна.
    5. Circular mean angle: круговое усреднение.

Запуск:    python3 test_math.py    (не требует зависимостей кроме math)
"""

import math


def welford_var(samples):
    n = 0
    mean = 0.0
    m2 = 0.0
    for x in samples:
        n += 1
        delta = x - mean
        mean += delta / n
        delta2 = x - mean
        m2 += delta * delta2
    if n < 2:
        return 0.0
    return m2 / (n - 1)


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def benford_expected():
    return [math.log10(1 + 1.0 / d) for d in range(1, 10)]


def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def circular_mean(angles):
    s = sum(math.sin(a) for a in angles)
    c = sum(math.cos(a) for a in angles)
    return math.atan2(s, c)


def run():
    failures = []

    # 1. Welford: Var([1,2,3,4,5]) = 2.5 (sample), 2.0 (population).
    var = welford_var([1, 2, 3, 4, 5])
    if abs(var - 2.5) > 1e-9:
        failures.append(f"Welford sample variance: ожидалось 2.5, получено {var}")

    # 2. Tashkent (41.31, 69.24) ↔ London (51.5, -0.12) ≈ 5260 км
    d = haversine(41.31, 69.24, 51.5, -0.12)
    if not (5000 < d < 5600):
        failures.append(f"Haversine TAS↔LON: ожидалось ~5260 км, получено {d:.0f}")

    # 3. Сумма ожидаемых частот Бенфорда = 1.0
    s = sum(benford_expected())
    if abs(s - 1.0) > 1e-9:
        failures.append(f"Benford expected sum: ожидалось 1.0, получено {s}")

    # 4. Poisson(k=1, λ=1) = e^-1 ≈ 0.3679
    p = poisson_pmf(1, 1.0)
    if abs(p - math.exp(-1)) > 1e-9:
        failures.append(f"Poisson PMF: {p}")

    # 5. Circular mean: углы, симметричные вокруг 0, дают 0.
    cm = circular_mean([-0.1, 0.0, 0.1])
    if abs(cm) > 1e-9:
        failures.append(f"Circular mean: {cm}")

    # 6. Welford устойчивость к большим n
    var2 = welford_var(list(range(1000)))
    expected = sum((x - 499.5) ** 2 for x in range(1000)) / 999
    if abs(var2 - expected) > 1e-6:
        failures.append(f"Welford on n=1000: {var2} vs {expected}")

    print("=" * 50)
    if failures:
        print(f"❌ FAIL ({len(failures)}):")
        for f in failures:
            print(f"   - {f}")
        return 1
    print("✅ Математические тесты пройдены")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
