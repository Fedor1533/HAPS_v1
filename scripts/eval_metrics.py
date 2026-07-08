import itertools
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr


def calculate_smart_auc(y_true, y_score, test_direction=True, verbose=False):
    """
    Рассчитывает AUC с автоматическим определением направления метрики.

    Поддерживает:
    - Бинарную классификацию (2 класса).
    - Мультиклассовую ординальную классификацию (3+ класса) - Macro-Average One-vs-One.
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score, dtype=float)

    # 1. Проверка направления метрики через корреляцию Спирмена
    if test_direction:
        # ожидаем, что с ростом класса (0 -> 2) значение метрики должно расти.
        corr, _ = spearmanr(y_true, y_score)
        if corr < 0:
            if verbose:
                print(f"[INFO] Обнаружена отрицательная корреляция ({corr:.3f}). Инвертируем метрику.")
            y_score = -y_score

    unique_classes = np.sort(np.unique(y_true))
    n_classes = len(unique_classes)

    # 2. Бинарный случай
    if n_classes == 2:
        return roc_auc_score(y_true, y_score)

    # 3. Мультиклассовый случай (Macro-Average One-vs-One)
    pair_aucs = []
    # Генерируем все возможные пары классов: (0,1), (0,2), (1,2)
    # Порядок важен: c1 - меньший класс, c2 - больший класс
    for c1, c2 in itertools.combinations(unique_classes, 2):
        # Маска для выбора только текущей пары классов
        mask = np.isin(y_true, [c1, c2])

        y_true_subset = y_true[mask]
        y_score_subset = y_score[mask]

        # Бинаризация для текущей пары:
        y_true_binary = (y_true_subset == c2).astype(int)

        # Считаем AUC для пары
        pair_auc = roc_auc_score(y_true_binary, y_score_subset)
        pair_aucs.append(pair_auc)
        if verbose:
            print(f"   AUC для пары {c1} vs {c2}: {pair_auc:.4f}")

    # Усредняем полученные AUC
    return np.mean(pair_aucs)
