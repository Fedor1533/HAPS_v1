import pandas as pd

# Исходный CSV
input_csv = '~/benchmarking/filters/train_classification_renamed_columns.csv'

# Куда сохранить
output_csv = '~/benchmarking/filters/train_classification_renamed_columns_2.csv'

# Загружаем
df = pd.read_csv(input_csv)

# patch_id имеет вид:
# 89M2100212_9_19
# -> pname = 89M2100212
df["pname"] = df["patch_id"].str.rsplit("_", n=2).str[0]

# Проверка
print(f"Unique WSI: {df['pname'].nunique()}")
print(df["pname"].value_counts().head())

# Сохраняем
df.to_csv(output_csv, index=False)

print(f"Saved to {output_csv}")