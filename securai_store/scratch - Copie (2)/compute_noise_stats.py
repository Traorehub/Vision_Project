import csv, os, statistics, sys

csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'noise_matrix.csv'))
if not os.path.exists(csv_path):
    print('CSV not found at', csv_path)
    sys.exit(1)

values = []
with open(csv_path, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            values.append(float(row['noise']))
        except Exception:
            continue
if not values:
    print('No noise values found')
    sys.exit(1)

median_noise = statistics.median(values)
mean_noise = statistics.mean(values)
print('Rows:', len(values))
print('Median noise:', median_noise)
print('Mean noise:', mean_noise)
