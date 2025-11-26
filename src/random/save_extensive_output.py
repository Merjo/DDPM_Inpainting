import re
import csv

log_file = "logs/run_extensive_inpainting_5608954.out"
log_file = "logs/run_extensive_inpainting_5624381.out"
output_csv = "results.csv"

# Regex to extract:
# coverage (the known-data fraction, e.g. 0.1%)
# lambda value
# MSE value
coverage_re = re.compile(r"Testing inpainting with ([0-9.]+)% known data and lambda ([0-9.]+):")
mse_re = re.compile(r"Inpainting MSE \(masked region\): ([0-9.]+)")

results = []

with open(log_file, "r") as f:
    lines = f.readlines()

current_coverage = None
current_lambda = None

for line in lines:
    cov_match = coverage_re.search(line)
    if cov_match:
        # Convert "0.1%" → 0.001
        coverage_percent = float(cov_match.group(1))
        current_coverage = coverage_percent / 100
        current_lambda = float(cov_match.group(2))
        continue

    mse_match = mse_re.search(line)
    if mse_match and current_coverage is not None and current_lambda is not None:
        mse = float(mse_match.group(1))
        results.append((current_coverage, current_lambda, mse))
        # reset lambda to avoid mis-association
        current_lambda = None

# Write CSV
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["coverage", "lambda", "mse"])
    writer.writerows(results)

print(f"Saved {len(results)} rows to {output_csv}")
