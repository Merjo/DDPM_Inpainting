from src.save.save_plot import plot_inpainting_mse_curves
import pandas as pd
from src.config import cfg

cfg.update_output_path('table_lam_cov_comparison')

table1path = "output_new/0.04342_extensive_inpainting_Nov18_2149_256_0.1/inpainting_results/inpainting_mse_results.csv"
df1 = pd.read_csv(table1path)
table2path = "output_new/0.04342_extensive_inpainting_Nov19_1123_256_0.1/inpainting_results/inpainting_mse_results.csv"
df2 = pd.read_csv(table2path)

df = pd.concat([df1, df2], ignore_index=True)

# Overwrites! TODO Decide

new_path = "results.csv"
df = pd.read_csv(new_path)

# If the desired curve is: one MSE per lambda per coverage
df = df.sort_values(["coverage", "lambda"])
df = df.drop_duplicates(subset=["coverage", "lambda"], keep="first")

plot_inpainting_mse_curves(df, title="Extensive Inpainting MSE Curves (Combined Runs)")
