import torch
import pandas as pd
from datetime import datetime
import os

def save_model(params, model, value):
    print("Best params:", params)
    print("Best value:", value)

    timestamp = datetime.now().strftime("%m%d_%H")

    params_dir = "output/params"
    model_dir = "output/models"

    params_filename = os.path.join(params_dir, f"best_params_{value:.4f}_{timestamp}.csv")
    model_filename = os.path.join(model_dir, f"best_model_{value:.4f}_{timestamp}.pt")


    # Save params
    df = pd.DataFrame([params])
    df["loss_value"] = value
    df.to_csv(params_filename, index=False)

    # Save model
    torch.save(model, model_filename)

    print("Best params saved to:", params_filename)
    print("Best model saved to:", model_filename)
    
    print('\n\n\n')

    return params_filename, model_filename