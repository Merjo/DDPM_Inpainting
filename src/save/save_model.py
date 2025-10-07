import torch
import pandas as pd
import datetime

def save_model(best_params, best_model, best_value):
    print("Best params:", best_params)
    print("Best value:", best_value)

    timestamp = datetime.now().strftime("%m%d_%H")

    params_filename = f"best_params_{best_value:.4f}_{timestamp}.csv"
    model_filename = f"best_model_{best_value:.4f}_{timestamp}.pt"

    # Save params
    df = pd.DataFrame([best_params])
    df["loss_value"] = best_value
    df.to_csv(params_filename, index=False)

    # Save model
    torch.save(best_model, model_filename)

    print("Best params saved to:", params_filename)
    print("Best model saved to:", model_filename)
    
    print('\n\n\n')

    return params_filename, model_filename