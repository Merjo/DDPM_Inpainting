from src.run.run_best import load_model
from src.config import cfg
from src.utils.output_manager import OutputManager

from src.run.run_sampling import run_sampling
from src.config import cfg

def run_extensive_sampling(param_file,
                           model_file,
                           sizes,
                           n=8):
    
    output = OutputManager(run_type="Xsampling")

    best_loss = float(param_file.split('/')[-1].split("_")[2])  # TODO make more stable in case of param file name changes

    for size in sizes:
        print(f'\n[Extensive Sampling] Running sampling for size: {size} \n')
        run_sampling(param_file=param_file,
                    model_file=model_file,
                    n=n,
                    width=size,
                    height=size,
                    verbose=False)

    diffusion, unet, params, optimizer, scheduler = load_model(param_file=param_file,
                                                               model_file=model_file)
    output.finalize(best_loss, unet, epochs=cfg.epochs, params=params)

if __name__=='__main__':

    param_file = 'output_new/0.09454_normal_daily_Dec10_1307_256_0.0/best_params_0.09454_256_0.0_8_1_normal_daily.csv'
    model_file = 'output_new/0.09454_normal_daily_Dec10_1307_256_0.0/model_0.09454.pkl'
    sizes = [64*i for i in range(4,20)]  # 256 to 1216
    n = 1

    #param_file, model_file, best_loss = find_best_saved_model()

    run_extensive_sampling(param_file=param_file, model_file=model_file, sizes=sizes, n=n)
