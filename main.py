from projected_diffusion.scoremodel import Model, AnnealedLangevinDynamic
from projected_diffusion import scoremodel, utils
import torch
import os
import argparse
import shutil

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')



def main():
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=['train', 'sample'], default='sample', help="Training or sampling")
    # Setting parameters
    parser.add_argument("--experiment", type=str, choices=['microstructures', 'trajectories', 'motion', 'human', 'other'], default='microstructures', help="Experiment setting")
    parser.add_argument("--projection_path", type=str, default=None, help="If experiment argument is \'other\', set path to custom projection operator")
    parser.add_argument("--model_path", type=str, default=None, help="Set path to diffusion model checkpoint")
    parser.add_argument("--train_set_path", type=str, default=None, help="Set path to training data if in training mode")
    parser.add_argument("--val_set_path", type=str, default=None, help="Set path to validation data if in training mode")
    parser.add_argument("--n_samples", type=int, default=1, help="Number of samples")
    # Model parameters
    parser.add_argument("--eps", type=float, default=1.5e-5, help="Epsilon of step size")
    parser.add_argument("--sigma_min", type=float, default=0.005, help="Sigma min of Langevin dynamic")
    parser.add_argument("--sigma_max", type=float, default=10., help="Sigma max of Langevin dynamic")
    parser.add_argument("--n_steps", type=int, default=10, help="Langevin steps")
    parser.add_argument("--annealed_step", type=int, default=25, help="Annealed steps")
    # Training parameters
    parser.add_argument("--total_iteration", type=int, default=3000, help="Total training iterations")
    parser.add_argument("--display_iteration", type=int, default=150, help="Logging frequency")
    parser.add_argument("--run_name", type=str, default='train', help="Run name for logging and saving")
    # Projection parameters
    parser.add_argument("--porosity", type=float, default=0.25, help="Porosity for microstructure projection [leave blank if not running \'microstructures\' sampling]")
    parser.add_argument("--gravity", type=float, default=9.8, help="Gravity for motion projection [leave blank if not running \'motion\' sampling]")
    

    args = parser.parse_args()

    if args.mode == 'train':
        _train(args)
    else:
        _sample(args)
    


def _train(args):
    
    # Load model
    model = Model(device, args.n_steps, args.sigma_min, args.sigma_max)
    # Resume training if applicable
    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.train()
    dynamic = AnnealedLangevinDynamic(args.sigma_min,args.sigma_max, args.n_steps, args.annealed_step, model, None, device, eps= args.eps)
    optim = torch.optim.Adam(model.parameters(), lr = 0.005)

    # Generate dataset
    if args.experiment != 'other':
        create_outputs_directory(f"examples/{args.experiment}/models")
        create_outputs_directory(os.path.join(f"examples/{args.experiment}/models", args.run_name))
        models_dir = f"examples/{args.experiment}/models"
        
    else:
        create_outputs_directory(f"examples/{args.projection_path}/models")
        create_outputs_directory(os.path.join(f"examples/{args.projection_path}/models", args.run_name))
        models_dir = f"examples/{args.projection_path}/models"
        
    
    if args.experiment == 'microstructures':
        from examples.microstructures.dataloader import get_dataset
        train_loader, validation_loader = get_dataset(args.train_set_path, args.val_set_path)

    elif args.experiment == 'trajectories':
        raise NotImplementedError('Support will be added in future releases.')
        from examples.trajectories.dataloader import get_dataset
        train_loader, validation_loader = get_dataset(args.train_set_path, os.path.join(f"examples/{args.experiment}/models", args.run_name), device)

    elif args.experiment == 'motion':
        from examples.motion.dataloader import get_dataset
        train_loader, validation_loader = get_dataset(args.train_set_path, args.val_set_path)
    
    elif args.experiment == 'human':
        raise NotImplementedError('Support will be added in future releases.')
    

    else:
        module_path = f"examples.{args.projection_path}.dataloader"
        projection_module = __import__(module_path, fromlist=["get_dataset"])
        get_dataset = getattr(projection_module, "get_dataset")
        train_loader, validation_loader = get_dataset(args.train_set_path, args.val_set_path)

    # Set up trainer
    current_iteration = 0
    best_val_loss = float('inf') 
    create_outputs_directory()


    # Set up logging
    losses = scoremodel.AverageMeter('Loss', ':.4f')
    progress = scoremodel.ProgressMeter(args.total_iteration, [losses], prefix='Iteration ')

    while current_iteration != args.total_iteration:
    
        ## Training Routine ##
        model.train()
        for data, _ in train_loader:
            data = data.to(device)
            loss = model.loss_fn(data)
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.update(loss.item())
            
        progress.display(current_iteration)
        current_iteration += 1
        
        
        ## Validation Routine ##
        model.eval()
        val_loss_accumulator = 0.0
        val_steps = 0
        with torch.no_grad():
            for data, _ in validation_loader:
                data = data.to(device)                
                val_loss = model.loss_fn(data)
                val_loss_accumulator += val_loss.item()
                val_steps += 1

        # Compute average validation loss for the epoch
        avg_validation_loss = val_loss_accumulator / val_steps
        
        # Checkpointing
        if avg_validation_loss < best_val_loss:
            best_val_loss = avg_validation_loss
            # Save original model checkpoint
            model_save_path = os.path.join(models_dir, args.run_name, f"ckpt.pt")
            torch.save(model.state_dict(), model_save_path)
            # Optionally save the optimizer state
            optimizer_save_path = os.path.join(models_dir, args.run_name, f"optim.pt")
            torch.save(optim.state_dict(), optimizer_save_path)
            
            
        ## Logging ##
        if current_iteration % args.display_iteration == 0:
            # Save original model checkpoint
            model_save_path = os.path.join(models_dir, args.run_name, f"ckpt_{current_iteration}.pt")
            torch.save(model.state_dict(), model_save_path)
            dynamic = scoremodel.AnnealedLangevinDynamic(args.sigma_min, args.sigma_max, args.n_steps, args.annealed_step, model, None, device, eps=args.eps)
            sample = dynamic.sampling(args.n_samples, only_final=True)
            for i in range(len(sample)):
                save_path = f'outputs/sample_{i}_step_{current_iteration}.png'
                utils.save_images(sample[i], save_path)
            





def _sample(args):
    
    # Load projection operator $P_{\matcal{C}}
    if args.experiment == 'microstructures':
        from examples.microstructures.projection import Projection
        # TODO: Move this logic to projection file
        pc_u = Projection(k=args.porosity+0.025, threshold=0.0, lower_bound=False)
        pc_l = Projection(k=args.porosity-0.025, threshold=0.0, lower_bound=True)
        projector = lambda x : pc_l.apply(pc_u.apply(x))
        
    elif args.experiment == 'trajectories':
        raise NotImplementedError('Support will be added in future releases.')
        from examples.trajectories.projection import Projection
        
    elif args.experiment == 'motion':
        from examples.motion.projection import Projection
        projector = lambda x : Projection(acceleration=args.gravity).apply(x)
        
    elif args.experiment == 'human':
        raise NotImplementedError('Support will be added in future releases.')
        
    else:
        module_path = f"examples.{args.projection_path}.projection"
        projection_module = __import__(module_path, fromlist=["Projection"])
        Projection = getattr(projection_module, "Projection")
        projector = lambda x : Projection().apply(x)
        



    # Load model
    model = Model(device, args.n_steps, args.sigma_min, args.sigma_max)
    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    dynamic = AnnealedLangevinDynamic(args.sigma_min,args.sigma_max, args.n_steps, args.annealed_step, model, projector, device, eps= args.eps)

    # Sampling function
    def generate_images(n_samples, only_final=True):
        sample = dynamic.sampling(n_samples, only_final)
        return sample
    create_outputs_directory()

    # Generate and save samples
    with torch.no_grad():
        sample = generate_images(args.n_samples)
        for i in range(len(sample)):
            save_path = f'outputs/sample_{i}.png'
            utils.save_images(sample[i], save_path)



def create_outputs_directory(path="outputs"):
    # Remove the directory if it exists
    if os.path.exists(path):
        shutil.rmtree(path)
    # Create a new empty directory
    os.makedirs(path)
    print(f"Directory '{path}' has been created.")



if __name__=='__main__':
    main()
