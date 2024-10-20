import torch
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp

def train(rank, world_size):
    # Initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    # Define a simple model and wrap it with DistributedDataParallel
    model = nn.Linear(100, 10).to(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    
    # Example input data for the specific rank
    data = torch.randn(64, 100).to(rank)
    
    # Forward pass
    output = model(data)
    
    # Clean up
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
