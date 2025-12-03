import torch

def get_mean_std(loader):
    # Varieabili accumulatrici
    batches_sum, channels_squared_sum, num_batches = 0, 0, 0
    
    for data, _ in loader:
        # data shape: [batch_size, channels, height, width]
        batches_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3]) #Per calcolare la devizione standard usiamo la formula di Konig-Huygens: varianza = media(x^2) - media(x)^2
        num_batches += 1
    
    mean = batches_sum / num_batches #la proprietà che sfruttiamo qui è quella per cui la media delle medie è la media totale

    std = ((channels_squared_sum / num_batches) - mean ** 2) ** 0.5
    
    return mean, std