import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset


#######################################################################
# DEPRECATED CLASS
#######################################################################


def download_cifar100_subset(root_folder: str = "./data"):
    """
    Downloads CIFAR-100 and generates the CIFAR-100 subset of 4 super-classes.
    """
    transform = transforms.Compose(
        # [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        [transforms.ToTensor()]
    )

    # Load entire CIFAR-100 dataset
    full_dataset = torchvision.datasets.CIFAR100(
        root=root_folder, train=True, download=True, transform=transform
    )

    # Define class indices for the selected superclasses
    selected_classes = {
        "aquatic_mammals": [4, 30, 55, 72, 95],  # beaver, dolphin, otter, seal, whale
        "fish": [1, 32, 67, 73, 91],  # aquarium fish, flatfish, ray, shark, trout
        "vehicles_1": [
            8,  # bicycle
            13,  # bus
            48,  # motorcycle
            58,  # pickup truck
            90,  # train
        ],
        "vehicles_2": [
            41,  # lawn-mower
            70,  # rocket
            82,  # streetcar
            92,  # tank
            93,  # tractor
        ],
    }

    # Create a list of indices for the selected images
    indices = [
        idx
        for idx, item in enumerate(full_dataset.targets)
        if item in sum(selected_classes.values(), [])
    ]

    # Create a subset from the full dataset
    subset_dataset = Subset(full_dataset, indices)

    # Save the subset dataset
    torch.save(subset_dataset, f"{root_folder}/cifar100_subset.pth")


if __name__ == "__main__":
    print("Starting the download...")
    download_cifar100_subset()
    print("Finished.")