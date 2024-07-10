from torchvision import datasets
from torch.utils.data import Subset


class CIFARN(datasets.CIFAR100):
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        selected_classes_names: list = None,
    ):
        if selected_classes_names is None or len(selected_classes_names) == 0:
            raise ValueError("selected_classes_names must have content.")

        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        selected_classes = {
            k: v for (k, v) in self.class_to_idx.items() if k in selected_classes_names
        }  # {'bicycle': 8, 'dolphin': 30, 'motorcycle': 48, 'ray': 67, 'shark': 73, 'tank': 85, 'tractor': 89, 'trout': 91}
        self.class_to_idx = {
            k: selected_classes_names.index(k)
            for (k, v) in self.class_to_idx.items()
            if k in selected_classes_names
        }  # {'bicycle': 4, 'dolphin': 3, 'motorcycle': 5, 'ray': 0, 'shark': 2, 'tank': 6, 'tractor': 7, 'trout': 1}
        self.original_class_mapping = {
            selected_classes[k]: self.class_to_idx[k] for k in selected_classes.keys()
        }  # {8: 4, 30: 3, 48: 5 ...}

        # Filter and remap classes
        mask = [target in self.original_class_mapping for target in self.targets]
        self.data = self.data[mask]
        self.targets = [
            self.original_class_mapping[target]
            for target in self.targets
            if target in self.original_class_mapping
        ]


class CIFAR8(CIFARN):
    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=False
    ):
        self.selected_classes_names = [
            "ray",
            "trout",
            "shark",
            "dolphin",
            "bicycle",
            "motorcycle",
            "tank",
            "tractor",
        ]

        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
            selected_classes_names=self.selected_classes_names,
        )

    def get_subset(self, labels: list[str] | list[int]) -> Subset:
        """
        Returns a torch.utils.data.Subset containing only the filtered data that match the labels passed by the argument.

        Args:
            labels (list[str] | list[int]): list of label names or label IDs.

        Returns:
            Subset: Torch Subset containing the filtered data.
        """
        idx_labels = [
            self.class_to_idx[lbl] if isinstance(lbl, str) else lbl for lbl in labels
        ]

        filtered_indices = [
            idx for idx, lbl in enumerate(self.targets) if lbl in idx_labels
        ]
        return Subset(self, filtered_indices)