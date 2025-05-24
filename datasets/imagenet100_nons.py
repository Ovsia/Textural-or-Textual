import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from .utils.make_dataset_train import make_image_text1, make_image_text2, make_image_text3


class ImageNet100Nons(Dataset):
    def __init__(self, root, split='train', preprocess=None):

        self._data_dir = Path(root) / split
        self._preprocessed_dir = Path(root) / split
        self._typographic_dir = Path(root) / 'typo_large' / split
        self._origin_dir = Path(root) / 'origin' / split
        self._nonsense_dir = Path(root) / 'not_corr'/ 'nonsense' / split
        self._typographic_dir_small = Path(root) / 'typo_small' / split

        self.transform = preprocess

        with open(Path(root) / 'Labels_train.json') as f:
            id_to_class = json.load(f)

        self.id_to_class = id_to_class
        classes = []
        for dir in self._data_dir.iterdir():
            if not dir.is_dir():
                continue
            id = str(dir).split('/')[-1]
            class_i = id_to_class[id]
            classes.append(class_i)

        self.classes = classes
        self.class_to_idx = dict(zip(classes, range(len(classes))))
        self.idx_to_class = {idx: class_name for class_name, idx in self.class_to_idx.items()}

        self._labels = []

        files = list(self._data_dir.rglob('*'))
        self._files = []
        for i in range(len(files)):
            if not files[i].is_file():
                continue
            self._files.append(files[i])

        for file in self._files:
            id = str(file).split('/')[-2]
            class_i = id_to_class[id].split(',')[0]
            self._labels.append(self.class_to_idx[class_i])

        self._samples = []
        for file in self._files:
            nonsense_path = self._nonsense_dir / file.relative_to(self._data_dir)
            self._samples.append(nonsense_path)


    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx):

        image = self._samples[idx]
        label = self._labels[idx]
        image = Image.open(image)
        catogery = self.idx_to_class[self._labels[idx]]
        id = "_".join([str(self._files[idx]).split('/')[-2], str(self._files[idx]).split('/')[-1]])
        if self.transform is not None:
            image = self.transform(image)
            image_description = f"A photo of {catogery}."
        return image, label, catogery, image_description,id

    def _check_exists_synthesized_dataset(self) -> bool:
        return self._typographic_dir.is_dir()
