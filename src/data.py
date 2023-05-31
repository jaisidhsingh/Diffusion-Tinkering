from config import cfg
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image


class StanfordCarsDatset(Dataset):
	def __init__(self, root_dir, split="train", transform=None):
		self.root_dir = root_dir
		self.split = split
		self.is_test_img = {"train": "0", "test": "1"}
		self.transform = transform
		
		if split != "combined":
			self.image_paths = self.get_split(split)
		else:
			self.image_paths = self.get_split("train") + self.get_split("test")
	
	def get_split(self, split):
		image_paths = [
				os.path.join(
					self.root_dir, 
					self.is_test_img[split], 
					f
				) for f in os.listdir(os.path.join(
					self.root_dir, 
					self.is_test_img[split]
			))
		]
		return image_paths

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, idx):
		image_path = self.image_paths[idx]
		image = Image.open(image_path).convert('RGB')
		if self.transform is not None:
			image = self.transform(image)
		return image

def init_transforms():
	preprocess = transforms.Compose(
		[
			transforms.Resize((cfg.image_size, cfg.image_size)),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
		]
	)
	return preprocess

def dataset_mapping():
	mapping = {
		"stanford-cars": StanfordCarsDatset,
	}
	return mapping

def get_loader(dataset_name, split, with_dataset=False):
	data_transform = init_transforms()
	dataset_class = dataset_mapping()[dataset_name]

	dataset = dataset_class(cfg.datasets[dataset_name]["root_dir"], split, data_transform)
	dataloader = DataLoader(
		dataset, 
		batch_size=cfg.batch_size, 
		shuffle=cfg.shuffle, 
		pin_memory=cfg.pin_memory, 
		num_workers=cfg.num_workers
	)

	if with_dataset:
		return dataloader, dataset
	else:
		return dataloader	