import os
import math
import torch

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


class LabelImage(Dataset):
	"""
	FDDB (http://vis-www.cs.umass.edu/fddb/)
	"""

	def __init__(self, data_dir, default_img_identifier=".jpg", transform=None, predict_mode=False):
		"""
		data directory
		- image directory
		 - image jpg
		 - annotation xml
		Args:
		"""
		self.transform = transform
		self.annotations = []
		self.predict_mode = predict_mode

		p = Path(data_dir)

		for xml in list(p.glob('**/*.xml')):
			xml = str(xml)
			img_name = xml.replace(".xml", default_img_identifier)
			tree = ET.parse(xml)
			root = tree.getroot()
			self.extrac_coordinate(root, img_name)

	def extrac_coordinate(self, root, img_name):
		for child in root:
			tmp_list = []
			if child.tag == "object":
				for object_coordinate in child:
					if object_coordinate.tag == "bndbox":
						coordinate_dict = {}
						tmp_list = [img_name]
						for each_coordinate in object_coordinate:
							coordinate_dict[each_coordinate.tag] = int(each_coordinate.text)
						for each_coordinate in coordinate_dict.values():
							tmp_list.append(each_coordinate)
					if len(tmp_list) > 1:
						self.annotations.append(tmp_list)

	def output_dims(self):
		return 4	# bbox left, top, right, bottom
			
	def __len__(self):
		return len(self.annotations)

	def __getitem__(self, idx):
		img_name, left, top, right, bottom = self.annotations[idx]
		img = load_image(img_name)
		if self.predict_mode:
			org_image = np.asarray(img)

		org_width = float(img.width)
		org_height = float(img.height)

		width = float(img.size[0])
		height = float(img.size[1])

		if self.transform is not None:
			img = self.transform(img)

		bbox_width = (right - left) / org_width
		bbox_height = (bottom - top) / org_height

		left = normalize(left, org_width, width)
		top = normalize(top, org_height, height)

		if self.predict_mode:
			return img, torch.Tensor([left, bbox_width, top, bbox_height]), org_image, torch.Tensor([org_width, org_height, width, height])
		else:
			return img, torch.Tensor([left, bbox_width, top, bbox_height])


def normalize(coord, original_dim, rescaled_dim):
	scale = rescaled_dim / original_dim
	coord = coord * scale
	return 2.0 * (coord / rescaled_dim - 0.5)	
		

def load_image(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
