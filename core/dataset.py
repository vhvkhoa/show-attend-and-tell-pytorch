from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import json
import os

class CocoImageDataset(Dataset):
    def __init__(self, root, image_paths):
        self.root = root
        self.image_paths = image_paths 
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transformation = transforms.Compose([transforms.Resize((224,224)),
                                                    transforms.ToTensor(), normalize])
    
    def __getitem__(self, index):
        path = self.image_paths[index]
        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transformation is not None:
            image = self.transformation(image)
        return path, image

    def __len__(self, ):
        return len(self.image_paths)

'''CocoCaptionDataset:
        Use: Cover processed captions and their corresponding features.'''
class CocoCaptionDataset(Dataset):
    def __init__(self, caption_file, split='train'):
        self.split = split
        with open(caption_file, 'r') as f:
            dataset = json.load(f)
        self.annotations = dataset['annotations']
        if split == 'train':
            self.word_to_idx = json.load(open('data/word_to_idx.json', 'r'))
    
    def __getitem__(self, index):
        annotation = self.annotations[index]
        caption = annotation['caption']
        cap_vec = annotation['vector']
        feature_path = os.path.join('data', self.split, 'feats', annotation['file_name'] + '.npy')
        feature = np.load(feature_path)
        return feature, cap_vec, caption

    def __len__(self, ):
        return len(self.annotations)

    def get_vocab_dict(self):
        return self.word_to_idx