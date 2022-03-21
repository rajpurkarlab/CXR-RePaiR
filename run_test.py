import numpy as np
import os
import pandas as pd
from PIL import Image
import h5py
from pathlib import Path
import argparse

import torch
from torch.utils import data
from tqdm import tqdm
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode

import clip
from model import CLIP

nonpretrained_params = {
    'embed_dim':768,
    'image_resolution': 320,
    'vision_layers': 12,
    'vision_width': 768,
    'vision_patch_size': 16,
    'context_length': 77,
    'vocab_size': 49408,
    'transformer_width': 512,
    'transformer_heads': 8,
    'transformer_layers': 12
}

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load in model
    if args.pretrained:
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
        print("Loaded in pretrained model.")
    else:
        model = CLIP(**nonpretrained_params)
        print("Loaded in clip model.")

    model.load_state_dict(torch.load(args.clip_model_path, map_location=device))
    model = model.to(device)

    # precalculated corpus clip embeddings
    corpus_embeddings_path = 'corpus_embeddings/' + args.corpus_embeddings_name
    raw_impressions, zeroshot_weights = get_zs_impressions(corpus_embeddings_path, args.compression)

    # load in test X-ray data
    if args.dataset == 'MIMIC-CXR':
        normalize_fn = Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944))
    else:
        normalize_fn = Normalize((129.4185, 129.4185, 129.4185), (73.3378, 73.3378, 73.3378))

    if args.pretrained:
        input_resolution = 224
        if args.dataset == 'MIMIC-CXR':
            transform = Compose([
                normalize_fn,
                Resize(input_resolution, interpolation=InterpolationMode.BICUBIC),
            ])
            print('Interpolation Mode: ', InterpolationMode.BICUBIC)
        else:
            transform = Compose([
                normalize_fn,
            ])
    else: 
        input_resolution = 320
        transform = Compose([
            normalize_fn,
        ])

    if args.dataset == 'MIMIC-CXR':
        dset = CXRTestDataset(img_path=args.test_cxr_path, transform=transform)
    else:
        dset = CheXpertDataset(img_path=args.test_cxr_path, transform=transform)
    loader = torch.utils.data.DataLoader(dset, shuffle=False)

    # select top report/sentences
    y_pred = predict(loader, zeroshot_weights, model, device)

    # save
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    out_path = args.out_dir + '/generated_reports.csv'
    save_reports(y_pred, raw_impressions, out_path, topk=args.topk)




def get_zs_impressions(weights_path, correlations_path):
    # Get the pre-generated zero_shot weights and corresponding impressions
    (raw_impressions, zeroshot_weights) = torch.load(weights_path)
    print(raw_impressions.shape, zeroshot_weights.shape)
    print(type(raw_impressions), type(zeroshot_weights))

    if correlations_path != "":
        reports_dict = torch.load(correlations_path)
        selected_reports = list(reports_dict.keys())
        raw_impressions = raw_impressions[selected_reports]
        zeroshot_weights = zeroshot_weights[selected_reports]
        print(raw_impressions.shape, zeroshot_weights.shape)
    raw_impressions.index = range(len(raw_impressions))
    return (raw_impressions, zeroshot_weights)

class CheXpertDataset(data.Dataset):
    def __init__(self, img_path, transform=None):
        super().__init__()
        all_images = []
        imgs_df = pd.read_csv(img_path)
        for _path in tqdm(imgs_df["Path"]):
            if "view1" not in _path: continue
            _pth = _path.replace("CheXpert-v1.0", "")
            _np_img = np.asarray(Image.open(Path(args.CheXpert_root_path+_pth)).resize((224, 224), resample=Image.BICUBIC)) # these images all have diff sizes
            _np_img = _np_img.astype('float32')
            all_images.append(_np_img)
        self.img_dset = np.array(all_images)               
        self.transform = transform
            
    def __len__(self):
        return len(self.img_dset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img = self.img_dset[idx]
        img = np.expand_dims(img, axis=0)
        img = np.repeat(img, 3, axis=0)
        img = torch.from_numpy(img)
        
        if self.transform:
            img = self.transform(img)

        sample = {'img': img}
        return sample

class CXRTestDataset(data.Dataset):
    def __init__(self, img_path, transform=None):
        super().__init__()
        self.img_dset = h5py.File(img_path, 'r')['cxr']
        self.transform = transform
        
    def __len__(self):
        return len(self.img_dset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img = self.img_dset[idx] # np array, (320, 320)
        img = np.expand_dims(img, axis=0)
        img = np.repeat(img, 3, axis=0)
        img = torch.from_numpy(img) # torch, (320, 320)
        
        if self.transform:
            img = self.transform(img)
        
        sample = {'img': img}
        return sample


def softmax(x):
    return np.exp(x)/sum(np.exp(x))

def predict(loader, zeroshot_weights, model, device, verbose=0,minibatch_size=100): 
    y_pred = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            images = data['img'].to(device)

            # predict
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            logits = image_features @ zeroshot_weights.T
            logits = logits.to('cpu')
            logits = np.squeeze(logits.numpy(), axis=0).astype('float64') # (14,)

            norm_logits = (logits - logits.mean()) / (logits.std())
            probs = softmax(norm_logits)
            y_pred.append(probs)
 
    y_pred = np.array(y_pred)
    return np.array(y_pred)

def save_reports(outputs, raw_impressions, out_path, topk=1):
    reports_list = []
    for output in outputs:
        # convert output to a report
        preds = np.argsort(output)[-topk:]
        preds = np.flip(preds)
        report = ""
        for pred in preds: report += raw_impressions[pred] + " "
        reports_list.append(report)
    # write reports to a csv
    _df = pd.DataFrame(reports_list)
    _df.columns = ["report"]
    _df.to_csv(out_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select the top report/sentences based on CXR-RePaiR method')
    parser.add_argument('--corpus_embeddings_name', type=str, required=True, help='name of corpus embeddings file generated by CLIP')
    parser.add_argument('--clip_model_path', type=str, required=True, help='name of clip model state dictionary')
    parser.add_argument('--out_dir', type=str, required=True, help='directory to save outputted generated reports')
    parser.add_argument('--test_cxr_path', type=str, required=True, help='path of test X-rays, .h5 file for MIMIC and .csv for CheXpert')
    parser.add_argument('--topk', type=int, required=False, default=1, help='number top sentences to retrieve')
    parser.add_argument('--sentences', type=bool, required=False, default=False, help='if true uses sentences')
    parser.add_argument('--compression', type=str, required=False, default='', help='set to file path for compression')
    parser.add_argument('--pretrained', type=bool, required=False, default=True, help='Whether or not to use clip model pre-trained on chest X-rays, should be same as used for generating corpus embeddings')
    parser.add_argument('--dataset', type=str, default='MIMIC-CXR', choices=['CheXpert', 'MIMIC-CXR'], help='dataset to evaluate on')
    parser.add_argument('--CheXpert_root_path', type=str, required=False, help='Root to folder where CheXpert test is')
    args = parser.parse_args()

    main(args)

