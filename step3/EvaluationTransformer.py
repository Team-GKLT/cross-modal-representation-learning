import os
import cv2
import math
import gc
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
import albumentations as A
import matplotlib.pyplot as plt
import itertools
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import timm
import random
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer, ViTModel, ViTFeatureExtractor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

CKPT_PATH = '/common/home/gg676/536/notebooks/best_ilab2_epoch2_concatenated_ingr_620k.pt'
VALID_DATA = '/filer/tmp1/gg676/im2recipe/df_merged_VAL_appended_ingredients.pkl'#df_merged_VAL_ingredients.pkl'

df_val = pd.read_pickle(VALID_DATA)[:1000]
df_val.reset_index(inplace=True, drop=True)

df_val.info()

class CFG:
    debug = False
    image_path = "/filer/tmp1/gg676/im2recipe/img_data/train_flattened"
    val_path = "/filer/tmp1/gg676/im2recipe/img_data/val_flattened"
    #captions_path = "."
    batch_size = 48
    num_workers = 16
    head_lr = 1e-3
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    cross_attn_lr = 1e-4
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 3
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    model_name = 'vit_base_patch16_224'#'resnet50'
    image_embedding = 1000
    text_encoder_model = "distilbert-base-uncased"
    text_embedding = 768
    text_tokenizer = "distilbert-base-uncased"
    max_length = 200

    pretrained = True # for both image encoder and text encoder
    trainable = True # for both image encoder and text encoder
    temperature = 1.0

    # image size
    size = 224

    # for projection head; used for both image and text encoders
    num_projection_layers = 1
    projection_dim = 256
    dropout = 0.1


class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, ingredients, tokenizer, transforms):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names
        """

        self.image_filenames = image_filenames
        #self.tokenizer_need = tokenizer_need
        self.ingredients = [tokenizer(ingr,
                               padding='max_length', max_length = 128, truncation=True,
                                return_tensors="pt") for ingr in ingredients]
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {}
        image = cv2.imread(f"{CFG.val_path}/{self.image_filenames[idx]}.jpg")
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['ingredients'] = self.ingredients[idx]
        return item['image'], self.ingredients[idx]['input_ids'], self.ingredients[idx]['attention_mask']
        #

    def __len__(self):
        return len(self.ingredients)


def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )

class CLIPDatasetPreprocessed(torch.utils.data.Dataset):
    def __init__(self, image, input_ids, attn_mask, transforms):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names 
        """

        self.image = image
        #self.tokenizer_need = tokenizer_need
        #self.ingredients = [tokenizer(ingr, 
        #                       padding='max_length', max_length = 128, truncation=True,
        #                        return_tensors="pt") for ingr in ingredients]
        self.input_ids = input_ids
        self.attn_mask = attn_mask
        self.transforms = transforms

    def __getitem__(self, idx):
        #item = {}
        #image = cv2.imread(f"{CFG.val_path}/{self.image_filenames[idx]}.jpg")
        #image = self.transforms(image=image)['image']
        #item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        #item['ingredients'] = self.ingredients[idx]
        return self.image[idx], self.input_ids[idx], self.attn_mask[idx]
        #

    def __len__(self):
        return len(self.input_ids)


def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )



def make_train_valid_dfs(df, mode='train'):
    dataframe = df#pd.read_csv(f"{CFG.captions_path}/captions.csv")
    max_id = dataframe["id"].max() + 1 if not CFG.debug else 100
    image_ids = np.arange(0, max_id)
    np.random.seed(42)
    if mode == 'train':
        train_dataframe = dataframe#[dataframe["id"].isin(train_ids)].reset_index(drop=True)
        return train_dataframe
    else:
        valid_dataframe = dataframe#[dataframe["id"].isin(valid_ids)].reset_index(drop=True)
        return valid_dataframe
    #return train_dataframe, valid_dataframe


def build_loaders(dataframe, tokenizer, mode):
    transforms = get_transforms(mode=mode)
    dataset = CLIPDataset(
        dataframe["img_id"].values,
        dataframe["ingredients"].values,
        tokenizer=tokenizer,
        transforms=transforms,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader

def build_preprocessed_loaders(img, input_ids, attn_mask, mode):
    transforms = get_transforms(mode=mode)
    dataset = CLIPDatasetPreprocessed(
        img,
        input_ids,
        attn_mask,
        transforms=transforms,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader


class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x



class TextEncoder(nn.Module):
    def __init__(self, model_name=CFG.text_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained('/filer/tmp1/gg676/distilbert-base-uncased')
        else:
            self.model = DistilBertModel(config=DistilBertConfig())
            
        for p in self.model.parameters():
            p.requires_grad = False

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state #last_hidden_state[:, self.target_token_idx, :]


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class CrossAttention(nn.Module):
    def __init__(self, model_dim=768, n_heads=2, n_layers=2, num_image_patches=197, num_classes=1, dropout=0.1):
        super().__init__()
        self.text_positional = PositionalEncoding(model_dim, dropout=dropout)
        self.sep_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(layers, num_layers=n_layers)
        self.cls_projection = nn.Linear(model_dim, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, image_features, text_features, src_key_padding_mask=None):
        #print(image_features.shape)
        batch_size = image_features.shape[0]
        text_features = self.text_positional(text_features)
        sep_token = self.sep_token.expand(batch_size, -1, -1)
        transformer_input = torch.cat((image_features, sep_token, text_features), dim=1)
        if src_key_padding_mask is not None:
            src_key_padding_mask = torch.cat((torch.zeros(image_features.shape[0],
                                                          image_features.shape[1] + 1).to(CFG.device),
                                             src_key_padding_mask.to(CFG.device)), 1)

        transformer_outputs = self.encoder(transformer_input, src_key_padding_mask=src_key_padding_mask)
        projected_output = transformer_outputs[:, 0, :]
        return self.sigmoid(self.cls_projection(projected_output))



class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        text_embedding=CFG.text_embedding,
    ):
        super().__init__()
        self.image_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        #self.image_encoder = self.image_encoder.to(CFG.device)
        self.image_encoder.eval()
        self.text_encoder = TextEncoder()
        #self.text_encoder = DistilBertModel.from_pretrained('/filer/tmp1/gg676/distilbert-base-uncased')
        #self.text_encoder = self.text_encoder.to(CFG.device)
        self.text_encoder.eval()
        #classifier = classify(﻿128﻿,﻿100﻿,﻿17496﻿,﻿12﻿,﻿2﻿)
        
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature
        
        self.ntokens = 709  # size of vocabulary
        self.emsize = 768  # embedding dimension
        self.d_hid = 200  # dimension of the feedforward network model in nn.TransformerEncoder
        self.nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        self.nhead = 2  # number of heads in nn.MultiheadAttention
        self.dropout = 0.2  # dropout probability
        #self.transformer_model = TransformerModel(self.ntokens, self.emsize, self.nhead, self.d_hid, self.nlayers, self.dropout).to(CFG.device)
        self.cross_attn = CrossAttention()
        
        

    def forward(self, batch_img, input_ids, attn_mask):
        # Getting Image and Text Features
        with torch.no_grad():
            image_features = self.image_encoder(batch_img, output_hidden_states=True)
            #print(image_features)
            #print(input_ids.shape, attn_mask.shape)
            input_ids, attn_mask = input_ids.squeeze(1), attn_mask.squeeze(1)
            #print("image: ", image_features.hidden_states[-1].shape)
            #print("inp ids: ", input_ids.shape, "--> ", attn_mask.shape)
            text_features = self.text_encoder(
                input_ids=input_ids, attention_mask=attn_mask
        )
        #print(self.text_encoder(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True))
        #print("text: ", text_features.shape)
        #print("image shape: ", image_features.shape)
        # Getting Image and Text Embeddings (with same dimension)
        #image_embeddings = self.image_projection(image_features)
        #text_embeddings = self.text_projection(text_features)
        #img_feature, text_feature, attn_mask_list, labels = create_binary_label_data(image_features.hidden_states[-1], text_features, attn_mask)
        #loss, outputs = classifier.forward(concatenated_img_text_pairs, labels)
        #print("labels: ", labels)
        #print(concatenated_img_text_pairs[0])
        #print(img_feature)
        #print(img_feature[0].shape)
        
        
        #src_mask = generate_square_subsequent_mask(CFG.batch_size).to(CFG.device)
        
        output = self.cross_attn(img_feature, text_feature, attn_mask)
        #print(output.shape)
        #print(output)
        
        
        
        
        
        
        """
        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        """
        #print("inside: ", loss.shape)
        return output, labels.unsqueeze(1)#.to(CFG.device)


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()



def prep_image_textvalid_df(valid_df):
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")
    img_all_list = []
    input_id_list = []
    attn_mask_list = []
    text_all_list = []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            img, input_ids, attn_mask = batch
            input_ids, attn_mask = input_ids.squeeze(1), attn_mask.squeeze(1)
            #print(attn_mask)
            img_all_list.extend(img)
            input_id_list.extend(input_ids)
            attn_mask_list.extend(attn_mask)
    #print(len(attn_mask_list))
    return img_all_list, input_id_list, attn_mask_list

def create_all_comb_img_text(img_all_list, input_id_list, attn_mask_list):
    img_final_list = []
    input_id_final_list = []
    attn_mask_final_list = []
    count = 0
    for i in img_all_list:
        #print(i)
        for inp, attn in zip(input_id_list, attn_mask_list):
            img_final_list.append(i)
            input_id_final_list.append(inp)
            attn_mask_final_list.append(attn)
        if count % 100 == 0:
            print(count)
        count += 1
    #print(img_final_list[0])
    return img_final_list,input_id_final_list, attn_mask_final_list

img_final_list,input_id_final_list, attn_mask_list = create_all_comb_img_text(img_all_list, input_id_list, attn_mask_list)

def save_artifacts(data, file_name):
    with open('/filer/tmp1/gg676/'+file_name+'.pkl', 'wb') as fp:
        pickle.dump(data, fp)
#save_artifacts(img_final_list, "img_final_list_val")
save_artifacts(input_id_final_list, "input_id_final_list_val")
save_artifacts(attn_mask_list, "attn_mask_list_val")
print("saved files")


def get_image_text_features(valid_df):
    #tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    valid_loader = build_preprocessed_loaders(img_final_list,input_id_final_list, attn_mask_list,  mode="valid")
    
    model = CLIPModel().to(CFG.device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=CFG.device))
    model.eval()
    model.image_encoder.eval()
    model.text_encoder.eval()
    model.cross_attn.eval()
    output_list = []
    #img_id_list = []
    img_feature_list = []
    text_feature_list = []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            img, input_ids, attn_mask = batch
            #img_id_list.extend(img_id)
            input_ids, attn_mask = input_ids.squeeze(1), attn_mask.squeeze(1)
            image_features = model.image_encoder(img.to(CFG.device), output_hidden_states=True)
            #print("image features shape: ", image_features.hidden_states[-1].shape)
            text_features = model.text_encoder(input_ids=input_ids.to(CFG.device), attention_mask=attn_mask.to(CFG.device))
            output = model.cross_attn(image_features.hidden_states[-1], text_features, attn_mask)
            output_list.extend(output.detach().cpu().numpy())
            img_feature_list.extend(image_features.hidden_states[-1].detach().cpu().numpy())
            text_feature_list.extend(text_features.detach().cpu().numpy())
            #image_embeddings = model.image_projection(image_features)
            #valid_image_featur.append(image_features)
    return output_list, img_feature_list, text_feature_list


#valid_df = make_train_valid_dfs(mode='valid')
output_list, img_feature_list, text_feature_list = get_image_text_features(df_val)




save_artifacts(output_list,, "output_final_list_val")
save_artifacts(img_feature_list, "img_final_list_val")
save_artifacts(text_feature_list, "text_final_list_val")









