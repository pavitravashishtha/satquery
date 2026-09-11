import torch
import torch.nn as nn
import math
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class VisionModelWrapper(nn.Module):
    def __init__(self, vision_model):
        super().__init__()
        self.vision_model = vision_model

    def forward(self, *args, **kwargs):
        return self.vision_model(*args, **kwargs)

    @property
    def dtype(self):
        return self.vision_model.dtype

    @property
    def device(self):
        return self.vision_model.device

    @property
    def config(self):
        return self.vision_model.config


class CLIPVisionTower(nn.Module):
    def clip_interpolate_embeddings(self, image_size=600, patch_size= 14):
        """This function helps interpolating positional embeddings during checkpoint loading,
        especially when you want to apply a pre-trained model on images with different resolution.
 
        Args:
            image_size (int): Image size of the new model.
            patch_size (int): Patch size of the new model.
            model_state (OrderedDict[str, torch.Tensor]): State dict of the pre-trained model.
            interpolation_mode (str): The algorithm used for upsampling. Default: bicubic.
            reset_heads (bool): If true, not copying the state of heads. Default: False.
 
        Returns:
            OrderedDict[str, torch.Tensor]: A state dict which can be loaded into the new model.
        """
        # Shape of pos_embedding is (1, seq_length, hidden_dim)
        vision_model = getattr(self.vision_tower, 'vision_model', self.vision_tower)
        state_dict = vision_model.embeddings.position_embedding.state_dict()
        pos_embedding = state_dict['weight']
        pos_embedding = pos_embedding.unsqueeze(0)
        n, seq_length, hidden_dim = pos_embedding.shape
        if n != 1:
            raise ValueError(f"Unexpected position embedding shape: {pos_embedding.shape}")
 
        new_seq_length = (image_size // patch_size) ** 2 + 1
 
        # Need to interpolate the weights for the position embedding.
        # We do this by reshaping the positions embeddings to a 2d grid, performing
        # an interpolation in the (h, w) space and then reshaping back to a 1d grid.
        if new_seq_length != seq_length:
            # The class token embedding shouldn't be interpolated so we split it up.
            seq_length -= 1
            new_seq_length -= 1
            pos_embedding_token = pos_embedding[:, :1, :]
            pos_embedding_img = pos_embedding[:, 1:, :]
 
            # (1, seq_length, hidden_dim) -> (1, hidden_dim, seq_length)
            pos_embedding_img = pos_embedding_img.permute(0, 2, 1)
            seq_length_1d = int(math.sqrt(seq_length))
            torch._assert(seq_length_1d * seq_length_1d == seq_length, "seq_length is not a perfect square!")
 
            # (1, hidden_dim, seq_length) -> (1, hidden_dim, seq_l_1d, seq_l_1d)
            pos_embedding_img = pos_embedding_img.reshape(1, hidden_dim, seq_length_1d, seq_length_1d)
            new_seq_length_1d = image_size // patch_size
 
            # Perform interpolation.
            # (1, hidden_dim, seq_l_1d, seq_l_1d) -> (1, hidden_dim, new_seq_l_1d, new_seq_l_1d)
            new_pos_embedding_img = nn.functional.interpolate(
                pos_embedding_img,
                size=new_seq_length_1d,
                mode='bicubic',
                align_corners=True,
            )
 
            # (1, hidden_dim, new_seq_l_1d, new_seq_l_1d) -> (1, hidden_dim, new_seq_length)
            new_pos_embedding_img = new_pos_embedding_img.reshape(1, hidden_dim, new_seq_length)
 
            # (1, hidden_dim, new_seq_length) -> (1, new_seq_length, hidden_dim)
            new_pos_embedding_img = new_pos_embedding_img.permute(0, 2, 1)
            new_pos_embedding = torch.cat([pos_embedding_token, new_pos_embedding_img], dim=1)[0]
            state_dict['weight'] = new_pos_embedding
            vision_model.embeddings.position_embedding = nn.Embedding(new_seq_length+1, hidden_dim, _weight=new_pos_embedding)
            vision_model.embeddings.image_size = image_size
            vision_model.embeddings.patch_size = patch_size
            vision_model.embeddings.position_ids = torch.arange(new_seq_length+1, device=new_pos_embedding.device).expand((1, -1))
            
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.is_interpolated = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model(interpolate=False)
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, interpolate=True):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        if not hasattr(self, "cfg_only"):
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
        vm = CLIPVisionModel(self.cfg_only)
        self.vision_tower = VisionModelWrapper(vm)
        self.vision_tower.requires_grad_(False)
        if interpolate:
            self.clip_interpolate_embeddings(image_size=504, patch_size=14)
            self.is_interpolated = True

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        vision_model = getattr(self.vision_tower, 'vision_model', self.vision_tower)
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = vision_model(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = vision_model(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
