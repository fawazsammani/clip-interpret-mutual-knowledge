import torch
import numpy as np
import cv2
import random
import clip
import os
from torchvision.ops import masks_to_boxes
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim
from PIL import Image
import torch.utils.data
from sklearn.decomposition import PCA
from skimage.measure import label as measure_label
import denseCRF
from kmeans_pytorch import kmeans
from sklearn.preprocessing import MinMaxScaler

def sinkhorn_knopp(C, epsilon, iters=1000):
    """
    Sinkhorn-Knopp algorithm for optimal transport.
    Parameters:
        - C: Cost matrix (torch.Tensor) of shape (m,n). Since it is a simialirty matrix, we need to invert it C = -C (already changed the formulation directly for simplicity)
        - epsilon: entropy regularization parameter: epsilon = (1 / lambda). Higher values leads to more confident / sharp results (less evenly spread)
          e.g. a single point in the source distribution is mapped to a single point in the target distribution with an infinite amount of mass.
          These solutions are often impractical and do not provide meaningful insights. Smaller values lead to the opposite effect.
        - max_iters: Number of iterations

    Returns:
        - P: Optimal transport matrix of shape (m,n) (torch.Tensor)
    """
    # Initialize the transport matrix P
    P = torch.exp(C * epsilon)

    for _ in range(iters):
        # Row normalization
        P /= P.sum(dim=1, keepdim=True)

        # Column normalization
        P /= P.sum(dim=0, keepdim=True)

    return P


def get_label_colors():
    # base colors
    label_colors = {
            0: [255, 0, 0], 
            1: [0, 255, 0],     
            2: [0, 0, 255],     
            3: [255, 255, 0],   
            4: [255, 165, 0],
            5: [255, 192, 203],
            6: [160, 32, 240],
            7: [0, 255, 255], 
            8: [128, 0, 0],
            9: [128, 128, 0],
            10: [128, 0, 128],
            11: [255, 105, 180],
            12: [75, 0, 130],
            13: [0, 128, 0],
            14: [0, 128, 128],
            15: [70, 130, 180],
            16: [255, 69, 0],
            17: [139, 69, 19],
            18: [0, 0, 128], 
            19: [255, 20, 147], 
            20: [255, 140, 0]}
    
    # add 30 other random colors
#     start_from = list(label_colors.keys())[-1] + 1
#     for c in range(start_from, 100):
#         label_colors[c] = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]
    
    return label_colors

def get_clip_model(model_name, device, img_size):
    clip_tower, _ = clip.load(model_name, device=device) # loads already in eval mode   # clip.available_models()
    model = clip_tower.visual
    patch_size = model.conv1.kernel_size[0]
    num_patches = img_size // patch_size
    return clip_tower, model, (patch_size, num_patches)

def get_random_cls_images(path):
    classes_folders = os.listdir(path)
    random_cls_index = random.randint(0, len(classes_folders) - 1)
    cls_folder = classes_folders[random_cls_index]
    img_folder = path + cls_folder
    cls_images = os.listdir(img_folder)
    img_path = [img_folder + "/" + img_name for img_name in cls_images]
    return img_path


class Extractor:

    def __init__(self, model, device):
        self.activations = []
        self.model = model
        self.dtype = model.conv1.weight.dtype  
        self.device = device
        
    def get_hook(self, weights):

        def inner_hook(module, input, output):
            inp = input[1].transpose(1,0)  # all inputs to attn are same, does not matter. Can also do input[0] or input[2] 
            output_qkv = torch.matmul(inp, weights.transpose(0,1)).detach()  # (B, 197, dim * 3)
            B, T, _ = output_qkv.shape
            num_heads = self.model.transformer.resblocks[-1].attn.num_heads
            output_qkv = output_qkv.reshape(B, T, 3, num_heads, -1 // num_heads).permute(2, 0, 3, 1, 4)
            k_feats = output_qkv[1].transpose(1, 2).reshape(B, T, -1)
            self.activations.append(k_feats)
        
        return inner_hook
    
    def save_activation(self, module, input, output):
        self.activations.append(output.detach())
        
    def clear(self):
        self.activations = []
        
    def extract(self, feat_type, input_imgs):
        
        input_imgs = input_imgs.to(self.device).type(self.dtype) 
        
        if feat_type == 'last_layer':
            handlers = [self.model.transformer.register_forward_hook(self.save_activation)]
        else:      # keys
            handlers = [self.model.transformer.resblocks[-1].attn.register_forward_hook(self.get_hook(self.model.transformer.resblocks[-1].attn.in_proj_weight))]
        
        with torch.no_grad():
            _ = self.model(input_imgs) 
            
        feats = self.activations[0]
        if feat_type == 'last_layer':
            feats = feats.permute(1, 0, 2)
            # add self.model.ln_post() for post-norm features
            
        for h in handlers:
            h.remove()
            
        return feats.float()
    
    def flatten_feats(self, features):
        dim = features.shape[-1]
        flat_features = features.reshape(-1, dim)     # (NxT)xC
        return flat_features
     
def show_image(x, squeeze = True, denormalize = True):
    
    if squeeze:
        x = x.squeeze(0)
        
    x = x.cpu().numpy().transpose((1, 2, 0))
    
    if denormalize:
        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        std = np.array([0.26862954, 0.26130258, 0.27577711])
        x = std * x + mean 
    
    return x.clip(0, 1)

    
def get_val_transform(img_size, crop = False):
    
    normalize = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                         std= [0.26862954, 0.26130258, 0.27577711])])
    
    if crop:   # original implementation of clip uses Resize followed by CenterCrop
        trs = transforms.Compose([transforms.Resize(img_size),
                                  transforms.CenterCrop(img_size),
                                  normalize])
    else:
        trs = transforms.Compose([transforms.Resize((img_size, img_size)),
                                  normalize])
    return trs


def get_largest_cc(mask, first_largest_indices = None):
    mask[mask!=0] = 1   # expects binary
    if first_largest_indices is not None:  # get the second largest component
        mask[first_largest_indices] = 0
    labels = measure_label(mask)  # get connected components
    if labels.sum() == 0: 
        return np.array([1])
    largest_cc_index = np.argmax(np.bincount(labels.flat)[1:]) + 1
    largest_cc_mask = (labels == largest_cc_index)
    return largest_cc_mask

def get_foreground_background_indices(pca_features, num_patches, number_of_images, initial_threshold, 
                                      foreground_component, patch_h, patch_w):
    
    # split_indices = [num_patches * i for i in range(1, number_of_images + 1)]
    
    initial_bg_indices = pca_features[:, foreground_component] < initial_threshold # on all features

    foreground = np.copy(pca_features)
    foreground[initial_bg_indices] = 0
    
    all_fg_indices = []
    all_bg_indices = []
    
    start = 0
    
    for i in range(1, number_of_images + 1):
        
        foreground_i = foreground[start : num_patches * i][:,foreground_component].reshape(patch_h, patch_w)

        # largest cc
        fg_indices = get_largest_cc(foreground_i)
        fg_indices_2 = get_largest_cc(foreground_i, fg_indices)

        # combined
        if fg_indices_2.sum() >=5:
            fg_indices = np.logical_or(fg_indices, fg_indices_2)

        bg_indices = ~fg_indices
        
        start = num_patches * i
        
        all_fg_indices.append(fg_indices)
        all_bg_indices.append(bg_indices)
        
    return all_fg_indices, all_bg_indices

def get_foreground_crf_map(upscaled_mask, image, img_size, DEFAULT_CRF_PARAMS):
    
    """
    foreground: foreground binary np array at original resolution (e.g., 14x14). The output of the largest connected component.
    image: PyTorch Tensor of the transformed image
    img_size: the image size to interpolate the foreground map to 
    """
    #resized_foreground = (np.array(foreground_reshaped_largestcc, dtype = np.float32) * 255).astype(np.uint8)
    image = show_image(image)
    img = (image * 255).astype(np.uint8)
    unary_potentials = torch.from_numpy(upscaled_mask).long()   # what if resize does not give 0 or 1?
    unary_potentials = F.one_hot(unary_potentials, num_classes = 2).float().numpy()
    out = denseCRF.densecrf(img, unary_potentials, DEFAULT_CRF_PARAMS)
    
    crf_mask = np.array((out!=0), dtype = np.float32)  # assumes 0 is background (hopefully always the case!)
    
    # visualize the mask
    seg_mask = np.zeros((out.shape[0], out.shape[1], 3), dtype=np.uint8)
    for label, color in get_label_colors().items():
        seg_mask[out == label] = color
        seg_mask[out == 0] = 0
        if label == out.max():
            break

    seg_mask = cv2.resize(seg_mask, dsize=(img_size, img_size), interpolation=cv2.INTER_NEAREST)   
    result = cv2.addWeighted(img, 0.5, seg_mask, 0.5, 0)
    
    return result, crf_mask

def upscale_foreground_masks(all_fg_indices, img_size):
    upscaled_masks = []

    for i in range(len(all_fg_indices)):
        upscaled_mask = np.array(all_fg_indices[i], dtype = np.float32) # convert binary to float
        upscaled_mask = cv2.resize(upscaled_mask, dsize=(img_size, img_size), interpolation=cv2.INTER_NEAREST) # resize
        upscaled_masks.append(upscaled_mask)
        
    upscaled_masks = np.stack(upscaled_masks).astype(np.uint8)
    return upscaled_masks


def get_bounding_boxes(masks):
    """
    masks: np array of shape (batch_size, img_size, img_size) representing binary masks of foreground objects, of type uint8
    """
    masks = torch.from_numpy(masks)  
    boxes = masks_to_boxes(masks)
    boxes = boxes.numpy().astype(np.int32)
    return boxes

def draw_bounding_box(img_path, box):
    cv2_img = cv2.imread(img_path)
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    cv2_img = cv2.resize(cv2_img, (224,224))
    cv2.rectangle(cv2_img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    return cv2_img

def get_pca_features(K, foreground_method, all_flat_features, all_flat_features_pt, threshold_by_negative):
    
    if foreground_method == 'pca':
        pca = PCA(n_components=K)
        pca.fit(all_flat_features)
        pca_features = pca.transform(all_flat_features)
        foreground_component = 0  # this has to be manually verified for each type of features!
        
    elif foreground_method == 'svd':
        all_flat_features_pt = F.normalize(all_flat_features_pt, p=2, dim=-1)
        USV = torch.linalg.svd(all_flat_features_pt, full_matrices=False)
        pca_features = USV[0][:, :K].to('cpu', non_blocking=True)
        pca_features = pca_features.numpy()
        foreground_component = 1   # this has to be manually verified for each type of features!
    
    # check that this works always regardless of the foreground method (PCA or SVD)
    for k in range(pca_features.shape[1]):
        if 0.5 < np.mean(pca_features[:,k] > 0) < 1.0:  # reverse segment
            pca_features[:,k] = 0 - pca_features[:,k]
    
    if not threshold_by_negative:
        scaler = MinMaxScaler(clip=True)
        scaler.fit(pca_features)
        pca_features = scaler.transform(pca_features)
        
    return pca_features, foreground_component

def color_segments(segments, img, background, crf_mask):
    image_to_show = show_image(img)
    patch_h, patch_w = segments.shape[0], segments.shape[1]
    seg_mask = np.zeros((patch_h, patch_w, 3), dtype=np.uint8)   # (14, 14, 3)
    background = np.tile(background.reshape(patch_h, patch_w)[:,:,None], (1,1,3))

    for label, color in get_label_colors().items():
        seg_mask[segments == label] = color
        # in the case of using argmax, it will return 0 when all componenets are 0, but zero is then the same for the segment 
        # identified and the background! so we need to set the background again to zero. 
        seg_mask[background] = 0   
        if label == segments.max():
            break
            
    img_size = image_to_show.shape[0]
    seg_mask_nst = cv2.resize(seg_mask, (img_size, img_size), interpolation = cv2.INTER_NEAREST)   
    seg_mask_viz = cv2.resize(seg_mask, (img_size, img_size))
    
    if crf_mask is not None:
        seg_mask_nst[crf_mask == 0] = 0  
        seg_mask_viz[crf_mask == 0] = 0  
        
    return seg_mask_nst, seg_mask_viz

def get_component_feats(feats, K_parts):
    pca = PCA(n_components = K_parts)
    pca.fit(feats)
    pca_features_rem = pca.transform(feats)
    # Min Max Normalization
    scaler = MinMaxScaler(clip=True)
    scaler.fit(pca_features_rem)
    pca_features_rem = scaler.transform(pca_features_rem)
    return pca_features_rem

def get_pca_part_segments(K_parts, pca_features_rem, all_fg_indices, crf_masks, images):
    patch_h, patch_w = all_fg_indices[0].shape
    split_idx_parts = np.cumsum([f.sum() for f in all_fg_indices])[:-1]
    parts = np.split(pca_features_rem, split_idx_parts)
    assert len(all_fg_indices) == len(parts)
    colored_segments = []
    colored_segments_viz = []
    for i in range(len(parts)):
        part = np.zeros((patch_h * patch_w, K_parts))
        foreground = all_fg_indices[i].reshape(-1)
        background = ~foreground
        part[background] = 0
        part[foreground] = parts[i]
        part = part.reshape(patch_h, patch_w, K_parts)
        segments = part.argmax(axis = -1)   
        seg_mask, seg_mask_viz = color_segments(segments, images[i], background, crf_masks[i])  
        colored_segments.append(seg_mask)
        colored_segments_viz.append(seg_mask_viz)
        
    return colored_segments, colored_segments_viz

def kmeans_clustering(K_parts, feats, all_fg_indices_flat, device):
    all_fg_feats_normalized = F.normalize(feats, p=2, dim=-1)
    cluster_ids, cluster_centers = kmeans(X = all_fg_feats_normalized, num_clusters=K_parts, distance='euclidean', device=device, tqdm_flag = False, iter_limit = 500) 
    # distances to the nearest cluster
    # distances = torch.sum(torch.pow(all_fg_feats_normalized - cluster_centers[cluster_ids], 2), dim = -1)
    segments = torch.zeros(all_fg_indices_flat.shape[0]).long()
    segments[all_fg_indices_flat] = cluster_ids
    return segments

def get_kmeans_part_segments(kmeans_segments, all_fg_indices, crf_masks, images):
    patch_h, patch_w = all_fg_indices[0].shape[0], all_fg_indices[1].shape[1]
    num_patches = patch_h  * patch_w
    number_of_images = len(images)
    split_idx_kmeans = np.cumsum([num_patches for _ in range(number_of_images)])[:-1]
    all_segments = np.split(kmeans_segments, split_idx_kmeans)
    all_segments = [p.reshape(patch_h, patch_w) for p in all_segments]
    colored_segments = []
    colored_segments_viz = []
    for i in range(len(all_segments)):
        foreground = all_fg_indices[i].reshape(-1)
        background = ~foreground
        seg_mask, seg_mask_viz = color_segments(all_segments[i], images[i], background, crf_masks[i])  
        colored_segments.append(seg_mask)
        colored_segments_viz.append(seg_mask_viz)

    return colored_segments, colored_segments_viz

def overlay_segments_on_image(img, seg):
    img = (show_image(img) * 255).astype(np.uint8)
    overlaid = cv2.addWeighted(img, 0.5, seg, 0.5, 0)
    return overlaid

def draw_circle_on_components_with_augments(img, colored_segments, circle_thickness = 1):
    img = np.ascontiguousarray((show_image(img) * 255), dtype=np.uint8)
    unique_colors = np.unique(colored_segments.reshape(-1, 3), axis=0)
    red_circle, red_circle_gray, red_circle_blur = [], [], []
    for color in unique_colors:
        if color.sum() != 0:   # skip background
            binary_image = np.all(colored_segments == color, axis=-1)
            binary_image = binary_image.astype(np.uint8)
            contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  
            largest_contour = max(contours, key=cv2.contourArea)
            
            if len(largest_contour) < 5:  # should not happen since we set CHAIN_APPROX_NONE (but just to be safe)
                num_missing = 5 - len(largest_contour)
                # random corner values
                randm = [np.array([[random.randint(0, largest_contour[-1][0][0] - 1), 
                                    random.randint(0, largest_contour[-1][0][1] - 1)]], 
                                  dtype=np.int32)[None,:,:] for _ in range(num_missing)]
            
                randm = np.concatenate(randm)
                largest_contour = np.concatenate((largest_contour, randm))
            
            ellipse_params = cv2.fitEllipse(largest_contour)
            
            # red circle
            img_red_circle = np.copy(img)
            cv2.ellipse(img_red_circle, ellipse_params, color = (255,0,0), thickness = circle_thickness)
            red_circle.append(img_red_circle)
            
            # red circle with grayscale and blur
            mask = np.zeros_like(img)
            cv2.ellipse(mask, ellipse_params, color = (255,255,255), thickness = -1)
            
            # grayscale
            grayscale_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            grayscale_img = np.tile(grayscale_img[:,:,None], (1,1,3))
            img_red_circle_gray = np.where(mask != 0, img, grayscale_img)
            cv2.ellipse(img_red_circle_gray, ellipse_params, color = (255,0,0), thickness = circle_thickness)
            red_circle_gray.append(img_red_circle_gray)
            
            # blur
            blurred_img = cv2.GaussianBlur(img, (11, 11), 0)
            img_red_circle_blur = np.where(mask != 0, img, blurred_img)
            cv2.ellipse(img_red_circle_blur, ellipse_params, color = (255,0,0), thickness = circle_thickness)
            red_circle_blur.append(img_red_circle_blur)
            
    return (red_circle, red_circle_gray, red_circle_blur, unique_colors)

def remove_duplicates(lst):
    seen = []
    for d in lst:
        if d not in seen:
            seen.append(d)
    return seen

def encode_text(model, text, device):
    text = clip.tokenize(['{}'.format(p) for p in text]).to(device)
    x = model.token_embedding(text).type(model.dtype)  # [batch_size, n_ctx, d_model]
    x = x + model.positional_embedding.type(model.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = model.ln_final(x).type(model.dtype)

    # x.shape = [batch_size, n_ctx, transformer.width]
    # take features from the eot embedding (eot_token is the highest number in each sequence)
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] 
    return x

def get_concept_relevance_with_image(clip_tower, concepts, img, dtype, device):
    image_features = img.unsqueeze(0).type(dtype)
    image_features = clip_tower.encode_image(image_features).float()
    image_features /= image_features.norm(dim=-1, keepdim=True)      # the image

    with torch.no_grad():
        text_features = clip.tokenize(['{}'.format(p) for p in concepts]).to(device)
        text_features = clip_tower.encode_text(text_features).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)  

    concept_to_image = text_features @ image_features.T
    return concept_to_image

def get_corresponding_textual_concepts(img, cirlce_images, transform, clip_tower, concept_classifier, all_descriptors, device, 
                                       threshold_by_relevance = False, topk = 3, use_ot = True, lambda_sinkhorn = 1, 
                                       mean_offset = 0.01, op_initial_topk = 100):
    
    """returns the textual explanations associated to each visual concept, in the same order as unique_colors"""
    
    red_circle, red_circle_gray, red_circle_blur =  cirlce_images
    red_circles = torch.stack([transform(Image.fromarray(rc)) for rc in red_circle], dim = 0)
    red_circles_gray = torch.stack([transform(Image.fromarray(rc)) for rc in red_circle_gray], dim = 0)
    red_circles_blur = torch.stack([transform(Image.fromarray(rc)) for rc in red_circle_blur], dim = 0)
    image_features = torch.stack([red_circles, red_circles_gray, red_circles_blur], dim = 0)
    
    num_aug, num_com = image_features.shape[:2]
    img_size = image_features.shape[-1]
    
    image_features = image_features.reshape(num_aug * num_com, 3, img_size, img_size)
    with torch.no_grad():
        image_features = image_features.to(device)
        image_features = clip_tower.encode_image(image_features)
        image_features = image_features.float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.reshape(num_aug, num_com, -1).mean(0)
        textual_concepts_logits = concept_classifier(image_features) 
        
        if use_ot:
            topk_values, topk_indices = torch.topk(textual_concepts_logits, op_initial_topk, dim=1)
            common_indices = set(topk_indices[0].tolist())
            for i in range(1, textual_concepts_logits.shape[0]):
                common_indices.intersection_update(topk_indices[i].tolist())
    
            common_indices = torch.tensor(list(common_indices)).to(device) # indices from the original logits
            common_concepts = [all_descriptors[v.item()] for v in common_indices]
            
            if threshold_by_relevance:
                relevance = get_concept_relevance_with_image(clip_tower, common_concepts, img, clip_tower.dtype, device).squeeze()
                relevance_threshold = (relevance.mean() - mean_offset).item()
                common_indices = common_indices[relevance > relevance_threshold]
                common_concepts = [all_descriptors[v.item()] for v in common_indices]
            
            common_logits = []
            for i in range(textual_concepts_logits.shape[0]):
                common_logits.append(textual_concepts_logits[i, common_indices]) # these will always be in the order of the common_indices
    
            common_logits = torch.stack(common_logits, dim = 0)
            
            common_logits = common_logits * 2.5
            common_logits = sinkhorn_knopp(common_logits, epsilon = 1/lambda_sinkhorn)
            
            _, textual_concepts_indices = common_logits.topk(topk, dim = -1)
            
            textual_concepts = []
            for i in range(textual_concepts_indices.shape[0]):
                descs = [common_concepts[v.item()] for v in textual_concepts_indices[i]]
                textual_concepts.append(descs)
                
            textual_concepts = [remove_duplicates(tc) for tc in textual_concepts]
            
        else:
            _, textual_concepts_indices = textual_concepts_logits.topk(topk, dim = -1)
            
            textual_concepts = []
            for i in range(textual_concepts_indices.shape[0]):
                descs = [all_descriptors[v.item()] for v in textual_concepts_indices[i]]
                textual_concepts.append(descs)
    
            textual_concepts = [remove_duplicates(tc) for tc in textual_concepts]
            
    return textual_concepts

def get_relevance_score(textual_concepts, img, clip_tower, device):
    predicted = [v for c in textual_concepts for v in c]
    concept_to_image = get_concept_relevance_with_image(clip_tower, predicted, img, clip_tower.dtype, device).squeeze()
    concept_to_image = concept_to_image * 2.5

    # remove duplicates
    pred_filtered = []
    scores_filtered = []
    for i in range(len(predicted)):
        if predicted[i] not in pred_filtered:
            pred_filtered.append(predicted[i])
            scores_filtered.append(concept_to_image[i].item())
    
    # sort
    sorted_indices = sorted(range(len(scores_filtered)), key=lambda i: scores_filtered[i], reverse=True)
    pred_filtered = [pred_filtered[i] for i in sorted_indices]
    scores_filtered = [scores_filtered[i] for i in sorted_indices]
    avg_score = sum(scores_filtered) / len(scores_filtered)
    
    return pred_filtered, scores_filtered, avg_score

def encode_all_predicted_for_eval(clip_tower, textual_concepts, device):

    predicted = [v for c in textual_concepts for v in c]
    lengths = [len(c) for c in textual_concepts]

    with torch.no_grad():
        features = encode_text(clip_tower, predicted, device)
        
    return features, lengths
        

def circle_analysis(i_texts, all_descriptors, weight_dissection, class_id, topk_w):
    
    i_indices = [all_descriptors.index(t) for t in i_texts]
    _, w_indices = weight_dissection[class_id].topk(topk_w)
    w_indices = w_indices.tolist()
    w_texts = [all_descriptors[ind] for ind in w_indices]
    
    inter = set(i_indices).intersection(w_indices)
    inter_texts = [all_descriptors[intr_id] for intr_id in list(inter)]
    return w_texts, inter_texts

def get_list_mean(l):
    return sum(l) / len(l)

def get_image_and_concept_classifier(classifier_folder, all_descriptors, imagenet_classes):
    ckpts = torch.load('classifiers/' + classifier_folder + '/clip_zeroshot_concept_cls_no_class.pth', map_location=torch.device('cpu'))  
    dim = ckpts['layer']['weight'].shape[-1]
    
    concept_classifier = nn.Linear(dim, len(all_descriptors), bias = False)
    concept_classifier.load_state_dict(ckpts['layer'])  

    image_classifier = nn.Linear(dim, len(imagenet_classes), bias = False)
    ckpts = torch.load('classifiers/' + classifier_folder + '/clip_zeroshot_cls.pth', map_location=torch.device('cpu'))  
    image_classifier.load_state_dict(ckpts['layer'])
    
    return concept_classifier, image_classifier

def get_intersection(X, Y):
    N = len(X)
    M = len(Y)
    
    # Calculate joint entropy H(X,Y)
    contingency_table = np.zeros((N, M))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            if x == y:
                contingency_table[i, j] = 1
    
    contingency_table /= np.sum(contingency_table) 
    joint_entropy = -np.sum(np.sum(contingency_table * np.log(contingency_table + 1e-9)))
    mutual_info = np.log(N) + np.log(M) + joint_entropy
    
    if np.isnan(mutual_info):
        return 0
    
    return mutual_info

def get_statistics(deletion_scores):
    auc = sum(deletion_scores) / len(deletion_scores)
    return auc

def get_deletion_score(concepts, all_descriptors, weight_dissection, class_id, topk_w):
    text_score_pairs = zip(concepts[0], concepts[1])
    text_score_pairs = sorted(text_score_pairs, key=lambda x: x[1], reverse=True)
    i_texts = [text for text, _ in text_score_pairs]
    i_indices = [all_descriptors.index(t) for t in i_texts]
    
    w_values, w_indices = weight_dissection[class_id].topk(topk_w)
    w_values, w_indices = w_values.tolist(), w_indices.tolist()

    sorted_indices_perturb = i_indices.copy()
    perturb_index = len(all_descriptors) + 1
    deletion_scores = []
    deletion_scores.append(get_intersection(sorted_indices_perturb, w_indices))

    for i in range(len(sorted_indices_perturb)):
        sorted_indices_perturb[i] = perturb_index   # smaller is better
        deletion_scores.append(get_intersection(sorted_indices_perturb, w_indices))
        
    return deletion_scores

def get_mean_scores(deletion_scores):
    # list containing the decreasing intersection scores for every image in the validation set
    max_len = max([len(l) for l in deletion_scores])  
    new_scores = []

    for l in deletion_scores:
        padding = max_len - len(l)
        padded_list = [0 for _ in range(padding)]
        new_scores.append(l + padded_list)

    new_scores = torch.FloatTensor(new_scores).mean(0)
    return new_scores.tolist()

def get_cls_data(all_class_folders, results, predictions, topk_w, all_descriptors, weight_dissection):
    
    deletion_scores, deletion_scores_k = [], []
    
    for class_folder in all_class_folders:
        cls_images = list(results[class_folder].keys())

        for img_name in cls_images:
            data = results[class_folder][img_name]
            class_id = predictions[img_name]
            
            concepts_pca, concepts_kmeans = data['concepts_pca'], data['concepts_kmeans']
            d_scores = get_deletion_score(concepts_pca, all_descriptors, weight_dissection, 
                                          class_id, topk_w = topk_w)
            

            deletion_scores.append(d_scores)


            d_scores_k = get_deletion_score(concepts_kmeans, all_descriptors, weight_dissection, 
                                            class_id, topk_w = topk_w)

            deletion_scores_k.append(d_scores_k)

    deletion_scores = get_mean_scores(deletion_scores)
    deletion_scores_k = get_mean_scores(deletion_scores_k)
    
    return deletion_scores, deletion_scores_k

def single_example_scores(class_folder, concepts_with_scores, weight_dissection, folder2id, topk_w, all_descriptors):
    class_id = folder2id[class_folder]
    deletion_scores = get_deletion_score(concepts_with_scores, all_descriptors, weight_dissection, class_id, topk_w = topk_w)
    return deletion_scores

def pad_zeros(scores, max_len):
    if len(scores) < max_len:
        missing_zeros = max_len - len(scores)
        scores = scores + [0 for _ in range(missing_zeros)]
    return scores

