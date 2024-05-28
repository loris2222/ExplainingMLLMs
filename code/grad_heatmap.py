from gradient_utils import compute_llava_logit_vision_gradient, compute_owl_vision_gradient
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageDraw
import torch
import io
from PIL import Image

def compute_grad_similarities(custom_llava, custom_owl, vision_projection, image, prompt, token_string=None, token_id=None, owl_class=None, token_position=-1):
    """
    Computes gradient similarities between LLaVa and Owl outputs for the same image
    """
    if token_string is not None:
        # Get token id from string, ensuring that the string only takes one token
        tokenized_string = custom_llava.llava_processor.tokenizer.encode(token_string)
        # assert len(tokenized_string) == 2
        token_id = tokenized_string[token_position]  # -1, token_position
    elif token_id is not None:
        token_string = custom_llava.llava_processor.tokenizer.decode(token_id)
    
    if owl_class is None:
        owl_class = token_string

    # Get Owl vision features
    owl_vision_embedding = custom_owl.get_vision_features(image)

    # Compute gradients over LLaVa
    llava_grad = compute_llava_logit_vision_gradient(custom_llava, vision_projection, token_id, owl_vision_embedding, prompt, image, token_position=token_position)

    # Compute gradients for all features in Owl-ViT (takes a few seconds)
    text_queries = ["a photo of a " + owl_class.strip()]
    all_owl_grads = compute_owl_vision_gradient(custom_owl, owl_vision_embedding, text_queries)

    # Normalize grads
    llava_grad = llava_grad / torch.norm(llava_grad)
    norms_all_owl_grads = all_owl_grads.pow(2).sum(keepdim=True, dim=[-3, -2, -1]).sqrt()
    all_owl_grads = all_owl_grads / norms_all_owl_grads

    # Compute cosine similarity between all_owl_grads and llava_grad
    grad_similarities = torch.tensordot(all_owl_grads, llava_grad, dims=[[-3, -2, -1], [-3, -2, -1]])

    # NOTE Normalization should be done per text-query, but since we only have one, I can do it in one pass
    grad_similarities[:,:-4] = (grad_similarities[:,:-4] - torch.min(grad_similarities[:,:-4])) / (torch.max(grad_similarities[:,:-4]) - torch.min(grad_similarities[:,:-4]))
    grad_similarities[:,-4:] = (grad_similarities[:,-4:] - torch.min(grad_similarities[:,-4:])) / (torch.max(grad_similarities[:,-4:]) - torch.min(grad_similarities[:,-4:]))

    # Display similarities over boxes and image
    return grad_similarities


def visualize_grad_similarities(image, owl_output, grad_similarities, class_sim_threshold=0.5, box_sim_threshold=0.0, owl_score_threshold=0.0, class_id=0, max_thick=10):
    """
    Visualizes the grad similarities using colored boxes 
    """

    logits = owl_output["logits"].detach()
    pred_boxes = owl_output["pred_boxes"].detach()
    cell_size = 32  # Owl ViT vision encoder transformer patch size
    grid_size = [24, 24]  # Owl ViT vision encoder token grid size
    cmap_scale = 256
    imsize = 768
    cmap = cm.get_cmap('viridis', 256)

    im_768 = image.resize((imsize, imsize))
    draw = ImageDraw.Draw(im_768)

    # Scan out featuremap
    for token_id in range(logits.shape[1]):
        # Extract box and grad similarity data for this feature map token
        box_logits = logits[0,token_id]
        box_coords = pred_boxes[0,token_id]
        grad_similarities_coords = grad_similarities[token_id, -4:]
        grad_similarities_class = grad_similarities[token_id, :-4]

        # Skip box if logits not large enough
        probs, _ = torch.max(box_logits, dim=-1)
        box_score = torch.sigmoid(probs)

        if box_score < owl_score_threshold:
            continue
        
        # Draw box
        x0 = int(imsize * (box_coords[0]-(box_coords[2]/2)))
        x1 = int(imsize * (box_coords[0]+(box_coords[2]/2)))
        y0 = int(imsize * (box_coords[1]-(box_coords[3]/2)))
        y1 = int(imsize * (box_coords[1]+(box_coords[3]/2)))

        # Height and width weight
        vweight = (grad_similarities_coords[2]).cpu()
        hweight = (grad_similarities_coords[3]).cpu()

        # Class weight
        cweight = (torch.max(grad_similarities_class[class_id])).cpu()

        if cweight < class_sim_threshold:  # What is a good strategy for this threshold?
            continue

        if vweight < box_sim_threshold and hweight < box_sim_threshold:
            continue

        # Thickness is similarity with class, border color is with width height
        draw.line((x0,y0,x1,y0), width=int(np.floor(max_thick*cweight)), fill=tuple(int(x*cmap_scale) for x in cmap(hweight)[:-1]))  # Top
        draw.line((x1,y0,x1,y1), width=int(np.floor(max_thick*cweight)), fill=tuple(int(x*cmap_scale) for x in cmap(vweight)[:-1]))  # Right
        draw.line((x0,y1,x1,y1), width=int(np.floor(max_thick*cweight)), fill=tuple(int(x*cmap_scale) for x in cmap(hweight)[:-1]))  # Bottom
        draw.line((x0,y0,x0,y1), width=int(np.floor(max_thick*cweight)), fill=tuple(int(x*cmap_scale) for x in cmap(vweight)[:-1]))  # Left

        # Draw middle cross
        # TODO

        # Draw red line from token position in featuremap to center
        token_pos_x = (token_id % grid_size[1]) * cell_size
        token_pos_y = (token_id // grid_size[0]) * cell_size
        draw.line((token_pos_x, token_pos_y, imsize * box_coords[0],imsize * box_coords[1]), width=1, fill=(255,0,0))  # Left


    fig, ax = plt.subplots(1)
    sm = plt.cm.ScalarMappable(cmap=cmap)
    fig.colorbar(sm, ax=ax)
    ax.imshow(im_768)

    # Convert plot to PIL image and return
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    ret_im = Image.open(buf)
    return ret_im

def get_grad_saliency(image, grad_similarities, owl_outputs, class_id=0, use_weight="class"):
    """
    Computes a heatmap that shows the class_similarities
    Returns a numpy array with the same shape as the image size and between 0 and 1
    """

    height = image.height
    width = image.width

    out_cam = np.zeros((height, width))

    # Generate all boxes and print them on the saliency
    logits = owl_outputs["logits"].detach()
    pred_boxes = owl_outputs["pred_boxes"].detach()

    # Scan out featuremap
    for token_id in range(logits.shape[1]):
        # Extract box and grad similarity data for this feature map token
        box_coords = pred_boxes[0,token_id]
        
        grad_similarities_coords = grad_similarities[token_id, -4:]
        grad_similarities_class = grad_similarities[token_id, :-4]
        
        # Draw box
        x0 = np.clip(int(width * (box_coords[0]-(box_coords[2]/2))), 0, width)
        x1 = np.clip(int(width * (box_coords[0]+(box_coords[2]/2))), 0, width)
        y0 = np.clip(int(height * (box_coords[1]-(box_coords[3]/2))), 0, height)
        y1 = np.clip(int(height * (box_coords[1]+(box_coords[3]/2))), 0, height)

        # Class weight
        cweight = (torch.max(grad_similarities_class[class_id])).cpu()
        coordweight = (torch.max(grad_similarities_coords)).cpu()
        anyweight = (torch.max(grad_similarities[token_id, :])).cpu()

        if use_weight == "class":
            box_weight = cweight
        if use_weight == "coord":
            box_weight = coordweight
        elif use_weight == "any":
            box_weight = anyweight

        # Set box pixels in saliency map
        overlay_mask = np.zeros_like(out_cam)
        overlay_mask[y0:y1,x0:x1] = box_weight

        out_cam = np.max(np.concatenate((overlay_mask[:,:,np.newaxis], out_cam[:,:,np.newaxis]), axis=2), axis=2)
    
    return out_cam