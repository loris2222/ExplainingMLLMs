from custom_llava import CustomLlava
from custom_owl import CustomOwl
import torch
import gc

def compute_llava_logit_vision_gradient(custom_llava: CustomLlava, token_projection, grad_token_id, image_features_fromowl, prompt, image, token_position=-1):
    """
    Returns the gradient of the grad_token_id logit at the output of LLaVa with respect to 
    the output of Owl image_features_fromowl before the alignment projection.
    """
    image_features_fromowl = image_features_fromowl.detach().requires_grad_(True)
    custom_llava.half()

    custom_llava.llava_model.requires_grad_(True)
    custom_llava.llava_model.zero_grad()

    in_vision = token_projection(image_features_fromowl).half()
    in_vision.retain_grad()

    llava_out = custom_llava.run_single(prompt, image, custom_vision=in_vision[:, 1:])  # llava_out.logits is a long sequence since the first 576 are just the image tokens!   # TODO do I need owl_vision_embedding[:, 1:]??
    llava_out.logits.retain_grad()

    loss = custom_llava.get_token_logit(llava_out, grad_token_id, token_position=token_position)

    loss.backward()

    llava_grad = image_features_fromowl.grad.detach().float()  # Gradient of output token prob w.r.t. custom_vision_tokens 

    # Clear memory and gradients or else they don't fit in memory
    import gc

    try:
        del image_features_fromowl
        del llava_out
        del loss
        custom_llava.llava_model.requires_grad_(False)
        gc.collect()
    except NameError:
        print(torch.cuda.memory_summary())
        pass

    return llava_grad

def compute_owl_vision_gradient(custom_owl: CustomOwl, image_features_fromowl, text_queries):
    """
    Returns the gradients of all owl outputs with respect to the vision image_features_fromowl.
    """
    image_features_fromowl = image_features_fromowl.detach().requires_grad_(True)
    
    custom_owl.owl_model.requires_grad_(True)
    custom_owl.owl_model.zero_grad()

    owlvit_outputs = custom_owl.run_from_vision_features(text_queries,
                                                        image_features_fromowl,
                                                        )

    num_classes = owlvit_outputs["logits"].shape[2]
    num_out_tokens = owlvit_outputs["logits"].shape[1]

    all_owl_grads = torch.zeros([num_out_tokens, num_classes + 4] + list(image_features_fromowl.shape)).to(custom_owl.device)

    for token_id in range(owlvit_outputs["logits"].shape[1]):
        for coord in range(4):
            grad = torch.autograd.grad(owlvit_outputs["pred_boxes"][0,token_id,coord], image_features_fromowl, retain_graph=True)[0]
            all_owl_grads[token_id, num_classes + coord] = grad.detach()

        for class_id in range(owlvit_outputs["logits"].shape[2]):
            grad = torch.autograd.grad(owlvit_outputs["logits"][0,token_id,class_id], image_features_fromowl, retain_graph=True)[0]
            all_owl_grads[token_id, class_id] = grad.detach()

    try:
        del image_features_fromowl, grad
        custom_owl.owl_model.requires_grad_(False)
        gc.collect()
    except NameError:
        print(torch.cuda.memory_summary())
        pass

    return all_owl_grads