# Attack the embedding so that the logits for text_queries are pushed to be lower
import torch.optim as optim
from torch.optim import Adam
from torch.autograd import gradcheck
import torch

# NOTE Old using adam
# def perturb_owl_embedding(original_embedding, custom_owl, negative_queries, positive_queries, lr=0.005, num_iterations=200, verbose=True):
#     perturbed_embedding = original_embedding.clone().detach()
#     perturbed_embedding.requires_grad = True

#     if not isinstance(negative_queries, list):
#         raise ValueError(f"Negative queries should be list, not {type(negative_queries)}")
#     if not isinstance(positive_queries, list):
#         raise ValueError(f"Positive queries should be list, not {type(positive_queries)}")

#     text_queries = negative_queries + positive_queries

#     def adversarial_loss_switch_logits(logits):
#         # Owl-ViT logits have shape [batch, 576, n_text_queries].
#         # For detection, you take the maximum logit, perform a sigmoid, and that's your probability
#         # this attack assumes that you have positive queries for which you want to increase the logit
#         # and negative queries for which you want to decrease logit.
#         target_logits = torch.full(logits.shape, -1*1e6).to(custom_owl.device)
#         target_logits[...,-len(positive_queries)] = 1e6
#         return torch.nn.functional.mse_loss(logits, target_logits)

#     optimizer = Adam([perturbed_embedding], lr=lr)  # 0.005

#     for i in range(num_iterations):
#         optimizer.zero_grad()

#         # Forward pass through the model
#         model_out = custom_owl.run_from_vision_features(text_queries, perturbed_embedding)
#         logits = model_out.logits

#         # Compute the adversarial loss
#         loss = adversarial_loss_switch_logits(logits)

#         # Backward pass and optimization step
#         loss.backward()
#         optimizer.step()
#         if verbose:
#             print(f"\rStep: {i} loss: {loss.item()}", end="")
    
#     if verbose:
#         print(f"\nThe new embedding differs by {torch.norm(perturbed_embedding-original_embedding)}")
    
#     return perturbed_embedding.clone().detach()

# NOTE new using FGSM
def perturb_owl_embedding(original_embedding, custom_owl, negative_queries, positive_queries, lr=0.01, num_iterations=200, verbose=True):
    perturbed_embedding = original_embedding.clone().detach()
    perturbed_embedding.requires_grad = True

    if not isinstance(negative_queries, list):
        raise ValueError(f"Negative queries should be list, not {type(negative_queries)}")
    if not isinstance(positive_queries, list):
        raise ValueError(f"Positive queries should be list, not {type(positive_queries)}")

    text_queries = negative_queries + positive_queries

    def adversarial_loss_switch_logits(logits):
        target_logits = torch.full(logits.shape, -1*1e6).to(custom_owl.device)
        target_logits[...,-len(positive_queries)] = 1e6
        return torch.nn.functional.mse_loss(logits, target_logits)

    # Compute gradient of loss w.r.t. input
    model_out = custom_owl.run_from_vision_features(text_queries, perturbed_embedding)
    logits = model_out.logits
    loss = adversarial_loss_switch_logits(logits)
    loss.backward()
    gradient = perturbed_embedding.grad.data

    # Generate perturbation using gradient sign
    perturbation = torch.sign(gradient)
    perturbed_embedding = original_embedding - lr * perturbation

    if verbose:
        print(f"The new embedding differs by {torch.norm(perturbed_embedding-original_embedding)}")

    return perturbed_embedding.clone().detach()


def get_delta_characteristic(custom_llava, custom_owl, token_projection, image, characteristic, original_characteristic, adversarial_characteristic, verbose=True, adversarial_lr=0.005, adversarial_iter=200):
    # Run LLaVa on original embedding
    owl_vision_embedding = custom_owl.get_vision_features(image)
    prompt = f"<image>\nUSER: From one to ten, how likely is the person to be a {characteristic}? Only answer with a number from 1 to 10.\nASSISTANT:"

    aligned_vision_embedding = token_projection(owl_vision_embedding[:, 1:])  # TODO do i need [:, 1:]??
    original_gen_text = custom_llava.generate(prompt, image, custom_vision=aligned_vision_embedding)
    try:
        original_score = int(original_gen_text.split("ASSISTANT: ")[1])
    except:
        raise ValueError("Cant parse model output")

    if verbose:
        print(f"Original score for {characteristic}: {original_score}")

    # Perturb embedding
    perturbed_embedding = perturb_owl_embedding(owl_vision_embedding, custom_owl, [f"a photo of a {original_characteristic}"], [f"a photo of a {adversarial_characteristic}"], lr=adversarial_lr, num_iterations=adversarial_iter, verbose=verbose)

    # Run LLaVa on perturbed vision
    aligned_perturbed_vision_embedding = token_projection(perturbed_embedding[:, 1:])  # TODO do i need [:, 1:]??
    perturbed_gen_text = custom_llava.generate(prompt, image, custom_vision=aligned_perturbed_vision_embedding)
    try:
        perturbed_score = int(perturbed_gen_text.split("ASSISTANT: ")[1])
    except:
        raise ValueError("Cant parse model output")
    
    if verbose:
        print(f"Perturbed score for {characteristic}: {perturbed_score}")
    
    return perturbed_score-original_score, perturbed_embedding

def get_characteristic_score(custom_llava, custom_owl, token_projection, image, characteristic):
    owl_vision_embedding = custom_owl.get_vision_features(image)
    prompt = f"<image>\nUSER: From one to ten, how likely is the person to be a {characteristic}? Only answer with a number from 1 to 10.\nASSISTANT:"

    aligned_vision_embedding = token_projection(owl_vision_embedding[:, 1:])  # TODO do i need [:, 1:]??
    gen_text = custom_llava.generate(prompt, image, custom_vision=aligned_vision_embedding)
    try:
        score = int(gen_text.split("ASSISTANT: ")[1])
    except:
        raise ValueError("Cant parse model output")
    return score