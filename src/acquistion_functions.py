import torch, torch.nn.functional as F

from taal_core import jsd_score_tta


def random_score(model, imgs, **kwargs):
    """Assigns a random score to each image in the batch.

    Args:
        model (nn.Module): The model (not used in this function, but included for API consistency).
        imgs (torch.Tensor): The batch of input images.
        **kwargs: Additional keyword arguments (for API consistency).

    Returns:
        torch.Tensor: A tensor of random scores, one for each image in the batch.
    """
    return torch.rand(imgs.size(0), device=imgs.device) # Shape: (B,)

def entropy(model, imgs, T=8, num_classes=5):
    """Calculates the predictive entropy for a batch of images.

    Predictive entropy is a measure of uncertainty. It is calculated by first obtaining
    the average of multiple stochastic forward passes (Monte Carlo dropout), and then
    computing the entropy of this averaged prediction.

    Args:
        model (nn.Module): The model with dropout layers.
        imgs (torch.Tensor): The batch of input images.
        T (int, optional): The number of stochastic forward passes. Defaults to 8.
        num_classes (int, optional): The number of classes. Defaults to 4.

    Returns:
        torch.Tensor: A tensor of entropy scores, one for each image in the batch.
    """
    model.train() # Keep dropout ON for stochasticity
    probs_sum = torch.zeros(
        imgs.size(0), num_classes, *imgs.shape[2:], device=imgs.device
    ) # Shape: (B, C, H, W)

    for _ in range(T):
        with torch.amp.autocast('cuda'):
            logits = model(imgs) # Shape: (B, C, H, W)
            probs  = F.softmax(logits, 1) # Shape: (B, C, H, W)
        probs_sum += probs

    probs_mean = probs_sum / T # Shape: (B, C, H, W)
    ent = -(probs_mean * probs_mean.log()).sum(dim=1) # Shape: (B, H, W)
    return ent.sum(dim=(1, 2)) # Shape: (B,)


def BALD(model, imgs, T=8, num_classes=4):
    """Calculates the BALD (Bayesian Active Learning by Disagreement) score for a batch of images.

    BALD measures the mutual information between the model's predictions and its parameters.
    It is calculated as the difference between the predictive entropy and the expected entropy
    over multiple stochastic forward passes.

    Args:
        model (nn.Module): The model with dropout layers.
        imgs (torch.Tensor): The batch of input images.
        T (int, optional): The number of stochastic forward passes. Defaults to 8.
        num_classes (int, optional): The number of classes. Defaults to 4.

    Returns:
        torch.Tensor: A tensor of BALD scores, one for each image in the batch.
    """
    model.train()
    probs_sum = torch.zeros(
        imgs.size(0), num_classes, *imgs.shape[2:], device=imgs.device
    ) # Shape: (B, C, H, W)
    entropies_sum = torch.zeros(
        imgs.size(0), *imgs.shape[2:], device=imgs.device
    ) # Shape: (B, H, W)

    for _ in range(T):
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                logits = model(imgs) # Shape: (B, C, H, W)
                probs = F.softmax(logits, dim=1) # Shape: (B, C, H, W)

        # Accumulate probabilities for calculating predictive entropy
        probs_sum += probs

        # Calculate entropy of the current prediction and accumulate it
        # H[y|x, Θ_t] = -Σ_c (p_c * log(p_c))
        entropy_t = -(probs * torch.log(probs + 1e-12)).sum(dim=1) # Shape: (B, H, W)
        entropies_sum += entropy_t

    # 1. Calculate Predictive Entropy: H[y|x]
    probs_mean = probs_sum / T # Shape: (B, C, H, W)
    predictive_entropy = -(probs_mean * torch.log(probs_mean + 1e-12)).sum(dim=1) # Shape: (B, H, W)

    # 2. Calculate Expected Entropy: E[H[y|x, Θ]]
    expected_entropy = entropies_sum / T # Shape: (B, H, W)

    # 3. Compute BALD score for each pixel
    # I(y; Θ|x) = H[y|x] - E[H[y|x, Θ]]
    bald_map = predictive_entropy - expected_entropy # Shape: (B, H, W)

    return bald_map.mean(dim=(1, 2)) # Shape: (B,)



def committee_kl_divergence(model, imgs, T=8, num_classes=4):
    """Calculates the KL divergence between the deterministic and stochastic predictions.

    This function measures the disagreement between the model's deterministic prediction
    (with dropout turned off) and its stochastic prediction (the average of multiple
    forward passes with dropout turned on). A higher KL divergence indicates greater
    model uncertainty.

    Args:
        model (nn.Module): The model with dropout layers.
        imgs (torch.Tensor): The batch of input images.
        T (int, optional): The number of stochastic forward passes. Defaults to 8.
        num_classes (int, optional): The number of classes. Defaults to 4.

    Returns:
        torch.Tensor: A tensor of KL divergence scores, one for each image in the batch.
    """
    B, _, H, W = imgs.shape
    device     = imgs.device

    # 1) Monte Carlo posterior under dropout
    model.train()
    all_probs = torch.zeros(T, B, num_classes, H, W, device=device) # Shape: (T, B, C, H, W)
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for i in range(T):
            logits      = model(imgs) # Shape: (B, C, H, W)
            all_probs[i] = F.softmax(logits, dim=1) # Shape: (B, C, H, W)
    posterior = all_probs.mean(dim=0) # Shape: (B, C, H, W)

    # 2) Deterministic “standard” prediction
    model.eval()   # Turn OFF dropout here
    with torch.no_grad(), torch.amp.autocast('cuda'):
        logits        = model(imgs) # Shape: (B, C, H, W)
        standard_probs = F.softmax(logits, dim=1) # Shape: (B, C, H, W)

    # 3) Compute per-pixel KL(standard || posterior)
    eps      = 1e-12
    P        = torch.clamp(standard_probs,   min=eps)
    Q        = torch.clamp(posterior,        min=eps)
    kl_map   = P * (P.log() - Q.log()) # Shape: (B, C, H, W)
    kl_pixel = kl_map.sum(dim=1) # Shape: (B, H, W)
    kl_score = kl_pixel.mean(dim=(1, 2)) # Shape: (B,)

    return kl_score

def committee_js_divergence(model, imgs, T=8, num_classes=4):
    """Calculates the Jensen-Shannon divergence between the deterministic and stochastic predictions.

    Similar to the KL divergence, the JS divergence measures the disagreement between the
    model's deterministic and stochastic predictions. It is a symmetric version of the
    KL divergence and is always finite.

    Args:
        model (nn.Module): The model with dropout layers.
        imgs (torch.Tensor): The batch of input images.
        T (int, optional): The number of stochastic forward passes. Defaults to 8.
        num_classes (int, optional): The number of classes. Defaults to 4.

    Returns:
        torch.Tensor: A tensor of JS divergence scores, one for each image in the batch.
    """
    B, _, H, W = imgs.shape
    device = imgs.device

    # 1) Monte Carlo posterior Q
    model.train()  # Keep dropout on
    all_probs = torch.zeros(T, B, num_classes, H, W, device=device) # Shape: (T, B, C, H, W)
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for i in range(T):
            logits = model(imgs) # Shape: (B, C, H, W)
            all_probs[i] = F.softmax(logits, dim=1) # Shape: (B, C, H, W)
    Q = all_probs.mean(dim=0) # Shape: (B, C, H, W)

    # 2) Deterministic standard prediction p
    model.eval()  # Turn dropout off
    with torch.no_grad(), torch.amp.autocast('cuda'):
        logits = model(imgs) # Shape: (B, C, H, W)
        p = F.softmax(logits, dim=1) # Shape: (B, C, H, W)

    # 3) Build mixture M and clamp for numerical stability
    eps      = 1e-12
    p = torch.clamp(p, min=eps)
    Q = torch.clamp(Q, min=eps)
    M = torch.clamp(0.5 * (p + Q), min=eps)

    # 4) Compute ½ KL(p‖M) + ½ KL(Q‖M) per pixel
    kl_p_m = p * (p.log() - M.log()) # Shape: (B, C, H, W)
    kl_q_m = Q * (Q.log() - M.log()) # Shape: (B, C, H, W)
    js_map   = 0.5 * (kl_p_m + kl_q_m).sum(dim=1) # Shape: (B, H, W)
    js_score = js_map.mean(dim=(1, 2)) # Shape: (B,)

    return js_score

def taal_unweighted_score(model, U, device, K=8):
    return jsd_score_tta(model, U, K=K, alpha=0.5, device=device)  # [B]

def taal_weighted_score(model, U, device, K=8):
    return jsd_score_tta(model, U, K=K, alpha=0.75, device=device) # [B]