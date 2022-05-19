import torch


def entropy(p):
    """
    Computes the entropy of a probability tensor p

    Argument:
        p - probability tensor
        q - probability tensor
    """
    return -1 * torch.matmul(p, torch.log(p).t()).sum()


def kl_div(p, q):
    """
    Computes the KL divergence between two distributions p and q

    Argument:
        p - probability tensor
        q - probability tensor
    """
    return torch.matmul(p, torch.log(p / q).t()).mean()


def kl_div_all_heads(p, q):
    """
    Computes the KL divergence among all head distributions [p] and [q]

    Argument:
        p - list of probability distributions
        q - list of probability distributions
    """
    assert p.shape == q.shape

    div = 0
    for _ in range(len(p)):
        kl = kl_div(p[_], q[_])
        div += kl
    return div


def max_kl_div(distances, indices, q):
    """
    Maximum divergence between a distribution q 
    and a set of distributions.
    """
    scores = []

    if len(indices) == 0:
        return 0.0

    for idx in indices:
        scores.append(
            kl_div_all_heads(q, distances[idx]).detach().cpu().numpy())
    return max(scores)