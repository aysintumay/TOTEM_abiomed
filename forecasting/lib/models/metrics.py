import torch

def pearsoncor(
    x: torch.Tensor, y: torch.Tensor, reduction: str = "mean", eps: float = 1e-4
) -> torch.Tensor:
    """
    Args:
        x: tensor of shape (B, T, S)
        y: tensor of shape (B, T, S)
        reduction: str of "mean" or "sum"
    Returns:
        corr: tensor of shape (1,)
    """

    mux = x.mean(dim=1, keepdim=True)
    muy = y.mean(dim=1, keepdim=True)

    u = torch.sum((x - mux) * (y - muy), dim=1)  # (B, S)
    d = torch.sqrt(
        torch.sum((x - mux) ** 2, dim=1) * torch.sum((y - muy) ** 2, dim=1)
    )  # (B, S)

    corr = u / (d + eps)  # (B, S)

    if reduction == "sum":
        corr = corr.sum()
    elif reduction == "mean":
        corr = corr.mean()
    else:
        raise ValueError("Uknown reduction mode")
    return corr


def dtw(x: torch.Tensor, y: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """
    Dynamic Time Warping distance: shape-similarity metric between predicted and
    true trajectories, robust to local time shifts/lag (unlike mse/mae/pearsoncor,
    which compare values index-by-index).

    Args:
        x: tensor of shape (B, T, S)
        y: tensor of shape (B, T, S)
        reduction: str of "mean" or "sum"
    Returns:
        dtw: tensor of shape (1,)
    """
    B, T, S = x.shape
    device = x.device

    cost = (x.unsqueeze(2) - y.unsqueeze(1)).abs()  # (B, Tx, Ty, S)

    D = torch.zeros(B, T, T, S, device=device, dtype=x.dtype)
    D[:, 0, 0, :] = cost[:, 0, 0, :]
    for j in range(1, T):
        D[:, 0, j, :] = D[:, 0, j - 1, :] + cost[:, 0, j, :]
    for i in range(1, T):
        D[:, i, 0, :] = D[:, i - 1, 0, :] + cost[:, i, 0, :]
    for i in range(1, T):
        for j in range(1, T):
            prev_min = torch.minimum(
                torch.minimum(D[:, i - 1, j, :], D[:, i, j - 1, :]), D[:, i - 1, j - 1, :]
            )
            D[:, i, j, :] = cost[:, i, j, :] + prev_min

    dtw_dist = D[:, T - 1, T - 1, :]  # (B, S)

    if reduction == "sum":
        return dtw_dist.sum()
    elif reduction == "mean":
        return dtw_dist.mean()
    else:
        raise ValueError("Unknown reduction mode")


def soft_dtw(x: torch.Tensor, y: torch.Tensor, gamma: float = 0.1, reduction: str = "mean") -> torch.Tensor:
    """
    Differentiable soft-DTW (Cuturi & Blondel 2017): same recurrence as dtw() above,
    but the hard min(a, b, c) is replaced by a soft-min
    -gamma * logsumexp(-a/gamma, -b/gamma, -c/gamma), so gradients flow through all
    three predecessors instead of only the argmin one. Used for the sDTW term in
    L_recon (Eq 31), which requires a smooth morphology-matching loss.

    Args:
        x: tensor of shape (B, T, S)
        y: tensor of shape (B, T, S)
        gamma: softmin temperature; gamma -> 0 recovers hard DTW
        reduction: str of "mean" or "sum"
    Returns:
        soft_dtw: tensor of shape (1,)
    """
    B, T, S = x.shape
    device = x.device

    cost = (x.unsqueeze(2) - y.unsqueeze(1)).abs()  # (B, Tx, Ty, S)

    def softmin(a, b, c):
        stacked = torch.stack([a, b, c], dim=0)  # (3, B, S)
        return -gamma * torch.logsumexp(-stacked / gamma, dim=0)

    D = torch.zeros(B, T, T, S, device=device, dtype=x.dtype)
    D[:, 0, 0, :] = cost[:, 0, 0, :]
    for j in range(1, T):
        D[:, 0, j, :] = D[:, 0, j - 1, :] + cost[:, 0, j, :]
    for i in range(1, T):
        D[:, i, 0, :] = D[:, i - 1, 0, :] + cost[:, i, 0, :]
    for i in range(1, T):
        for j in range(1, T):
            prev_softmin = softmin(D[:, i - 1, j, :], D[:, i, j - 1, :], D[:, i - 1, j - 1, :])
            D[:, i, j, :] = cost[:, i, j, :] + prev_softmin

    dtw_dist = D[:, T - 1, T - 1, :]  # (B, S)

    if reduction == "sum":
        return dtw_dist.sum()
    elif reduction == "mean":
        return dtw_dist.mean()
    else:
        raise ValueError("Unknown reduction mode")


def test_dtw():
    import numpy as np

    def dtw_reference(a, b):
        Ta, Tb = len(a), len(b)
        D = np.zeros((Ta, Tb))
        D[0, 0] = abs(a[0] - b[0])
        for j in range(1, Tb):
            D[0, j] = D[0, j - 1] + abs(a[0] - b[j])
        for i in range(1, Ta):
            D[i, 0] = D[i - 1, 0] + abs(a[i] - b[0])
        for i in range(1, Ta):
            for j in range(1, Tb):
                D[i, j] = abs(a[i] - b[j]) + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
        return D[Ta - 1, Tb - 1]

    B, T, S = 2, 16, 12
    x = torch.randn((B, T, S))
    y = torch.randn((B, T, S))

    our_dtw = dtw(x, y)

    x_np = x.numpy()
    y_np = y.numpy()
    ref_dtw = []
    for b in range(B):
        for s in range(S):
            ref_dtw.append(dtw_reference(x_np[b, :, s], y_np[b, :, s]))
    ref_dtw = sum(ref_dtw) / B / S  # the mean

    assert torch.allclose(
        our_dtw, torch.zeros_like(our_dtw).fill_(ref_dtw), atol=1e-4
    ), "DTW test failed"


def test_pearsoncorr():
    import scipy

    B, T, S = 2, 16, 12
    x = torch.randn((B, T, S))
    y = torch.randn((B, T, S))

    our_corr = pearsoncor(x, y)

    # scipy corr is not vectorized
    x_np = x.numpy()
    y_np = y.numpy()
    scipy_corr = []
    for b in range(B):
        for s in range(S):
            cc = scipy.stats.pearsonr(x_np[b, :, s], y_np[b, :, s])
            scipy_corr.append(cc.statistic)
    scipy_corr = sum(scipy_corr) / B / S  # the mean

    assert torch.allclose(
        our_corr, torch.zeros_like(our_corr).fill_(scipy_corr)
    ), "Pearson test failed"


if __name__ == "__main__":
    test_pearsoncorr()