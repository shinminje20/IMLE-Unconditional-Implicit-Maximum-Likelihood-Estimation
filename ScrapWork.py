class CPSRLoss:
    """Contrastive Perceptual Similarity Regression Loss.

    1. Let cos(x,y) be the cosine distance between *embeddings of* x and y
    2. Let lpips(x, y) be the LPIPS distance between *images* x and y, divided
        by a constant M, and then truncated at one. M is chosen to be the
        minimum/average LPIPS distance between any two images in the dataset.

    For negatives n1 ... nN and positives p1 and p2, loss is for p1 is

            exp[ - (cos(lpips(p1, p2) / pi) - cos(p1, p2)) / temp ]
        —————————————————————————————————————————————————————————
            sum over [i=1 ... N] exp[ - cos(p1, ni) / temp]

    Basically, this is regressing cosine embedding distances to modified LPIPS
    distances in the numerator, and regular contrastive loss in the denominator.
    This modifies the familiar NT X-Entropy loss to include a *graded similarity
    metric* that should make it mirror LPIPS distances in dense regions of
    embedding space.

    Hopefully features are better!
    """

    def __init__(self, max_lpips_dist, temp=.5):
        """Args:
        max_lpips_dist  -- the maximum
        temp            -- contrastive loss temperature
        """
        self.temp = temp
        self.lpips = LPIPS(net="vgg").to(device)
        self.pi = torch.tensor(math.pi)
        self.max_lpips_dist = max_lpips_dist

    def __call__(self, x1, x2, fx1, fx2):
        """Returns the loss from pre-normalized projections [fx1] and [fx2]."""
        out = torch.cat([fx1, fx2], dim=0)
        n_samples = len(out)

        # Compute the positive samples loss. At a high level, this is regression
        # on normalized LPIPs distances.
        percept_dists = torch.min(self.lpips(x1, x2) / self.max_lpips_dist, 1)
        percept_dists = torch.cos(percept_dists, out=percept_dists)

        embed_dists = torch.sum(fx1 * fx2, dim=-1)
        pos = torch.exp((embed_dists - percept_dists) / self.temp)

        # Compute negative samples loss. This is exactly the denominator in
        # standard NT X-entropy loss.
        cov = torch.mm(out, out.t().contiguous())
        sim = torch.exp(cov / self.temp)
        mask = ~torch.eye(n_samples, device=sim.device).bool()
        neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)

        return -torch.log(pos / neg).mean()
