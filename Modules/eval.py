import torch

def evaluate_metrics(cover, stego, secret, revealed_secret):
    """Calculate evaluation metrics"""

    def psnr(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(1.0 / torch.sqrt(mse))

    def ssim(img1, img2):
        # Simplified SSIM calculation
        mu1 = torch.mean(img1)
        mu2 = torch.mean(img2)
        sigma1 = torch.var(img1)
        sigma2 = torch.var(img2)
        sigma12 = torch.mean((img1 - mu1) * (img2 - mu2))

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        ssim_val = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))
        return ssim_val

    # Calculate metrics
    cover_psnr = psnr(cover, stego)
    secret_psnr = psnr(secret, revealed_secret)
    cover_ssim = ssim(cover, stego)
    secret_ssim = ssim(secret, revealed_secret)

    return {
        'cover_psnr': cover_psnr.item(),
        'secret_psnr': secret_psnr.item(),
        'cover_ssim': cover_ssim.item(),
        'secret_ssim': secret_ssim.item()
    }
