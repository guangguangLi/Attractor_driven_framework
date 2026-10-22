import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt

class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))



class GatedFusion(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels*2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x1, x2):
        g = self.gate(torch.cat([x1, x2], dim=1))
        return g * x1 + (1.0 - g) * x2


class MultiScaleFusion(nn.Module):
    """两分支：原图 + 小波分解图"""
    def __init__(self, in_channels=3, base_channels=64, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        self.branch_convs_img = nn.ModuleList()
        self.branch_convs_wave = nn.ModuleList()
        self.fusions = nn.ModuleList()

        for i in range(num_scales):
            c = base_channels * (2**i)

            in_c_img = in_channels if i==0 else base_channels*2**(i-1)
            self.branch_convs_img.append(nn.Sequential(
                ConvBlock(in_c_img, c),
                ConvBlock(c, c)
            ))

            in_c_wave = in_channels if i==0 else base_channels*2**(i-1)
            self.branch_convs_wave.append(nn.Sequential(
                ConvBlock(in_c_wave, c),
                ConvBlock(c, c)
            ))

            self.fusions.append(GatedFusion(c))

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = base_channels * (2**(num_scales-1))

    def forward(self, img, wave_img):
        f_img = img
        f_wave = wave_img
        for i in range(self.num_scales):
            f_img = self.branch_convs_img[i](f_img)
            f_wave = self.branch_convs_wave[i](f_wave)
            f = self.fusions[i](f_img, f_wave)
            if i != self.num_scales - 1:
                f_img = F.avg_pool2d(f_img + f, 2)
                f_wave = F.avg_pool2d(f_wave + f, 2)
        feat = self.global_pool(f).flatten(1)
        return feat


class AttractorModule(nn.Module):
    def __init__(self, dim, n_iters=10, alpha=0.5, use_prototypes=True, n_prototypes=2, proto_tau=0.1, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.dim = dim
        self.n_iters = n_iters
        self.alpha = alpha
        self.use_prototypes = use_prototypes
        self.proto_tau = proto_tau

        self.W = nn.utils.spectral_norm(nn.Linear(dim, hidden_dim, bias=False))
        self.Uc = nn.Linear(dim, hidden_dim, bias=False)
        self.Va = nn.Linear(dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.ln = nn.LayerNorm(dim)

        if use_prototypes:
            self.prototypes = nn.Parameter(torch.randn(n_prototypes, dim) * 0.1)
        else:
            self.register_parameter('prototypes', None)

        self.context_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(dim)
        )

    def forward(self, z, context=None):
        z = self.ln(z)
        context_vec = None
        if context is not None:
            context_vec = self.context_proj(context)
    
        proto_attn = None
        z_prev = None
        for i in range(self.n_iters):
            h = self.W(z)
            if context_vec is not None:
                h = h + self.Uc(context_vec)
            if self.use_prototypes:
                proto = F.normalize(self.prototypes, dim=-1)
                z_n = F.normalize(z, dim=-1)
                sim = torch.matmul(z_n, proto.t())
                weights = F.softmax(sim / (self.proto_tau + 1e-8), dim=-1)
                attract = torch.matmul(weights, self.prototypes)
                h = h + self.Va(attract)
                proto_attn = weights
    
            h = torch.tanh(h)
            delta = self.out_proj(h) - z
    
            if i == self.n_iters - 2:
                z_prev = z.clone()
    
            z = z + self.alpha * delta
            z = self.ln(z)
    
        extras = {
            'proto_attn': proto_attn,
            'z_T_last': z,
            'z_T_prev': z_prev
        }
        return z, extras

def wavelet_decompose(img, wavelet='haar', level=1):
    """
    """
    B,C,H,W = img.shape
    recon_imgs = []
    for b in range(B):
        ch_list = []
        for c in range(C):
            coeffs = pywt.wavedec2(img[b,c].cpu().numpy(), wavelet=wavelet, level=level)
            approx = coeffs[0]
            approx = torch.tensor(approx, dtype=torch.float32)
            approx = F.interpolate(approx.unsqueeze(0).unsqueeze(0), size=(H,W), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            ch_list.append(approx)
        recon_imgs.append(torch.stack(ch_list, dim=0))
    recon_imgs = torch.stack(recon_imgs, dim=0).to(img.device)
    return recon_imgs

class MultiModalClassifier(nn.Module):
    def __init__(self, num_classes=2, base_channels=32, num_scales=3,
                 attractor_kwargs=None):
        super().__init__()
        self.encoder = MultiScaleFusion(
            in_channels=3,
            base_channels=base_channels,
            num_scales=num_scales
        )
        feat_dim = self.encoder.feature_dim
        self.attractor = AttractorModule(dim=feat_dim, **(attractor_kwargs or {}))
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, img):
        wave_img = wavelet_decompose(img, level=1)
        feat = self.encoder(img, wave_img)
        z_T, extras = self.attractor(feat)
        out = self.classifier(z_T)
        return out,extras
