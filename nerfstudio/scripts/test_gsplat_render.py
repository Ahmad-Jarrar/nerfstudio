"""
Simple script to sanity–check the gsplat rasterization function.

This does **not** depend on any dataset; it just:
  - creates a handful of synthetic 3D Gaussian splats,
  - sets up a basic pinhole camera,
  - calls `gsplat.rendering.rasterization`,
  - composites with a white background, and
  - writes an RGB image to disk.

Usage (from repo root):
    pixi run python -m nerfstudio.scripts.test_gsplat_render --output out.png
or:
    python -m nerfstudio.scripts.test_gsplat_render --output out.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

try:
    from gsplat.rendering import rasterization
except ImportError as exc:
    raise SystemExit("Please install `gsplat>=1.0.0` to run this script.") from exc


def build_intrinsics(width: int, height: int, fov_deg: float, device: torch.device) -> torch.Tensor:
    """Construct a simple pinhole intrinsics matrix with square pixels."""
    import math
    fov_rad = math.radians(fov_deg)
    # fx = fy = 0.5 * width / tan(fov/2)
    fx = fy = 0.5 * width / math.tan(0.5 * fov_rad)
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    K = torch.tensor(
        [[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
        device=device,
    )
    return K


def build_viewmat(device: torch.device) -> torch.Tensor:
    """World-to-camera matrix.

    Here we place the camera at the origin, looking down -Z, with +X to the right and +Y up.
    This is the standard OpenGL-style convention that gsplat uses.
    """
    viewmat = torch.eye(4, dtype=torch.float32, device=device)[None, ...]
    return viewmat


def build_gaussians(
    num_points: int,
    device: torch.device,
    feat_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Build a simple cloud of Gaussians in front of the camera, with optional latent features per Gaussian."""
    # Means: spread in X/Y, between Z=1 and Z=3
    means = torch.empty((num_points, 3), dtype=torch.float32, device=device)
    means[:, 0] = torch.empty(num_points, device=device).uniform_(-1.0, 1.0)  # X
    means[:, 1] = torch.empty(num_points, device=device).uniform_(-1.0, 1.0)  # Y
    means[:, 2] = torch.empty(num_points, device=device).uniform_(1.0, 3.0)  # Z (in front of camera)

    # Scales: log of standard deviation per axis; use small isotropic Gaussians.
    base_scale = 0.05
    scales = torch.full((num_points, 3), fill_value=torch.log(torch.tensor(base_scale)), device=device)

    # Quaternions: random unit quaternions; rasterization normalizes anyway.
    quats = torch.randn((num_points, 4), dtype=torch.float32, device=device)
    quats = torch.nn.functional.normalize(quats, dim=-1)

    # Opacities in logit space; 0.5 after sigmoid is a good middle ground.
    opacities = torch.full((num_points, 1), fill_value=torch.logit(torch.tensor(0.5)), device=device)

    # RGB colors per point in [0, 1]
    colors = torch.rand((num_points, 3), dtype=torch.float32, device=device)

    # Optional latent features per Gaussian (e.g. DINO/CLIP-like features)
    if feat_dim is not None and feat_dim > 0:
        latent = torch.randn((num_points, feat_dim), dtype=torch.float32, device=device)
    else:
        latent = None

    return means, scales, quats, opacities, colors, latent


def save_image(rgb: torch.Tensor, output_path: Path) -> None:
    """Save an [H, W, 3] float32 tensor in [0, 1] to disk as PNG."""
    rgb = rgb.clamp(0.0, 1.0).cpu()
    # Use PIL to save without adding new heavy deps.
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("Pillow is required to save images. Install with `pip install Pillow`.") from exc

    img = (rgb * 255.0 + 0.5).to(torch.uint8).numpy()
    image = Image.fromarray(img, mode="RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test gsplat rasterization with a synthetic Gaussian cloud.")
    parser.add_argument("--width", type=int, default=800, help="Output image width.")
    parser.add_argument("--height", type=int, default=600, help="Output image height.")
    parser.add_argument("--num-points", type=int, default=5_000, help="Number of Gaussian splats.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (e.g., 'cuda', 'cuda:0', 'cpu').",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="test_gsplat_render.png",
        help="Path to save the rendered RGB image.",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=60.0,
        help="Horizontal field of view in degrees.",
    )
    parser.add_argument(
        "--rasterize-mode",
        type=str,
        default="classic",
        choices=["classic", "antialiased"],
        help="Rasterization mode to use.",
    )
    parser.add_argument(
        "--feat-dim",
        type=int,
        default=13,
        help="Latent feature dimension per Gaussian (in addition to RGB). Set 0 to disable.",
    )
    parser.add_argument(
        "--output-feat",
        type=str,
        default="test_gsplat_features_pca.png",
        help="Path to save the PCA visualization of rendered latent features.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    width, height = int(args.width), int(args.height)
    output_path = Path(args.output)

    print(f"[test_gsplat_render] Using device: {device}")
    print(f"[test_gsplat_render] Resolution: {width} x {height}")
    print(f"[test_gsplat_render] Number of Gaussians: {args.num_points}")
    print(f"[test_gsplat_render] Latent feature dim: {args.feat_dim}")

    # Build test scene and camera
    means, scales_log, quats, opacities_logit, colors, latent = build_gaussians(
        args.num_points, device=device, feat_dim=args.feat_dim if args.feat_dim > 0 else None
    )
    K = build_intrinsics(width, height, args.fov, device=device)
    viewmat = build_viewmat(device=device)

    # Convert parameterization to what rasterization expects
    scales = torch.exp(scales_log)
    opacities = torch.sigmoid(opacities_logit).squeeze(-1)

    # Fuse RGB colors and latent features into a single per-Gaussian feature vector
    if latent is not None:
        fused_colors = torch.cat([colors, latent], dim=-1)  # [N, 3 + feat_dim]
        feat_dim = latent.shape[-1]
    else:
        fused_colors = colors
        feat_dim = 0

    # Background color (white)
    background = torch.ones(3, dtype=torch.float32, device=device)

    render, alpha, info = rasterization(
        means=means,
        quats=quats,  # will be normalized internally
        scales=scales,
        opacities=opacities,
        colors=fused_colors,
        viewmats=viewmat,  # [1, 4, 4]
        Ks=K,  # [1, 3, 3]
        width=width,
        height=height,
        packed=False,
        near_plane=0.01,
        far_plane=10.0,
        render_mode="RGB+ED",
        sh_degree=None,
        sparse_grad=False,
        absgrad=False,
        rasterize_mode=args.rasterize_mode,
        with_ut=True,
        with_eval3d=True,
    )

    # Composite with background for RGB
    alpha_img = alpha[:, ...]
    rgb = render[:, ..., :3] + (1.0 - alpha_img) * background
    rgb = rgb[0]  # [H, W, 3]
    save_image(rgb, output_path)
    print(f"[test_gsplat_render] Saved RGB image to: {output_path.resolve()}")

    # If we have latent features, extract them from the rendered tensor and save a PCA visualization
    if feat_dim > 0:
        # render has channels: [3 RGB] + [feat_dim latent] + [1 depth]
        feature_start = 3
        feature_end = 3 + feat_dim
        feat_img = render[0, ..., feature_start:feature_end]  # [H, W, feat_dim]

        H, W, D = feat_img.shape
        feat_flat = feat_img.reshape(-1, D)

        # Only keep pixels where alpha > 0 for PCA
        alpha_flat = alpha_img[0].reshape(-1, 1)  # [H*W, 1]
        valid_mask = alpha_flat.squeeze(-1) > 0
        valid_feat = feat_flat[valid_mask]

        if valid_feat.shape[0] >= 3:
            # Center features
            mean = valid_feat.mean(dim=0, keepdim=True)
            X = valid_feat - mean

            # PCA via SVD: X ~ U S V^T, components are rows of V^T
            U, S, Vh = torch.linalg.svd(X, full_matrices=False)
            components = Vh[:3]  # [3, D]

            # Project all pixels (including background) for a full image
            X_all = feat_flat - mean
            proj = (X_all @ components.T)  # [H*W, 3]

            # Normalize each PC to [0, 1]
            proj_min = proj.min(dim=0, keepdim=True).values
            proj_max = proj.max(dim=0, keepdim=True).values
            denom = (proj_max - proj_min).clamp(min=1e-6)
            proj_norm = (proj - proj_min) / denom

            proj_img = proj_norm.reshape(H, W, 3).clamp(0.0, 1.0)

            feat_output_path = Path(args.output_feat)
            save_image(proj_img, feat_output_path)
            print(f"[test_gsplat_render] Saved latent feature PCA image to: {feat_output_path.resolve()}")
        else:
            print("[test_gsplat_render] Not enough valid pixels for PCA; skipping feature visualization.")

    if "num_rendered" in info:
        print(f"[test_gsplat_render] Rendered Gaussians: {info['num_rendered']}")


if __name__ == "__main__":
    main()

