"""Hand-derived analytic backward for the matrix-native hot path.

The two NEW pieces: poly-splat weight derivative + WSR (normalized weighted sum) backward.
Geometry Jacobians are reused upstream (autograd / ported), not here. `wsr_polysplat_ref` is the
plain-torch (autograd) reference; `WSRPolySplatHot` is the manual forward+backward that a Metalium
kernel will mirror. Both are validated against finite-diff gradcheck in test_backward.py.
"""
import torch


def wsr_polysplat_ref(theta_u, o, color, w_b, Phi, c_b, keep=None):
    """Plain-torch reference forward (autograd handles backward). Returns C[P,3]."""
    u = Phi @ theta_u.t()                       # [P,G]
    w_geo = u.clamp(min=0.0) ** 2
    if keep is not None:
        w_geo = w_geo * keep[None, :]
    w = o[None, :] * w_geo                       # [P,G]
    num = w @ color + w_b * c_b[None, :]         # [P,3]
    den = w.sum(1, keepdim=True) + w_b           # [P,1]
    return num / den


class WSRPolySplatHot(torch.autograd.Function):
    """Manual forward+backward for the hot path. Leaves: theta_u[G,6], o[G], color[G,3], w_b(scalar).
    Phi[P,6], c_b[3], keep[G] are constants (no grad)."""

    @staticmethod
    def forward(ctx, theta_u, o, color, w_b, Phi, c_b, keep):
        u = Phi @ theta_u.t()                    # [P,G]
        relu_u = u.clamp(min=0.0)
        w_geo = relu_u * relu_u
        if keep is not None:
            w_geo = w_geo * keep[None, :]
        w = o[None, :] * w_geo                   # [P,G]
        num = w @ color + w_b * c_b[None, :]     # [P,3]
        den = w.sum(1, keepdim=True) + w_b       # [P,1]
        C = num / den
        ctx.save_for_backward(color, Phi, c_b, o, relu_u, w_geo, w, C, den)
        ctx.keep = keep                          # constant (no grad)
        return C

    @staticmethod
    def backward(ctx, gC):                       # gC [P,3]
        color, Phi, c_b, o, relu_u, w_geo, w, C, den = ctx.saved_tensors
        keep = ctx.keep
        inv = 1.0 / den                          # [P,1]
        gnum = gC * inv                          # [P,3]
        gden = -(gC * C).sum(1, keepdim=True) * inv     # [P,1]   (TRAP #1: keep this term)
        gw = gnum @ color.t() + gden             # [P,G]
        gcolor = w.t() @ gnum                    # [G,3]
        gw_b = (gnum * c_b[None, :]).sum() + gden.sum()
        go = (gw * w_geo).sum(0)                 # [G]
        gw_geo = gw * o[None, :]
        if keep is not None:
            gw_geo = gw_geo * keep[None, :]
        gu = gw_geo * (2.0 * relu_u)             # [P,G]   (TRAP #2: clamp mask is implicit in relu_u)
        gtheta = gu.t() @ Phi                    # [G,6]
        return gtheta, go, gcolor, gw_b, None, None, None


def wsr_polysplat(theta_u, o, color, w_b, Phi, c_b, keep=None):
    """Manual-backward entry point (same signature/result as the reference)."""
    return WSRPolySplatHot.apply(theta_u, o, color, w_b, Phi, c_b, keep)
