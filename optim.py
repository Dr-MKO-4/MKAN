"""
optim.py  Optimiseur Adam compatible DirectML (Intel Iris Xe / Arc).

PyTorch 2.x utilise exp_avg.lerp_() pour la mise à jour du 1er moment
et addcmul_() pour le 2e moment ; ces deux ops ne sont pas supportées par
le backend DirectML et tombent en fallback CPU silencieux, ce qui crée un
round-trip GPU→CPU→GPU à chaque optimizer.step().

DMLAdam reproduit exactement Adam (Kingma & Ba, 2015) avec des ops primitives
garanties sur DirectML :
  mul_(), add_(alpha=), square(), sqrt()  →  toutes supportées nativement.

Référence mathématique (section 4.3.4 du mémoire) :
  m_t = β1·m_{t-1} + (1-β1)·g_t            (1er moment, eq. Adam)
  v_t = β2·v_{t-1} + (1-β2)·g_t²           (2e moment, eq. Adam)
  p_t = p_{t-1} − (lr/b1) · m_t / (√(v_t/b2) + ε)
"""

import torch


class DMLAdam(torch.optim.Optimizer):
    """
    Adam (Kingma & Ba, 2015) compatible DirectML.

    Identique à torch.optim.Adam(foreach=False) sur le plan mathématique,
    mais remplace les ops non supportées par Intel Iris Xe :
      • exp_avg.lerp_(grad, 1-β1)         →  exp_avg.mul_(β1).add_(grad, alpha=1-β1)
      • exp_avg_sq.addcmul_(grad, grad)   →  exp_avg_sq.mul_(β2).add_(grad.square(), alpha=1-β2)
      • p.addcdiv_(m, denom)              →  p.add_(m / denom, alpha=−lr/bc1)

    Args:
        params       : paramètres du modèle (model.parameters())
        lr           : taux d'apprentissage (défaut 1e-3)
        betas        : coefficients de déclin (β1, β2) (défaut (0.9, 0.999))
        eps          : stabilité numérique (défaut 1e-8)
        weight_decay : régularisation L2 (défaut 0, non utilisé dans MKAN)
    """

    def __init__(self, params, lr: float = 1e-3,
                 betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr           = group['lr']
            beta1, beta2 = group['betas']
            eps          = group['eps']
            wd           = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad  = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step']       = 0
                    state['exp_avg']    = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                state['step'] += 1
                t          = state['step']
                exp_avg    = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                if wd != 0.0:
                    grad = grad.add(p, alpha=wd)

                # 1er moment : m_t = β1·m_{t-1} + (1-β1)·g_t
                # lerp_(grad, 1-β1) → mul_(β1).add_(grad, alpha=1-β1)
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                # 2e moment : v_t = β2·v_{t-1} + (1-β2)·g_t²
                # addcmul_(grad, grad, value=1-β2) → mul_(β2).add_(g², alpha=1-β2)
                exp_avg_sq.mul_(beta2).add_(grad.square(), alpha=1.0 - beta2)

                # Correction du biais (scalaires Python, aucun tenseur alloué)
                bc1 = 1.0 - beta1 ** t
                bc2 = 1.0 - beta2 ** t

                # Mise à jour : p -= (lr/bc1) · m_t / (√(v_t)/√bc2 + ε)
                # addcdiv_ remplacé par / + add_ (addcdiv non garanti DirectML)
                # reciprocal_ in-place : 1 mul au lieu d'1 div tensorielle
                denom = exp_avg_sq.sqrt().mul_(1.0 / (bc2 ** 0.5)).add_(eps).reciprocal_()
                p.add_(exp_avg * denom, alpha=-(lr / bc1))

        return loss
