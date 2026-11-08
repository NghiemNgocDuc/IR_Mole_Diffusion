"""Molecular coordinate diffusion, with lazy imports for standalone data tools."""

from importlib import import_module

_EXPORTS = {
    'SinusoidalTimeEmb': ('layers', 'SinusoidalTimeEmb'),
    'StableEGNNLayer': ('layers', 'StableEGNNLayer'),
    'EquivariantDenoiser': ('denoiser', 'EquivariantDenoiser'),
    'PhysicsLoss': ('physics', 'PhysicsLoss'),
    'DEFAULT_BOND_LENGTHS': ('physics', 'DEFAULT_BOND_LENGTHS'),
    'VDW_RADII': ('physics', 'VDW_RADII'),
    'EquivariantDiffusion3D_v4': ('model', 'EquivariantDiffusion3D_v4'),
    'SpectrumTokenEncoder': ('encoder', 'SpectrumTokenEncoder'),
    'SpectrumStructureModel': ('spectrum', 'SpectrumStructureModel'),
    'cosine_beta_schedule': ('utils', 'cosine_beta_schedule'),
    'masked_center': ('utils', 'masked_center'),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(f'diffusion.{module}'), attribute)
    globals()[name] = value
    return value
