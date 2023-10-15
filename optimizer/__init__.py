import jax

from .main import CompositeMinimization

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
