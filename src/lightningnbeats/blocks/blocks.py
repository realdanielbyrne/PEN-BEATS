import torch
from torch import nn
from ..constants import ACTIVATIONS
import numpy as np
import pywt
from pywt import Wavelet as PyWavelet  # type: ignore[attr-defined]

def squeeze_last_dim(tensor):
  if len(tensor.shape) == 3 and tensor.shape[-1] == 1:  # (128, 10, 1) => (128, 10).
      return tensor[..., 0]
  return tensor
class RootBlock(nn.Module):
  def __init__(self, backcast_length, units, activation='ReLU'):
    """The Block class is the basic building block of the N-BEATS network.
    It consists of a stack of fully connected layers.It serves as the base
    class for the GenericBlock, SeasonalityBlock, and TrendBlock classes.

    Args:
        backcast_length (int):
          The length of the historical data.  It is customary to use a
          multiple of the forecast_length (H)orizon (2H,3H,4H,5H,...).
        units (int):
          The width of the fully connected layers in the blocks comprising
          the stacks of the generic model
        activation (str, optional):
          The activation function applied to each of the fully connected
          Linear layers. Defaults to 'ReLU'.

    Raises:
          ValueError: If the activation function is not in ACTIVATIONS.
    """
    super(RootBlock, self).__init__()
    self.units = units
    self.backcast_length = backcast_length

    if activation not in ACTIVATIONS:
      raise ValueError(f"'{activation}' is not in {ACTIVATIONS}")

    self.activation = getattr(nn, activation)()

    self.fc1 = nn.Linear(backcast_length, units)
    self.fc2 = nn.Linear(units, units)
    self.fc3 = nn.Linear(units, units)
    self.fc4 = nn.Linear(units, units)

  def forward(self, x):
    x = self.activation(self.fc1(x))
    x = self.activation(self.fc2(x))
    x = self.activation(self.fc3(x))
    x = self.activation(self.fc4(x))
    return x

class AutoEncoder(RootBlock):
  def __init__(self,
                units:int,
                backcast_length:int,
                forecast_length:int,
                thetas_dim:int,
                share_weights:bool,
                activation:str = 'ReLU',
                active_g:bool = False):

      super(AutoEncoder, self).__init__(backcast_length, units, activation)

      self.units = units
      self.thetas_dim = thetas_dim
      self.share_weights = share_weights
      self.activation = getattr(nn, activation)()
      self.backcast_length = backcast_length
      self.forecast_length = forecast_length
      self.active_g = active_g


      # Encoders
      if share_weights:
        self.b_encoder = self.f_encoder = nn.Sequential(
          nn.Linear(units, thetas_dim),
          nn.ReLU(),
        )
      else:
        self.b_encoder = nn.Sequential(
            nn.Linear(units, thetas_dim),
            nn.ReLU(),
        )
        self.f_encoder = nn.Sequential(
            nn.Linear(units, thetas_dim),
            nn.ReLU(),
        )

      self.b_decoder = nn.Sequential(
          nn.Linear(thetas_dim, units),
          nn.ReLU(),
          nn.Linear(units, backcast_length),
      )
      self.f_decoder = nn.Sequential(
          nn.Linear(thetas_dim, units),
          nn.ReLU(),
          nn.Linear(units, forecast_length),
      )

  def forward(self, x):
      x = super(AutoEncoder, self).forward(x)
      b = self.b_encoder(x)
      b = self.b_decoder(b)

      f = self.f_encoder(x)
      f = self.f_decoder(f)

      # N-BEATS paper does not apply activation here, but Generic models will not converge sometimes without it
      if self.active_g:
        if self.active_g != 'forecast':
          b = self.activation(b)
        if self.active_g != 'backcast':
          f = self.activation(f)

      return b,f


class _VAEMixin:
  """Shared VAE utility methods: reparameterization, KL divergence, and logvar clamping.

  This mixin centralises the three core VAE operations to eliminate code duplication
  across all VAE-type block hierarchies (RootBlock, AERootBlockVAE, VAE2RootBlock).
  """
  LOGVAR_MIN: float = -20.0
  LOGVAR_MAX: float = 4.0

  def _clamp_logvar(self, logvar: torch.Tensor) -> torch.Tensor:
    """Clamp logvar to [LOGVAR_MIN, LOGVAR_MAX] for numerical stability."""
    return torch.clamp(logvar, min=self.LOGVAR_MIN, max=self.LOGVAR_MAX)

  def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Reparameterization trick: z = mu + std * eps during training, z = mu during eval."""
    if self.training:
      std = torch.exp(0.5 * logvar)
      eps = torch.randn_like(std)
      return mu + std * eps
    return mu

  def _kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor, batch_size: int) -> torch.Tensor:
    """KL divergence vs unit Gaussian: -0.5 * sum(1 + logvar - mu^2 - exp(logvar)) / batch_size."""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size


class VAE(_VAEMixin, RootBlock):
  """Variational AutoEncoder block using the RootBlock backbone.

  Alternative to the AutoEncoder block that replaces the deterministic encoder
  bottleneck with a stochastic latent space using the reparameterization trick.
  Each path (backcast and forecast) has its own mu/logvar heads. During training,
  z = mu + std * eps; during eval, z = mu (deterministic). Stores self.kl_loss
  after each forward pass for collection by the training step.

  Args:
      units (int): Width of the fully connected layers.
      backcast_length (int): Length of the historical data.
      forecast_length (int): Length of the forecast horizon.
      thetas_dim (int): Latent space dimensionality for the VAE bottleneck.
      share_weights (bool): If True, backcast and forecast encoders share parameters.
      activation (str, optional): Activation function name. Defaults to 'ReLU'.
      active_g (bool, optional): Apply activation to outputs. Defaults to False.
  """
  def __init__(self,
                units:int,
                backcast_length:int,
                forecast_length:int,
                thetas_dim:int,
                share_weights:bool,
                activation:str = 'ReLU',
                active_g:bool = False):

      super(VAE, self).__init__(backcast_length, units, activation)

      self.units = units
      self.thetas_dim = thetas_dim
      self.share_weights = share_weights
      self.activation = getattr(nn, activation)()
      self.backcast_length = backcast_length
      self.forecast_length = forecast_length
      self.active_g = active_g

      # VAE Encoders — mu and logvar heads instead of deterministic bottleneck
      if share_weights:
        self.b_mu = self.f_mu = nn.Linear(units, thetas_dim)
        self.b_logvar = self.f_logvar = nn.Linear(units, thetas_dim)
      else:
        self.b_mu = nn.Linear(units, thetas_dim)
        self.b_logvar = nn.Linear(units, thetas_dim)
        self.f_mu = nn.Linear(units, thetas_dim)
        self.f_logvar = nn.Linear(units, thetas_dim)

      self.b_decoder = nn.Sequential(
          nn.Linear(thetas_dim, units),
          nn.ReLU(),
          nn.Linear(units, backcast_length),
      )
      self.f_decoder = nn.Sequential(
          nn.Linear(thetas_dim, units),
          nn.ReLU(),
          nn.Linear(units, forecast_length),
      )

      # KL divergence loss stored after each forward pass
      self.kl_loss = torch.tensor(0.0)

  def forward(self, x):
      x = super(VAE, self).forward(x)
      batch_size = x.shape[0]

      # Backcast path
      b_mu = self.b_mu(x)
      b_logvar = self._clamp_logvar(self.b_logvar(x))
      b_z = self._reparameterize(b_mu, b_logvar)
      b = self.b_decoder(b_z)

      # Forecast path
      f_mu = self.f_mu(x)
      f_logvar = self._clamp_logvar(self.f_logvar(x))
      f_z = self._reparameterize(f_mu, f_logvar)
      f = self.f_decoder(f_z)

      # Store combined KL loss from both paths
      self.kl_loss = (self._kl_divergence(b_mu, b_logvar, batch_size) +
                      self._kl_divergence(f_mu, f_logvar, batch_size))

      if self.active_g:
        if self.active_g != 'forecast':
          b = self.activation(b)
        if self.active_g != 'backcast':
          f = self.activation(f)

      return b, f


class Generic(RootBlock):
  def __init__(self,
               units:int,
               backcast_length:int,
               forecast_length:int,
               thetas_dim:int = 5,
               share_weights:bool= False,
               activation:str = 'ReLU',
               active_g:bool = False):
    """Paper-faithful Generic Block as defined in Oreshkin et al. (2019).

    Uses a single linear layer producing theta of size (backcast_length + forecast_length),
    then slices into backcast and forecast components. This matches the paper's formulation
    exactly: 4 FC+ReLU layers followed by one linear projection, with no intermediate
    bottleneck dimension.

    Args:
        units (int):
          The width of the fully connected layers.
        backcast_length (int):
          The length of the historical data.
        forecast_length (int):
          The length of the forecast horizon.
        thetas_dim (int, optional):
          Not used in paper-faithful Generic (kept for API compatibility). Defaults to 5.
        share_weights (bool, optional):
          If True, the initial weights of the Linear layers are shared.
          Defaults to False.
        activation (str, optional):
          The activation function used in the FC layers. Defaults to 'ReLU'.
        active_g (bool, optional):
          If True, applies activation after the basis expansion.
          Not a feature in the original paper. Defaults to False.
    """
    super(Generic, self).__init__(backcast_length, units, activation)

    self.backcast_length = backcast_length
    self.forecast_length = forecast_length
    self.active_g = active_g

    self.theta_b_fc = nn.Linear(units, backcast_length, bias=False)
    self.theta_f_fc = nn.Linear(units, forecast_length, bias=False)

  def forward(self, x):
    x = super(Generic, self).forward(x)

    backcast = self.theta_b_fc(x)
    forecast = self.theta_f_fc(x)

    if self.active_g:
      if self.active_g != 'forecast':
        backcast = self.activation(backcast)
      if self.active_g != 'backcast':
        forecast = self.activation(forecast)

    return backcast, forecast

class BottleneckGeneric(RootBlock):
  def __init__(self,
               units:int,
               backcast_length:int,
               forecast_length:int,
               thetas_dim:int = 5,
               share_weights:bool= False,
               activation:str = 'ReLU',
               active_g:bool = False):
    """Bottleneck Generic Block — a novel extension of the paper's Generic block.

    Uses a two-stage projection through an intermediate thetas_dim bottleneck:
    units → thetas_dim → target_length. This is equivalent to a rank-d factorization
    of the basis expansion matrix, where d = thetas_dim, providing a tunable knob
    to control basis complexity.

    Args:
        units (int):
          The width of the fully connected layers in the blocks comprising
          the stacks of the generic model.
        backcast_length (int):
          The length of the historical data.
        forecast_length (int):
          The length of the forecast horizon.
        thetas_dim (int, optional):
          The dimensionality of the bottleneck (rank of factorized basis).
          Defaults to 5.
        share_weights (bool, optional):
          If True, the initial weights of the Linear layers are shared.
          Defaults to False.
        activation (str, optional):
          The activation function used in the parent class Block, and
          optionally as the non-linear activation of the backcast_g and
          forecast_g layers. Defaults to 'ReLU'.
        active_g (bool, optional):
          If True, applies activation after the basis expansion.
          Not a feature in the original paper. Defaults to False.
    """
    super(BottleneckGeneric, self).__init__(backcast_length, units, activation)

    if share_weights:
        self.backcast_linear = self.forecast_linear = nn.Linear(units, thetas_dim)
    else:
        self.backcast_linear = nn.Linear(units, thetas_dim)
        self.forecast_linear = nn.Linear(units, thetas_dim)

    self.backcast_g = nn.Linear(thetas_dim, backcast_length, bias = False)
    self.forecast_g = nn.Linear(thetas_dim, forecast_length, bias = False)
    self.active_g = active_g

  def forward(self, x):
    x = super(BottleneckGeneric, self).forward(x)
    theta_b = self.backcast_linear(x)
    theta_f = self.forecast_linear(x)

    backcast = self.backcast_g(theta_b)
    forecast = self.forecast_g(theta_f)

    if self.active_g:
      if self.active_g != 'forecast':
        backcast = self.activation(backcast)
      if self.active_g != 'backcast':
        forecast = self.activation(forecast)

    return backcast, forecast

class GenericAEBackcast(RootBlock):
  def __init__(self,
                units:int,
                backcast_length:int,
                forecast_length:int,
                thetas_dim:int,
                share_weights:bool,
                activation:str = 'ReLU',
                active_g:bool = False):

      super(GenericAEBackcast, self).__init__(backcast_length, units, activation)

      self.units = units
      self.thetas_dim = thetas_dim
      self.share_weights = share_weights
      self.activation = getattr(nn, activation)()
      self.backcast_length = backcast_length
      self.forecast_length = forecast_length
      self.active_g = active_g

      self.forecast_linear = nn.Linear(units, self.thetas_dim)
      self.forecast_g = nn.Linear(self.thetas_dim, forecast_length, bias = False)

      self.b_encoder = nn.Linear(units, self.thetas_dim)
      self.b_decoder = nn.Linear(self.thetas_dim, units)
      self.backcast_g = nn.Linear(units, backcast_length)

  def forward(self, x):
    x = super(GenericAEBackcast, self).forward(x)
    b = self.activation(self.b_encoder(x))
    b = self.activation(self.b_decoder(b))
    b = self.backcast_g(b)

    theta_f = self.forecast_linear(x)
    f = self.forecast_g(theta_f)

    # N-BEATS paper does not apply activation here, but Generic models will not converge sometimes without it
    if self.active_g:
      if self.active_g != 'forecast':
        b = self.activation(b)
      if self.active_g != 'backcast':
        f = self.activation(f)
    return b, f

class _SeasonalityGenerator(nn.Module):
  def __init__(self, len):
    """Generates the basis for the Fourier basic expansion of the Seasonality Block.

    Args:
        forecast_length (int): The length of the forecast_length horizon.
    """
    super().__init__()
    half_minus_one = int(len / 2 - 1)
    cos_vectors = [
        torch.cos(torch.arange(len) / len * 2 * np.pi * i)
        for i in range(1, half_minus_one + 1)
    ]
    sin_vectors = [
        torch.sin(torch.arange(len) / len * 2 * np.pi * i)
        for i in range(1, half_minus_one + 1)
    ]

    # basis is of size (2 * int(forecast_length / 2 - 1) + 1, forecast_length)
    basis = torch.stack(
        [torch.ones(len)] + cos_vectors + sin_vectors, dim=1
    ).T

    self.basis = nn.Parameter(basis, requires_grad=False)

  def forward(self, x):
    return torch.matmul(x, self.basis)

class Seasonality(RootBlock):
  def __init__(self, units, backcast_length, forecast_length,  thetas_dim=5,
               share_weights = False, activation='ReLU', active_g:bool = False):

    super(Seasonality, self).__init__(backcast_length, units, activation)

    self.backcast_linear = nn.Linear(units, 2 * int(backcast_length / 2 - 1) + 1, bias = False)
    self.forecast_linear = nn.Linear(units, 2 * int(forecast_length / 2 - 1) + 1, bias = False)

    self.backcast_g = _SeasonalityGenerator(backcast_length)
    self.forecast_g = _SeasonalityGenerator(forecast_length)

  def forward(self, x):
    x = super(Seasonality, self).forward(x)
    # linear compression
    backcast_thetas = self.backcast_linear(x)
    forecast_thetas = self.forecast_linear(x)

    # fourier expansion
    backcast = self.backcast_g(backcast_thetas)
    forecast = self.forecast_g(forecast_thetas)

    return backcast, forecast

class _TrendGenerator(nn.Module):
  def __init__(self, thetas_dim, target_length):
    """ Trend model. A typical characteristic of trend is that most of the time it is a
    monotonic function, or at least a slowly varying function. In order to mimic this
    behaviour this block implements a low pass filter, or slowly varying function of a
    small degree polynomial across forecast_length window.

    Args:
        expansion_coefficient_dim (int): The dimensionality of the expansion coefficients used in
        the Vandermonde expansion.  The N-BEATS paper uses 2 or 3, but any positive integer can be used.
        5 is also a reasonalbe choice.
        target_length (int): The length of the forecast_length horizon.
    """
    super().__init__()

    # basis is of size (expansion_coefficient_dim, target_length)
    basis = torch.stack(
        [
            (torch.arange(target_length) / target_length) ** i
            for i in range(thetas_dim)
        ],
        dim=1,
    ).T

    self.basis = nn.Parameter(basis, requires_grad=False)

  def forward(self, x):
    return torch.matmul(x, self.basis)

class Trend(RootBlock):
  def __init__(self, units, backcast_length, forecast_length, thetas_dim,
               share_weights = False, activation='ReLU', active_g:bool = False):
    """The Trend Block implements the function whose parameters are generated by the _TrendGenerator block.

    Args:
        units (int):
          The width of the fully connected layers in the blocks comprising the parent class Block.
        backcast_length (int):
          The length of the historical data.  It is customary to use a multiple of the forecast_length (H)orizon (2H,3H,4H,5H,...).
        forecast_length (int):
          The length of the forecast_length horizon.
        thetas_dim (int):
          The dimensionality of the _TrendGenerator polynomial.
        share_weights (bool, optional):
          If True, the inital weights of the Linear layers are shared. Defaults to False.
        activation (str, optional):
          The activation function passed to the parent class Block. Defaults to 'ReLU'.
    """
    super(Trend, self).__init__(backcast_length, units, activation)
    self.share_weights = share_weights
    if self.share_weights:
        self.backcast_linear = self.forecast_linear = nn.Linear(units, thetas_dim)
    else:
        self.backcast_linear = nn.Linear(units, thetas_dim)
        self.forecast_linear = nn.Linear(units, thetas_dim)

    self.backcast_g = _TrendGenerator(thetas_dim, backcast_length)
    self.forecast_g = _TrendGenerator(thetas_dim, forecast_length)

  def forward(self, x):
    x = super(Trend, self).forward(x)

    # linear compression
    backcast_thetas = self.backcast_linear(x)
    forecast_thetas = self.forecast_linear(x)

    #  Vandermonde basis expansion
    backcast = self.backcast_g(backcast_thetas)
    forecast = self.forecast_g(forecast_thetas)

    return backcast, forecast


# ---------------------------------------------------------------------------
# AutoEncoderRoot Blocks
# ---------------------------------------------------------------------------
class AERootBlock(nn.Module):
  def __init__(self, backcast_length, units, activation='ReLU', latent_dim=5):
    """The AERootBlock class is the basic building block of the N-BEATS network.
    It consists of a stack of fully connected layers organized as an Autoencoder.
    It serves as the base class for the GenericAEBlock, SeasonalityAEBlock,
    and the TrendAEBlock classes.

    Args:
        backcast_length (int):
          The length of the historical data.  It is customary to use a multiple
          of the forecast_length (H)orizon (2H,3H,4H,5H,...).
        units (int):
          The width of the fully connected layers in the blocks comprising the
          stacks of the generic model.
        activation (str, optional):
          The activation function applied to each of the fully connected Linear
          layers. Defaults to 'ReLU'.
        latent_dim (int, optional):
          The dimensionality of the latent space. Defaults to 5.

    Raises:
          ValueError: If the activation function is not in ACTIVATIONS.
    """
    super(AERootBlock, self).__init__()
    self.units = units
    self.backcast_length = backcast_length
    self.latent_dim = latent_dim

    if not activation in ACTIVATIONS:
      raise ValueError(f"'{activation}' is not in {ACTIVATIONS}")

    self.activation = getattr(nn, activation)()

    self.fc1 = nn.Linear(backcast_length, units//2)
    self.fc2 = nn.Linear(units//2, latent_dim)
    self.fc3 = nn.Linear(latent_dim, units//2)
    self.fc4 = nn.Linear(units//2, units)


  def forward(self, x):
    x = squeeze_last_dim(x)
    x = self.activation(self.fc1(x))
    x = self.activation(self.fc2(x))
    x = self.activation(self.fc3(x))
    x = self.activation(self.fc4(x))
    return x

class AERootBlockLG(nn.Module):
  def __init__(self, backcast_length, units, activation='ReLU', latent_dim=5):
    """Learned-Gate AE backbone. Same encoder-decoder structure as AERootBlock
    but adds a learnable gate vector that applies sigmoid-scaled per-dimension
    masking at the latent bottleneck, allowing the network to discover the
    effective latent dimensionality during training.

    Args:
        backcast_length (int): Length of the historical data.
        units (int): Width of the fully connected layers.
        activation (str, optional): Activation function name. Defaults to 'ReLU'.
        latent_dim (int, optional): Maximum latent dimensionality. Defaults to 5.
    """
    super(AERootBlockLG, self).__init__()
    self.units = units
    self.backcast_length = backcast_length
    self.latent_dim = latent_dim

    if activation not in ACTIVATIONS:
      raise ValueError(f"'{activation}' is not in {ACTIVATIONS}")

    self.activation = getattr(nn, activation)()

    self.fc1 = nn.Linear(backcast_length, units // 2)
    self.fc2 = nn.Linear(units // 2, latent_dim)
    self.fc3 = nn.Linear(latent_dim, units // 2)
    self.fc4 = nn.Linear(units // 2, units)

    # Learned gate: sigmoid(latent_gate) scales each latent dimension
    self.latent_gate = nn.Parameter(torch.ones(latent_dim))

  def forward(self, x):
    x = squeeze_last_dim(x)
    x = self.activation(self.fc1(x))
    x = self.activation(self.fc2(x))
    x = x * torch.sigmoid(self.latent_gate)
    x = self.activation(self.fc3(x))
    x = self.activation(self.fc4(x))
    return x


class AERootBlockVAE(_VAEMixin, nn.Module):
  def __init__(self, backcast_length, units, activation='ReLU', latent_dim=5):
    """Variational AE backbone. Replaces the deterministic bottleneck with a
    stochastic latent space using the reparameterization trick. Produces mu and
    log_var heads, samples z = mu + sigma * epsilon during training, and stores
    self.kl_loss for collection by the training step.

    Args:
        backcast_length (int): Length of the historical data.
        units (int): Width of the fully connected layers.
        activation (str, optional): Activation function name. Defaults to 'ReLU'.
        latent_dim (int, optional): Latent space dimensionality. Defaults to 5.
    """
    super(AERootBlockVAE, self).__init__()
    self.units = units
    self.backcast_length = backcast_length
    self.latent_dim = latent_dim

    if activation not in ACTIVATIONS:
      raise ValueError(f"'{activation}' is not in {ACTIVATIONS}")

    self.activation = getattr(nn, activation)()

    self.fc1 = nn.Linear(backcast_length, units // 2)
    self.fc1b = nn.Linear(units // 2, units // 4)
    self.fc2_mu = nn.Linear(units // 4, latent_dim)
    self.fc2_logvar = nn.Linear(units // 4, latent_dim)
    self.fc3 = nn.Linear(latent_dim, units // 4)
    self.fc3b = nn.Linear(units // 4, units // 2)
    self.fc4 = nn.Linear(units // 2, units)

    # KL divergence loss stored after each forward pass
    self.kl_loss = torch.tensor(0.0)

  def forward(self, x):
    x = squeeze_last_dim(x)
    x = self.activation(self.fc1(x))
    x = self.activation(self.fc1b(x))

    mu = self.fc2_mu(x)
    logvar = self._clamp_logvar(self.fc2_logvar(x))

    z = self._reparameterize(mu, logvar)
    self.kl_loss = self._kl_divergence(mu, logvar, x.shape[0])

    x = self.activation(self.fc3(z))
    x = self.activation(self.fc3b(x))
    x = self.activation(self.fc4(x))
    return x


# ---------------------------------------------------------------------------
# VAE2RootBlock — compact VAE backbone and derivative blocks
# Architecture: backcast_length → units → units//2 → latent_dim*2
#               (torch.chunk → mu, logvar) → latent_dim (z) → units//2 → units
# ---------------------------------------------------------------------------

class VAE2RootBlock(_VAEMixin, nn.Module):
  """Compact VAE backbone with a 3-layer encoder and 2-layer decoder.

  Simpler than AERootBlockVAE (which steps down to units//4 before the bottleneck).
  This block steps down to units//2, then uses a single combined linear head that
  outputs latent_dim * 2, split via torch.chunk into mu and logvar of size latent_dim
  each. The decoder mirrors the encoder back up through units//2 to units.

  Architecture:
    Encoder: backcast_length → units → units//2 → latent_dim*2 (→ mu, logvar)
    Latent:  z = reparameterize(mu, logvar)
    Decoder: z (latent_dim) → units//2 → units

  Stores self.kl_loss after each forward pass for collection by the training step.

  Args:
      backcast_length (int): Length of the historical input sequence.
      units (int): Width of the fully connected layers.
      activation (str, optional): Activation function name. Defaults to 'ReLU'.
      latent_dim (int, optional): Effective latent dimension (each of mu/logvar has
          this size). The encoder's combined head outputs latent_dim*2. Defaults to 5.
  """
  def __init__(self, backcast_length, units, activation='ReLU', latent_dim=5):
    super(VAE2RootBlock, self).__init__()
    self.units = units
    self.backcast_length = backcast_length
    self.latent_dim = latent_dim

    if activation not in ACTIVATIONS:
      raise ValueError(f"'{activation}' is not in {ACTIVATIONS}")

    self.activation = getattr(nn, activation)()

    # Encoder layers
    self.fc1 = nn.Linear(backcast_length, units)
    self.fc2 = nn.Linear(units, units // 2)
    self.fc3 = nn.Linear(units // 2, latent_dim * 2)   # outputs mu || logvar

    # Decoder layers
    self.fc4 = nn.Linear(latent_dim, units // 2)
    self.fc5 = nn.Linear(units // 2, units)

    # KL divergence loss stored after each forward pass
    self.kl_loss = torch.tensor(0.0)

  def forward(self, x):
    x = squeeze_last_dim(x)
    x = self.activation(self.fc1(x))
    x = self.activation(self.fc2(x))

    combined = self.fc3(x)
    mu, logvar = torch.chunk(combined, 2, dim=-1)
    logvar = self._clamp_logvar(logvar)

    z = self._reparameterize(mu, logvar)
    self.kl_loss = self._kl_divergence(mu, logvar, x.shape[0])

    x = self.activation(self.fc4(z))
    x = self.activation(self.fc5(x))
    return x


class GenericVAE2(VAE2RootBlock):
  """Paper-faithful Generic Block with VAE2RootBlock (compact VAE) backbone.

  Uses direct linear projections from units to target lengths (no intermediate
  thetas_dim bottleneck), matching the paper's Generic formulation.

  Args:
      units (int): Width of the fully connected layers.
      backcast_length (int): Length of the historical data.
      forecast_length (int): Length of the forecast horizon.
      thetas_dim (int, optional): Not used (kept for API compatibility). Defaults to 5.
      share_weights (bool, optional): Defaults to False.
      activation (str, optional): Defaults to 'ReLU'.
      active_g (bool, optional): Apply activation after projection. Defaults to False.
      latent_dim (int, optional): Effective latent dimension for the VAE2 backbone. Defaults to 5.
  """
  def __init__(self, units, backcast_length, forecast_length, thetas_dim=5,
               share_weights=False, activation='ReLU', active_g=False, latent_dim=5):
    super(GenericVAE2, self).__init__(backcast_length, units, activation, latent_dim=latent_dim)
    self.backcast_length = backcast_length
    self.forecast_length = forecast_length
    self.active_g = active_g

    self.theta_b_fc = nn.Linear(units, backcast_length, bias=False)
    self.theta_f_fc = nn.Linear(units, forecast_length, bias=False)

  def forward(self, x):
    x = super(GenericVAE2, self).forward(x)
    backcast = self.theta_b_fc(x)
    forecast = self.theta_f_fc(x)
    if self.active_g:
      if self.active_g != 'forecast':
        backcast = self.activation(backcast)
      if self.active_g != 'backcast':
        forecast = self.activation(forecast)
    return backcast, forecast


class TrendVAE2(VAE2RootBlock):
  """Trend Block with VAE2RootBlock (compact VAE) backbone.

  Uses a polynomial basis expansion (_TrendGenerator) for both backcast and
  forecast paths on top of the shared VAE2 trunk.

  Args:
      units (int): Width of the fully connected layers.
      backcast_length (int): Length of the historical data.
      forecast_length (int): Length of the forecast horizon.
      thetas_dim (int): Polynomial degree for the trend generator.
      share_weights (bool, optional): Share projection heads. Defaults to False.
      activation (str, optional): Defaults to 'ReLU'.
      active_g (bool, optional): Defaults to False.
      latent_dim (int, optional): Effective latent dimension. Defaults to 5.
  """
  def __init__(self, units, backcast_length, forecast_length, thetas_dim=5,
               share_weights=False, activation='ReLU', active_g=False, latent_dim=5):
    super(TrendVAE2, self).__init__(backcast_length, units, activation, latent_dim=latent_dim)
    self.share_weights = share_weights
    if share_weights:
      self.backcast_linear = self.forecast_linear = nn.Linear(units, thetas_dim)
    else:
      self.backcast_linear = nn.Linear(units, thetas_dim)
      self.forecast_linear = nn.Linear(units, thetas_dim)
    self.backcast_g = _TrendGenerator(thetas_dim, backcast_length)
    self.forecast_g = _TrendGenerator(thetas_dim, forecast_length)

  def forward(self, x):
    x = super(TrendVAE2, self).forward(x)
    backcast = self.backcast_g(self.backcast_linear(x))
    forecast = self.forecast_g(self.forecast_linear(x))
    return backcast, forecast


class SeasonalityVAE2(VAE2RootBlock):
  """Seasonality Block with VAE2RootBlock (compact VAE) backbone.

  Uses a Fourier basis expansion (_SeasonalityGenerator) for both backcast and
  forecast paths on top of the shared VAE2 trunk.

  Args:
      units (int): Width of the fully connected layers.
      backcast_length (int): Length of the historical data.
      forecast_length (int): Length of the forecast horizon.
      thetas_dim (int, optional): Not used (kept for API compatibility). Defaults to 5.
      share_weights (bool, optional): Defaults to False.
      activation (str, optional): Defaults to 'ReLU'.
      active_g (bool, optional): Defaults to False.
      latent_dim (int, optional): Effective latent dimension. Defaults to 5.
  """
  def __init__(self, units, backcast_length, forecast_length, thetas_dim=5,
               share_weights=False, activation='ReLU', active_g=False, latent_dim=5):
    super(SeasonalityVAE2, self).__init__(backcast_length, units, activation, latent_dim=latent_dim)

    self.backcast_linear = nn.Linear(units, 2 * int(backcast_length / 2 - 1) + 1, bias=False)
    self.forecast_linear = nn.Linear(units, 2 * int(forecast_length / 2 - 1) + 1, bias=False)
    self.backcast_g = _SeasonalityGenerator(backcast_length)
    self.forecast_g = _SeasonalityGenerator(forecast_length)

  def forward(self, x):
    x = super(SeasonalityVAE2, self).forward(x)
    backcast = self.backcast_g(self.backcast_linear(x))
    forecast = self.forecast_g(self.forecast_linear(x))
    return backcast, forecast


class VAE2(VAE2RootBlock):
  """AutoEncoder-style block using the VAE2RootBlock (compact VAE) backbone.

  Analogous to AutoEncoderVAE (which uses AERootBlockVAE backbone) but built on
  the more compact VAE2 architecture.  Adds separate encoder-decoder paths for
  backcast and forecast on top of the shared VAE2 trunk, providing an additional
  learned compression per output direction.

  Args:
      units (int): Width of the fully connected layers.
      backcast_length (int): Length of the historical data.
      forecast_length (int): Length of the forecast horizon.
      thetas_dim (int, optional): Bottleneck dimension for the per-path encoder-decoder.
          Defaults to 5.
      share_weights (bool, optional): Share backcast/forecast encoders. Defaults to False.
      activation (str, optional): Defaults to 'ReLU'.
      active_g (bool, optional): Apply activation after decoding. Defaults to False.
      latent_dim (int, optional): Effective latent dimension for the VAE2 backbone. Defaults to 5.
  """
  def __init__(self, units, backcast_length, forecast_length, thetas_dim=5,
               share_weights=False, activation='ReLU', active_g=False, latent_dim=5):
    super(VAE2, self).__init__(backcast_length, units, activation, latent_dim=latent_dim)
    self.active_g = active_g

    if share_weights:
      self.b_encoder = self.f_encoder = nn.Sequential(
          nn.Linear(units, thetas_dim),
          getattr(nn, activation)(),
      )
    else:
      self.b_encoder = nn.Sequential(
          nn.Linear(units, thetas_dim),
          getattr(nn, activation)(),
      )
      self.f_encoder = nn.Sequential(
          nn.Linear(units, thetas_dim),
          getattr(nn, activation)(),
      )

    self.b_decoder = nn.Sequential(
        nn.Linear(thetas_dim, units),
        getattr(nn, activation)(),
        nn.Linear(units, backcast_length),
    )
    self.f_decoder = nn.Sequential(
        nn.Linear(thetas_dim, units),
        getattr(nn, activation)(),
        nn.Linear(units, forecast_length),
    )

  def forward(self, x):
    x = super(VAE2, self).forward(x)
    b = self.b_decoder(self.b_encoder(x))
    f = self.f_decoder(self.f_encoder(x))
    if self.active_g:
      if self.active_g != 'forecast':
        b = self.activation(b)
      if self.active_g != 'backcast':
        f = self.activation(f)
    return b, f


class GenericAEBackcastAE(AERootBlock):
  def __init__(self,
                units:int,
                backcast_length:int,
                forecast_length:int,
                thetas_dim:int,
                share_weights:bool,
                activation:str = 'ReLU',
                active_g:bool = False,
                latent_dim:int = 5):

      super(GenericAEBackcastAE, self).__init__(backcast_length, units, activation,latent_dim=latent_dim)

      self.units = units
      self.thetas_dim = thetas_dim
      self.share_weights = share_weights
      self.activation = getattr(nn, activation)()
      self.backcast_length = backcast_length
      self.forecast_length = forecast_length
      self.active_g = active_g
      branch_latent_dim = latent_dim

      self.forecast_linear = nn.Linear(units, thetas_dim)
      self.forecast_g = nn.Linear(thetas_dim, forecast_length, bias = False)

      self.b_encoder = nn.Linear(units, branch_latent_dim)
      self.b_decoder = nn.Linear(branch_latent_dim, units)
      self.backcast_g = nn.Linear(units, backcast_length)

  def forward(self, x):
    x = super(GenericAEBackcastAE, self).forward(x)
    b = self.activation(self.b_encoder(x))
    b = self.activation(self.b_decoder(b))
    b = self.backcast_g(b)

    theta_f = self.forecast_linear(x)
    f = self.forecast_g(theta_f)

    # N-BEATS paper does not apply activation here;
    # however Generic models will not always converge without it
    if self.active_g:
      if self.active_g != 'forecast':
        b = self.activation(b)
      if self.active_g != 'backcast':
        f = self.activation(f)
    return b,f

class AutoEncoderAE(AERootBlock):
  def __init__(self,
                units:int,
                backcast_length:int,
                forecast_length:int,
                thetas_dim:int,
                share_weights:bool,
                activation:str = 'ReLU',
                active_g:bool = False,
                latent_dim:int = 5):
    """AutoEncoder block with an AE backbone.

    Args:
        units (int): The number of inoput and output units
        backcast_length (int): The length of the historical data.
        forecast_length (int): The length of the forecast_length horizon.
        thetas_dim (int): Unused. Kept for API compatibility with other block types.
        share_weights (bool): The weights of the encoder are shared if True.
        activation (str, optional): Activation function name. Defaults to 'ReLU'.
        active_g (bool, optional): Apply activation after decoding. Defaults to False.
        latent_dim (int, optional): Branch-local AE bottleneck size. Defaults to 5.
    """

    super(AutoEncoderAE, self).__init__(backcast_length, units, activation, latent_dim=latent_dim)

    self.units = units
    self.thetas_dim = thetas_dim
    self.share_weights = share_weights
    self.activation = getattr(nn, activation)()
    self.backcast_length = backcast_length
    self.forecast_length = forecast_length
    self.active_g = active_g
    branch_latent_dim = latent_dim

    if share_weights:
      self.b_encoder = self.f_encoder = nn.Linear(units, branch_latent_dim)
    else:
      self.b_encoder = nn.Linear(units, branch_latent_dim)
      self.f_encoder = nn.Linear(units, branch_latent_dim)

    self.b_decoder = nn.Linear(branch_latent_dim, units)
    self.f_decoder = nn.Linear(branch_latent_dim, units)
    self.backcast_g = nn.Linear(units, backcast_length)
    self.forecast_g = nn.Linear(units, forecast_length)

  def forward(self, x):
    x = super(AutoEncoderAE, self).forward(x)
    b = self.activation(self.b_encoder(x))
    b = self.activation(self.b_decoder(b))
    b = self.backcast_g(b)

    f = self.activation(self.f_encoder(x))
    f = self.activation(self.f_decoder(f))
    f = self.forecast_g(f)

    # N-BEATS paper does not apply activation here, but Generic models will not converge sometimes without it
    if self.active_g:
      if self.active_g != 'forecast':
        b = self.activation(b)
      if self.active_g != 'backcast':
        f = self.activation(f)

    return b,f

class GenericAE(AERootBlock):
  def __init__(self,
               units:int,
               backcast_length:int,
               forecast_length:int,
               thetas_dim:int = 5,
               share_weights:bool= False,
               activation:str = 'ReLU',
               active_g:bool = False,
               latent_dim:int = 5):
    """Paper-faithful Generic Block with AERootBlock (bottleneck pre-split) backbone.

    Uses direct linear projections from units to target lengths (no intermediate
    thetas_dim bottleneck), matching the paper's Generic formulation.

    Args:
        units (int): Width of the fully connected layers.
        backcast_length (int): Length of the historical data.
        forecast_length (int): Length of the forecast horizon.
        thetas_dim (int, optional): Not used (kept for API compatibility). Defaults to 5.
        share_weights (bool, optional): Defaults to False.
        activation (str, optional): Defaults to 'ReLU'.
        active_g (bool, optional): If True, applies activation after projection. Defaults to False.
        latent_dim (int, optional): Latent dimension for the AE backbone. Defaults to 5.
    """
    super(GenericAE, self).__init__(backcast_length, units, activation, latent_dim=latent_dim)

    self.backcast_length = backcast_length
    self.forecast_length = forecast_length
    self.active_g = active_g

    self.theta_b_fc = nn.Linear(units, backcast_length, bias=False)
    self.theta_f_fc = nn.Linear(units, forecast_length, bias=False)

  def forward(self, x):
    x = super(GenericAE, self).forward(x)

    backcast = self.theta_b_fc(x)
    forecast = self.theta_f_fc(x)

    if self.active_g:
      if self.active_g != 'forecast':
        backcast = self.activation(backcast)
      if self.active_g != 'backcast':
        forecast = self.activation(forecast)

    return backcast, forecast

class BottleneckGenericAE(AERootBlock):
  def __init__(self,
               units:int,
               backcast_length:int,
               forecast_length:int,
               thetas_dim:int = 5,
               share_weights:bool= False,
               activation:str = 'ReLU',
               active_g:bool = False,
               latent_dim:int = 5):
    """Bottleneck Generic Block with AERootBlock (bottleneck pre-split) backbone.

    Uses a two-stage projection through an intermediate thetas_dim bottleneck,
    equivalent to a rank-d factorization of the basis expansion matrix.

    Args:
        units (int): Width of the fully connected layers.
        backcast_length (int): Length of the historical data.
        forecast_length (int): Length of the forecast horizon.
        thetas_dim (int, optional): Bottleneck dimension. Defaults to 5.
        share_weights (bool, optional): Defaults to False.
        activation (str, optional): Defaults to 'ReLU'.
        active_g (bool, optional): If True, applies activation after expansion. Defaults to False.
        latent_dim (int, optional): Latent dimension for the AE backbone. Defaults to 5.
    """
    super(BottleneckGenericAE, self).__init__(backcast_length, units, activation, latent_dim=latent_dim)

    if share_weights:
        self.backcast_linear = self.forecast_linear = nn.Linear(units, thetas_dim)
    else:
        self.backcast_linear = nn.Linear(units, thetas_dim)
        self.forecast_linear = nn.Linear(units, thetas_dim)

    self.backcast_g = nn.Linear(thetas_dim, backcast_length, bias = False)
    self.forecast_g = nn.Linear(thetas_dim, forecast_length, bias = False)
    self.active_g = active_g

  def forward(self, x):
    x = super(BottleneckGenericAE, self).forward(x)
    theta_b = self.backcast_linear(x)
    theta_f = self.forecast_linear(x)

    backcast = self.backcast_g(theta_b)
    forecast = self.forecast_g(theta_f)

    if self.active_g:
      if self.active_g != 'forecast':
        backcast = self.activation(backcast)
      if self.active_g != 'backcast':
        forecast = self.activation(forecast)

    return backcast, forecast

class TrendAE(AERootBlock):
  def __init__(self, units, backcast_length, forecast_length, thetas_dim,
               share_weights = False, activation='ReLU', active_g:bool = False, latent_dim = 5):
    """The TrendAEBlock implements the function whose parameters are generated by the _TrendGenerator block.
    The TrendAEBlock is an AutoEncoder version of the TrendBlock where the presplit section of the network
    is an AutoEncoder.

    Args:
        units (int):
          The width of the fully connected layers in the blocks comprising the parent class Block.
        backcast_length (int):
          The length of the historical data.  It is customary to use a multiple of the forecast_length (H)orizon (2H,3H,4H,5H,...).
        forecast_length (int):
          The length of the forecast_length horizon.
        thetas_dim (int):
          The dimensionality of the _TrendGenerator polynomial.
        share_weights (bool, optional):
          If True, the inital weights of the Linear layers are shared. Defaults to False.
        activation (str, optional):
          The activation function passed to the parent class Block. Defaults to 'ReLU'.
    """
    super(TrendAE, self).__init__(backcast_length, units, activation, latent_dim=latent_dim)
    if share_weights:
        self.backcast_linear = self.forecast_linear = nn.Linear(units, thetas_dim)
    else:
        self.backcast_linear = nn.Linear(units, thetas_dim)
        self.forecast_linear = nn.Linear(units, thetas_dim)

    self.backcast_g = _TrendGenerator(thetas_dim, backcast_length)
    self.forecast_g = _TrendGenerator(thetas_dim, forecast_length)

  def forward(self, x):
    x = super(TrendAE, self).forward(x)

    # linear compression
    backcast_thetas = self.backcast_linear(x)
    forecast_thetas = self.forecast_linear(x)

    #  Vandermonde basis expansion
    backcast = self.backcast_g(backcast_thetas)
    forecast = self.forecast_g(forecast_thetas)

    return backcast, forecast


class TrendWaveletAE(AERootBlock):
  def __init__(self, units, backcast_length, forecast_length,
               trend_dim=4, wavelet_dim=16, basis_offset=0,
               share_weights=False, activation='ReLU', active_g=False,
               wavelet_type='db3', latent_dim=5, forecast_basis_dim=None):
    """TrendWaveletAE merges polynomial (Vandermonde) and orthonormal DWT basis expansions
    into a single block with an AutoEncoder backbone (Option A — parallel basis, additive output).

    The AERootBlock backbone compresses the input into a latent representation and decodes
    it back to `units` width.  A single linear projection head per path maps the decoded
    representation to concatenated trend + wavelet coefficient vectors, which are split and
    expanded through their respective frozen basis matrices.  The two components are summed
    to produce the final backcast/forecast.

    Args:
        units (int): Width of the fully connected layers in the AE backbone.
        backcast_length (int): Length of the historical input sequence.
        forecast_length (int): Length of the forecast horizon.
        trend_dim (int, optional): Polynomial degree (number of Vandermonde basis vectors).
            Defaults to 4.
        wavelet_dim (int, optional): Number of DWT basis rows. Defaults to 16.
        basis_offset (int, optional): Row offset into SVD-ordered DWT basis for
            frequency-band selection.  0 = lowest/smoothest.  Defaults to 0.
        share_weights (bool, optional): Share the projection head between backcast and
            forecast paths when their total coefficient dimensions match. Defaults to False.
        activation (str, optional): Activation function for the AE backbone. Defaults to 'ReLU'.
        active_g (bool or str, optional): Apply activation after basis expansion.
            False = no activation.  True = both paths.
            'backcast'/'forecast' = single path.  Defaults to False.
        wavelet_type (str, optional): PyWavelets wavelet family string. Defaults to 'db3'.
        latent_dim (int, optional): Dimensionality of the AE bottleneck. Defaults to 5.
        forecast_basis_dim (int, optional): Override for wavelet basis dim on the forecast
            path.  When set the forecast wavelet head uses this value (clamped to
            forecast_length) while the backcast path continues to use wavelet_dim.
            Defaults to None (both paths use wavelet_dim).
    """
    super(TrendWaveletAE, self).__init__(backcast_length, units, activation, latent_dim=latent_dim)
    self.trend_dim = trend_dim
    self.active_g = active_g
    self.wavelet_type = wavelet_type

    # Effective wavelet dimension for each path (clamped to available rows after offset)
    eff_back_wave = min(wavelet_dim, backcast_length - min(basis_offset, backcast_length - 1))
    fore_wavelet_dim = forecast_basis_dim if forecast_basis_dim is not None else wavelet_dim
    eff_fore_wave = min(fore_wavelet_dim, forecast_length - min(basis_offset, forecast_length - 1))

    # Total coefficient dims: trend + wavelet for each path
    total_backcast_dim = trend_dim + eff_back_wave
    total_forecast_dim = trend_dim + eff_fore_wave

    if share_weights and total_backcast_dim == total_forecast_dim:
      self.backcast_linear = self.forecast_linear = nn.Linear(units, total_backcast_dim)
    else:
      self.backcast_linear = nn.Linear(units, total_backcast_dim)
      self.forecast_linear = nn.Linear(units, total_forecast_dim)

    # Frozen polynomial (Vandermonde) basis generators
    self.backcast_trend_g = _TrendGenerator(trend_dim, backcast_length)
    self.forecast_trend_g = _TrendGenerator(trend_dim, forecast_length)

    # Frozen orthonormal DWT basis generators
    self.backcast_wavelet_g = _WaveletGeneratorV3(backcast_length, wavelet_type=wavelet_type,
                                                   basis_dim=eff_back_wave, basis_offset=basis_offset)
    self.forecast_wavelet_g = _WaveletGeneratorV3(forecast_length, wavelet_type=wavelet_type,
                                                   basis_dim=eff_fore_wave, basis_offset=basis_offset)

  def forward(self, x):
    x = super(TrendWaveletAE, self).forward(x)

    # Project AE output to concatenated trend + wavelet coefficient vectors
    backcast_coeffs = self.backcast_linear(x)
    forecast_coeffs = self.forecast_linear(x)

    # Split coefficient vectors into trend and wavelet parts
    backcast_trend_theta = backcast_coeffs[:, :self.trend_dim]
    backcast_wavelet_theta = backcast_coeffs[:, self.trend_dim:]
    forecast_trend_theta = forecast_coeffs[:, :self.trend_dim]
    forecast_wavelet_theta = forecast_coeffs[:, self.trend_dim:]

    # Expand each through its frozen basis and sum (parallel basis, additive output)
    backcast = self.backcast_trend_g(backcast_trend_theta) + self.backcast_wavelet_g(backcast_wavelet_theta)
    forecast = self.forecast_trend_g(forecast_trend_theta) + self.forecast_wavelet_g(forecast_wavelet_theta)

    if self.active_g:
      if self.active_g != 'forecast':
        backcast = self.activation(backcast)
      if self.active_g != 'backcast':
        forecast = self.activation(forecast)

    return backcast, forecast


class TrendWaveletAELG(AERootBlockLG):
  def __init__(self, units, backcast_length, forecast_length,
               trend_dim=4, wavelet_dim=16, basis_offset=0,
               share_weights=False, activation='ReLU', active_g=False,
               wavelet_type='db3', latent_dim=5, forecast_basis_dim=None):
    """TrendWaveletAELG extends TrendWaveletAE with the Learned-Gate AE backbone.

    Identical to TrendWaveletAE in its projection heads and frozen basis expansions
    (parallel Vandermonde polynomial + orthonormal DWT, additive output), but replaces
    the plain AERootBlock backbone with AERootBlockLG.  The LG backbone adds a learnable
    ``nn.Parameter`` gate vector of size ``latent_dim``; each forward pass applies
    ``sigmoid(latent_gate) * z`` after the bottleneck, allowing the network to
    discover the effective latent dimensionality during training and selectively suppress
    uninformative latent dimensions.

    Args:
        units (int): Width of the fully connected layers in the AE backbone.
        backcast_length (int): Length of the historical input sequence.
        forecast_length (int): Length of the forecast horizon.
        trend_dim (int, optional): Polynomial degree (number of Vandermonde basis vectors).
            Defaults to 4.
        wavelet_dim (int, optional): Number of DWT basis rows. Defaults to 16.
        basis_offset (int, optional): Row offset into SVD-ordered DWT basis for
            frequency-band selection.  0 = lowest/smoothest.  Defaults to 0.
        share_weights (bool, optional): Share the projection head between backcast and
            forecast paths when their total coefficient dimensions match. Defaults to False.
        activation (str, optional): Activation function for the AE backbone. Defaults to 'ReLU'.
        active_g (bool or str, optional): Apply activation after basis expansion.
            False = no activation.  True = both paths.
            'backcast'/'forecast' = single path.  Defaults to False.
        wavelet_type (str, optional): PyWavelets wavelet family string. Defaults to 'db3'.
        latent_dim (int, optional): Dimensionality of the AE bottleneck. Defaults to 5.
        forecast_basis_dim (int, optional): Override for wavelet basis dim on the forecast
            path.  When set the forecast wavelet head uses this value (clamped to
            forecast_length) while the backcast path continues to use wavelet_dim.
            Defaults to None (both paths use wavelet_dim).
    """
    super(TrendWaveletAELG, self).__init__(backcast_length, units, activation, latent_dim=latent_dim)
    self.trend_dim = trend_dim
    self.active_g = active_g
    self.wavelet_type = wavelet_type

    # Effective wavelet dimension for each path (clamped to available rows after offset)
    eff_back_wave = min(wavelet_dim, backcast_length - min(basis_offset, backcast_length - 1))
    fore_wavelet_dim = forecast_basis_dim if forecast_basis_dim is not None else wavelet_dim
    eff_fore_wave = min(fore_wavelet_dim, forecast_length - min(basis_offset, forecast_length - 1))

    # Total coefficient dims: trend + wavelet for each path
    total_backcast_dim = trend_dim + eff_back_wave
    total_forecast_dim = trend_dim + eff_fore_wave

    if share_weights and total_backcast_dim == total_forecast_dim:
      self.backcast_linear = self.forecast_linear = nn.Linear(units, total_backcast_dim)
    else:
      self.backcast_linear = nn.Linear(units, total_backcast_dim)
      self.forecast_linear = nn.Linear(units, total_forecast_dim)

    # Frozen polynomial (Vandermonde) basis generators
    self.backcast_trend_g = _TrendGenerator(trend_dim, backcast_length)
    self.forecast_trend_g = _TrendGenerator(trend_dim, forecast_length)

    # Frozen orthonormal DWT basis generators
    self.backcast_wavelet_g = _WaveletGeneratorV3(backcast_length, wavelet_type=wavelet_type,
                                                   basis_dim=eff_back_wave, basis_offset=basis_offset)
    self.forecast_wavelet_g = _WaveletGeneratorV3(forecast_length, wavelet_type=wavelet_type,
                                                   basis_dim=eff_fore_wave, basis_offset=basis_offset)

  def forward(self, x):
    x = super(TrendWaveletAELG, self).forward(x)

    # Project LG-AE output to concatenated trend + wavelet coefficient vectors
    backcast_coeffs = self.backcast_linear(x)
    forecast_coeffs = self.forecast_linear(x)

    # Split coefficient vectors into trend and wavelet parts
    backcast_trend_theta = backcast_coeffs[:, :self.trend_dim]
    backcast_wavelet_theta = backcast_coeffs[:, self.trend_dim:]
    forecast_trend_theta = forecast_coeffs[:, :self.trend_dim]
    forecast_wavelet_theta = forecast_coeffs[:, self.trend_dim:]

    # Expand each through its frozen basis and sum (parallel basis, additive output)
    backcast = self.backcast_trend_g(backcast_trend_theta) + self.backcast_wavelet_g(backcast_wavelet_theta)
    forecast = self.forecast_trend_g(forecast_trend_theta) + self.forecast_wavelet_g(forecast_wavelet_theta)

    if self.active_g:
      if self.active_g != 'forecast':
        backcast = self.activation(backcast)
      if self.active_g != 'backcast':
        forecast = self.activation(forecast)

    return backcast, forecast

class SeasonalityAE(AERootBlock):
  def __init__(self, units, backcast_length, forecast_length,  thetas_dim=5,
               share_weights = False, activation='ReLU', active_g:bool = False, latent_dim = 5):
    """The SeasonalityAEBlock is the basic building block of the N-BEATS network.
    It consists of a stack of fully connected layers defined the parent class Block,
    followed by a linear layer, which generates the parameters of a Fourier expansion.

    The SeasonalityAEBlock is an AutoEncoder version of the SeasonalityBlock where the presplit
    section of the network is an AutoEncoder.

    Args:
        units (int):
          The width of the fully connected layers in the blocks comprising the parent
          class Block.
        backcast_length (int):
          The length of the historical data.  It is customary to use a multiple of
          the forecast_length (H)orizon (2H,3H,4H,5H,...).
        forecast_length (int):
          The length of the forecast_length horizon.
        activation (str, optional):
          The activation function passed to the parent class Block. Defaults to 'LeakyReLU'.
    """
    super(SeasonalityAE, self).__init__(backcast_length, units, activation, latent_dim = latent_dim)

    self.backcast_linear = nn.Linear(units, 2 * int(backcast_length / 2 - 1) + 1, bias = False)
    self.forecast_linear = nn.Linear(units, 2 * int(forecast_length / 2 - 1) + 1, bias = False)

    self.backcast_g = _SeasonalityGenerator(backcast_length)
    self.forecast_g = _SeasonalityGenerator(forecast_length)

  def forward(self, x):
    x = super(SeasonalityAE, self).forward(x)
    # linear compression
    backcast_thetas = self.backcast_linear(x)
    forecast_thetas = self.forecast_linear(x)

    # fourier expansion
    backcast = self.backcast_g(backcast_thetas)
    forecast = self.forecast_g(forecast_thetas)

    return backcast, forecast


# ---------------------------------------------------------------------------
# Learned-Gate (LG) AE Subclasses — AERootBlockLG backbone
# ---------------------------------------------------------------------------

class GenericAELG(AERootBlockLG):
  def __init__(self, units, backcast_length, forecast_length, thetas_dim=5,
               share_weights=False, activation='ReLU', active_g=False, latent_dim=5):
    super(GenericAELG, self).__init__(backcast_length, units, activation, latent_dim=latent_dim)
    self.backcast_length = backcast_length
    self.forecast_length = forecast_length
    self.active_g = active_g
    self.theta_b_fc = nn.Linear(units, backcast_length, bias=False)
    self.theta_f_fc = nn.Linear(units, forecast_length, bias=False)

  def forward(self, x):
    x = super(GenericAELG, self).forward(x)
    backcast = self.theta_b_fc(x)
    forecast = self.theta_f_fc(x)
    if self.active_g:
      if self.active_g != 'forecast':
        backcast = self.activation(backcast)
      if self.active_g != 'backcast':
        forecast = self.activation(forecast)
    return backcast, forecast

class BottleneckGenericAELG(AERootBlockLG):
  def __init__(self, units, backcast_length, forecast_length, thetas_dim=5,
               share_weights=False, activation='ReLU', active_g=False, latent_dim=5):
    super(BottleneckGenericAELG, self).__init__(backcast_length, units, activation, latent_dim=latent_dim)
    if share_weights:
      self.backcast_linear = self.forecast_linear = nn.Linear(units, thetas_dim)
    else:
      self.backcast_linear = nn.Linear(units, thetas_dim)
      self.forecast_linear = nn.Linear(units, thetas_dim)
    self.backcast_g = nn.Linear(thetas_dim, backcast_length, bias=False)
    self.forecast_g = nn.Linear(thetas_dim, forecast_length, bias=False)
    self.active_g = active_g

  def forward(self, x):
    x = super(BottleneckGenericAELG, self).forward(x)
    theta_b = self.backcast_linear(x)
    theta_f = self.forecast_linear(x)
    backcast = self.backcast_g(theta_b)
    forecast = self.forecast_g(theta_f)
    if self.active_g:
      if self.active_g != 'forecast':
        backcast = self.activation(backcast)
      if self.active_g != 'backcast':
        forecast = self.activation(forecast)
    return backcast, forecast

class TrendAELG(AERootBlockLG):
  def __init__(self, units, backcast_length, forecast_length, thetas_dim,
               share_weights=False, activation='ReLU', active_g=False, latent_dim=5):
    super(TrendAELG, self).__init__(backcast_length, units, activation, latent_dim=latent_dim)
    if share_weights:
      self.backcast_linear = self.forecast_linear = nn.Linear(units, thetas_dim)
    else:
      self.backcast_linear = nn.Linear(units, thetas_dim)
      self.forecast_linear = nn.Linear(units, thetas_dim)
    self.backcast_g = _TrendGenerator(thetas_dim, backcast_length)
    self.forecast_g = _TrendGenerator(thetas_dim, forecast_length)

  def forward(self, x):
    x = super(TrendAELG, self).forward(x)
    backcast_thetas = self.backcast_linear(x)
    forecast_thetas = self.forecast_linear(x)
    backcast = self.backcast_g(backcast_thetas)
    forecast = self.forecast_g(forecast_thetas)
    return backcast, forecast

class SeasonalityAELG(AERootBlockLG):
  def __init__(self, units, backcast_length, forecast_length, thetas_dim=5,
               share_weights=False, activation='ReLU', active_g=False, latent_dim=5):
    super(SeasonalityAELG, self).__init__(backcast_length, units, activation, latent_dim=latent_dim)
    self.backcast_linear = nn.Linear(units, 2 * int(backcast_length / 2 - 1) + 1, bias=False)
    self.forecast_linear = nn.Linear(units, 2 * int(forecast_length / 2 - 1) + 1, bias=False)
    self.backcast_g = _SeasonalityGenerator(backcast_length)
    self.forecast_g = _SeasonalityGenerator(forecast_length)

  def forward(self, x):
    x = super(SeasonalityAELG, self).forward(x)
    backcast_thetas = self.backcast_linear(x)
    forecast_thetas = self.forecast_linear(x)
    backcast = self.backcast_g(backcast_thetas)
    forecast = self.forecast_g(forecast_thetas)
    return backcast, forecast

class AutoEncoderAELG(AERootBlockLG):
  def __init__(self, units, backcast_length, forecast_length, thetas_dim,
               share_weights=False, activation='ReLU', active_g=False, latent_dim=5):
    super(AutoEncoderAELG, self).__init__(backcast_length, units, activation, latent_dim=latent_dim)
    self.units = units
    self.thetas_dim = thetas_dim
    self.share_weights = share_weights
    self.activation = getattr(nn, activation)()
    self.backcast_length = backcast_length
    self.forecast_length = forecast_length
    self.active_g = active_g
    branch_latent_dim = latent_dim
    if share_weights:
      self.b_encoder = self.f_encoder = nn.Linear(units, branch_latent_dim)
    else:
      self.b_encoder = nn.Linear(units, branch_latent_dim)
      self.f_encoder = nn.Linear(units, branch_latent_dim)
    self.b_decoder = nn.Linear(branch_latent_dim, units)
    self.f_decoder = nn.Linear(branch_latent_dim, units)
    self.backcast_g = nn.Linear(units, backcast_length)
    self.forecast_g = nn.Linear(units, forecast_length)

  def forward(self, x):
    x = super(AutoEncoderAELG, self).forward(x)
    b = self.activation(self.b_encoder(x))
    b = self.activation(self.b_decoder(b))
    b = self.backcast_g(b)
    f = self.activation(self.f_encoder(x))
    f = self.activation(self.f_decoder(f))
    f = self.forecast_g(f)
    if self.active_g:
      if self.active_g != 'forecast':
        b = self.activation(b)
      if self.active_g != 'backcast':
        f = self.activation(f)
    return b, f


class GenericAEBackcastAELG(AERootBlockLG):
  def __init__(self, units, backcast_length, forecast_length, thetas_dim,
               share_weights=False, activation='ReLU', active_g=False, latent_dim=5):
    super(GenericAEBackcastAELG, self).__init__(backcast_length, units, activation, latent_dim=latent_dim)
    self.units = units
    self.thetas_dim = thetas_dim
    self.share_weights = share_weights
    self.activation = getattr(nn, activation)()
    self.backcast_length = backcast_length
    self.forecast_length = forecast_length
    self.active_g = active_g
    branch_latent_dim = latent_dim
    self.forecast_linear = nn.Linear(units, thetas_dim)
    self.forecast_g = nn.Linear(thetas_dim, forecast_length, bias=False)
    self.b_encoder = nn.Linear(units, branch_latent_dim)
    self.b_decoder = nn.Linear(branch_latent_dim, units)
    self.backcast_g = nn.Linear(units, backcast_length)

  def forward(self, x):
    x = super(GenericAEBackcastAELG, self).forward(x)
    b = self.activation(self.b_encoder(x))
    b = self.activation(self.b_decoder(b))
    b = self.backcast_g(b)
    theta_f = self.forecast_linear(x)
    f = self.forecast_g(theta_f)
    if self.active_g:
      if self.active_g != 'forecast':
        b = self.activation(b)
      if self.active_g != 'backcast':
        f = self.activation(f)
    return b, f


# ---------------------------------------------------------------------------
# Variational AE (VAE) Subclasses — AERootBlockVAE backbone
# ---------------------------------------------------------------------------

class GenericVAE(AERootBlockVAE):
  def __init__(self, units, backcast_length, forecast_length, thetas_dim=5,
               share_weights=False, activation='ReLU', active_g=False, latent_dim=5):
    super(GenericVAE, self).__init__(backcast_length, units, activation, latent_dim=latent_dim)
    self.backcast_length = backcast_length
    self.forecast_length = forecast_length
    self.active_g = active_g
    self.theta_b_fc = nn.Linear(units, backcast_length, bias=False)
    self.theta_f_fc = nn.Linear(units, forecast_length, bias=False)

  def forward(self, x):
    x = super(GenericVAE, self).forward(x)
    backcast = self.theta_b_fc(x)
    forecast = self.theta_f_fc(x)
    if self.active_g:
      if self.active_g != 'forecast':
        backcast = self.activation(backcast)
      if self.active_g != 'backcast':
        forecast = self.activation(forecast)
    return backcast, forecast


class BottleneckGenericVAE(AERootBlockVAE):
  def __init__(self, units, backcast_length, forecast_length, thetas_dim=5,
               share_weights=False, activation='ReLU', active_g=False, latent_dim=5):
    super(BottleneckGenericVAE, self).__init__(backcast_length, units, activation, latent_dim=latent_dim)
    if share_weights:
      self.backcast_linear = self.forecast_linear = nn.Linear(units, thetas_dim)
    else:
      self.backcast_linear = nn.Linear(units, thetas_dim)
      self.forecast_linear = nn.Linear(units, thetas_dim)
    self.backcast_g = nn.Linear(thetas_dim, backcast_length, bias=False)
    self.forecast_g = nn.Linear(thetas_dim, forecast_length, bias=False)
    self.active_g = active_g

  def forward(self, x):
    x = super(BottleneckGenericVAE, self).forward(x)
    theta_b = self.backcast_linear(x)
    theta_f = self.forecast_linear(x)
    backcast = self.backcast_g(theta_b)
    forecast = self.forecast_g(theta_f)
    if self.active_g:
      if self.active_g != 'forecast':
        backcast = self.activation(backcast)
      if self.active_g != 'backcast':
        forecast = self.activation(forecast)
    return backcast, forecast


class TrendVAE(AERootBlockVAE):
  def __init__(self, units, backcast_length, forecast_length, thetas_dim,
               share_weights=False, activation='ReLU', active_g=False, latent_dim=5):
    super(TrendVAE, self).__init__(backcast_length, units, activation, latent_dim=latent_dim)
    if share_weights:
      self.backcast_linear = self.forecast_linear = nn.Linear(units, thetas_dim)
    else:
      self.backcast_linear = nn.Linear(units, thetas_dim)
      self.forecast_linear = nn.Linear(units, thetas_dim)
    self.backcast_g = _TrendGenerator(thetas_dim, backcast_length)
    self.forecast_g = _TrendGenerator(thetas_dim, forecast_length)

  def forward(self, x):
    x = super(TrendVAE, self).forward(x)
    backcast_thetas = self.backcast_linear(x)
    forecast_thetas = self.forecast_linear(x)
    backcast = self.backcast_g(backcast_thetas)
    forecast = self.forecast_g(forecast_thetas)
    return backcast, forecast


class SeasonalityVAE(AERootBlockVAE):
  def __init__(self, units, backcast_length, forecast_length, thetas_dim=5,
               share_weights=False, activation='ReLU', active_g=False, latent_dim=5):
    super(SeasonalityVAE, self).__init__(backcast_length, units, activation, latent_dim=latent_dim)
    self.backcast_linear = nn.Linear(units, 2 * int(backcast_length / 2 - 1) + 1, bias=False)
    self.forecast_linear = nn.Linear(units, 2 * int(forecast_length / 2 - 1) + 1, bias=False)
    self.backcast_g = _SeasonalityGenerator(backcast_length)
    self.forecast_g = _SeasonalityGenerator(forecast_length)

  def forward(self, x):
    x = super(SeasonalityVAE, self).forward(x)
    backcast_thetas = self.backcast_linear(x)
    forecast_thetas = self.forecast_linear(x)
    backcast = self.backcast_g(backcast_thetas)
    forecast = self.forecast_g(forecast_thetas)
    return backcast, forecast


class AutoEncoderVAE(AERootBlockVAE):
  def __init__(self, units, backcast_length, forecast_length, thetas_dim,
               share_weights=False, activation='ReLU', active_g=False, latent_dim=5):
    super(AutoEncoderVAE, self).__init__(backcast_length, units, activation, latent_dim=latent_dim)
    self.units = units
    self.thetas_dim = thetas_dim
    self.share_weights = share_weights
    self.activation = getattr(nn, activation)()
    self.backcast_length = backcast_length
    self.forecast_length = forecast_length
    self.active_g = active_g
    branch_latent_dim = latent_dim
    if share_weights:
      self.b_encoder = self.f_encoder = nn.Linear(units, branch_latent_dim)
    else:
      self.b_encoder = nn.Linear(units, branch_latent_dim)
      self.f_encoder = nn.Linear(units, branch_latent_dim)
    self.b_decoder = nn.Linear(branch_latent_dim, units)
    self.f_decoder = nn.Linear(branch_latent_dim, units)
    self.backcast_g = nn.Linear(units, backcast_length)
    self.forecast_g = nn.Linear(units, forecast_length)

  def forward(self, x):
    x = super(AutoEncoderVAE, self).forward(x)
    b = self.activation(self.b_encoder(x))
    b = self.activation(self.b_decoder(b))
    b = self.backcast_g(b)
    f = self.activation(self.f_encoder(x))
    f = self.activation(self.f_decoder(f))
    f = self.forecast_g(f)
    if self.active_g:
      if self.active_g != 'forecast':
        b = self.activation(b)
      if self.active_g != 'backcast':
        f = self.activation(f)
    return b, f


class GenericAEBackcastVAE(AERootBlockVAE):
  def __init__(self, units, backcast_length, forecast_length, thetas_dim,
               share_weights=False, activation='ReLU', active_g=False, latent_dim=5):
    super(GenericAEBackcastVAE, self).__init__(backcast_length, units, activation, latent_dim=latent_dim)
    self.units = units
    self.thetas_dim = thetas_dim
    self.share_weights = share_weights
    self.activation = getattr(nn, activation)()
    self.backcast_length = backcast_length
    self.forecast_length = forecast_length
    self.active_g = active_g
    branch_latent_dim = latent_dim
    self.forecast_linear = nn.Linear(units, thetas_dim)
    self.forecast_g = nn.Linear(thetas_dim, forecast_length, bias=False)
    self.b_encoder = nn.Linear(units, branch_latent_dim)
    self.b_decoder = nn.Linear(branch_latent_dim, units)
    self.backcast_g = nn.Linear(units, backcast_length)

  def forward(self, x):
    x = super(GenericAEBackcastVAE, self).forward(x)
    b = self.activation(self.b_encoder(x))
    b = self.activation(self.b_decoder(b))
    b = self.backcast_g(b)
    theta_f = self.forecast_linear(x)
    f = self.forecast_g(theta_f)
    if self.active_g:
      if self.active_g != 'forecast':
        b = self.activation(b)
      if self.active_g != 'backcast':
        f = self.activation(f)
    return b, f


# ---------------------------------------------------------------------------
# V3 Wavelet Blocks — Orthonormal DWT basis via impulse-response synthesis
#
# Root cause fix for V1/V2 instability: the rolled phi/psi basis construction
# produces ill-conditioned matrices (DB3 cond~604K). V3 replaces this entirely
# with proper DWT impulse-response synthesis + SVD orthogonalization, producing
# a genuinely orthonormal basis where ALL singular values = 1.0 (cond = 1.0).
#
# Architecture: 2-projection pattern matching Seasonality/Trend (no downsampling
# layer needed since the basis is always target_length x target_length).
# ---------------------------------------------------------------------------

class _WaveletGeneratorV3(nn.Module):
  def __init__(self, target_length, wavelet_type='db3', max_decomp_level=5, basis_dim=None,
               basis_offset=0):
    super().__init__()
    basis = self._build_basis(target_length, wavelet_type, max_decomp_level, basis_dim,
                              basis_offset)
    self.basis = nn.Parameter(basis, requires_grad=False)

  @staticmethod
  def _build_basis(target_length, wavelet_type, max_decomp_level, basis_dim=None,
                   basis_offset=0):
    import logging

    wavelet = PyWavelet(wavelet_type)
    max_level = pywt.dwt_max_level(target_length, wavelet.dec_len)
    level = max(1, min(max_level, max_decomp_level))

    # Get DWT coefficient structure
    dummy = np.zeros(target_length)
    coeffs = pywt.wavedec(dummy, wavelet_type, level=level)
    coeff_lengths = [len(c) for c in coeffs]

    # Build raw synthesis matrix via impulse responses
    basis_rows = []
    for band_idx, band_len in enumerate(coeff_lengths):
      for j in range(band_len):
        impulse = [np.zeros(l) for l in coeff_lengths]
        impulse[band_idx][j] = 1.0
        reconstructed = pywt.waverec(impulse, wavelet_type)
        basis_rows.append(reconstructed[:target_length])

    raw_basis = np.array(basis_rows, dtype=np.float64)

    # SVD orthogonalization — rows of Vt are ordered low→high frequency by singular value
    U, S, Vt = np.linalg.svd(raw_basis, full_matrices=False)
    tol = S[0] * max(raw_basis.shape) * np.finfo(np.float64).eps
    full_rank = int(np.sum(S > tol))

    if full_rank < target_length:
      logging.warning(f"WaveletV3 rank-deficient: {full_rank}/{target_length} for '{wavelet_type}'")

    # Select a frequency band via offset + window:
    #   basis_offset=0,  basis_dim=32  → rows [0:32]   (low-frequency)
    #   basis_offset=32, basis_dim=32  → rows [32:64]  (mid-frequency)
    #   basis_offset=0,  basis_dim=None → all rows      (full spectrum)
    # Clamp so the slice never exceeds full_rank.
    offset = min(basis_offset, full_rank - 1)
    available = full_rank - offset
    effective_dim = available if basis_dim is None else min(available, basis_dim)
    ortho_basis = Vt[offset : offset + effective_dim, :]  # (effective_dim, target_length)

    return torch.tensor(ortho_basis, dtype=torch.float32)

  def forward(self, x):
    # x: (batch, effective_dim), basis: (effective_dim, target_length)
    return torch.matmul(x, self.basis)


class WaveletV3(RootBlock):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False, wavelet_type='db3',
               forecast_basis_dim=None):
    super(WaveletV3, self).__init__(backcast_length, units, activation)
    self.wavelet_type = wavelet_type
    self.share_weights = share_weights
    self.active_g = active_g

    # Clamp basis_dim to the available rows after offset (can't exceed sequence length).
    # basis_dim=None means no truncation; basis_offset shifts which frequency band is used.
    eff_back_dim = (backcast_length - min(basis_offset, backcast_length - 1)
                    if basis_dim is None
                    else min(basis_dim, backcast_length - min(basis_offset, backcast_length - 1)))
    fore_dim = forecast_basis_dim if forecast_basis_dim is not None else basis_dim
    eff_fore_dim = (forecast_length - min(basis_offset, forecast_length - 1)
                    if fore_dim is None
                    else min(fore_dim, forecast_length - min(basis_offset, forecast_length - 1)))

    self.backcast_linear = nn.Linear(units, eff_back_dim, bias=False)
    self.forecast_linear = nn.Linear(units, eff_fore_dim, bias=False)

    self.backcast_g = _WaveletGeneratorV3(backcast_length, wavelet_type=wavelet_type,
                                          basis_dim=eff_back_dim, basis_offset=basis_offset)
    self.forecast_g = _WaveletGeneratorV3(forecast_length, wavelet_type=wavelet_type,
                                          basis_dim=eff_fore_dim, basis_offset=basis_offset)

  def forward(self, x):
    x = super(WaveletV3, self).forward(x)
    backcast_thetas = self.backcast_linear(x)
    forecast_thetas = self.forecast_linear(x)
    backcast = self.backcast_g(backcast_thetas)
    forecast = self.forecast_g(forecast_thetas)
    if self.active_g:
      if self.active_g != 'forecast':
        backcast = self.activation(backcast)
      if self.active_g != 'backcast':
        forecast = self.activation(forecast)
    return backcast, forecast


# --- V3 Wavelet subclasses (thin wrappers setting wavelet_type) ---

class HaarWaveletV3(WaveletV3):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None):
    super(HaarWaveletV3, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                       basis_offset, share_weights, activation, active_g,
                                       wavelet_type='haar',
                                       forecast_basis_dim=forecast_basis_dim)
  def forward(self, x):
    return super(HaarWaveletV3, self).forward(x)

class DB2WaveletV3(WaveletV3):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None):
    super(DB2WaveletV3, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                      basis_offset, share_weights, activation, active_g,
                                      wavelet_type='db2',
                                      forecast_basis_dim=forecast_basis_dim)
  def forward(self, x):
    return super(DB2WaveletV3, self).forward(x)

class DB3WaveletV3(WaveletV3):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None):
    super(DB3WaveletV3, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                      basis_offset, share_weights, activation, active_g,
                                      wavelet_type='db3',
                                      forecast_basis_dim=forecast_basis_dim)
  def forward(self, x):
    return super(DB3WaveletV3, self).forward(x)

class DB4WaveletV3(WaveletV3):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None):
    super(DB4WaveletV3, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                      basis_offset, share_weights, activation, active_g,
                                      wavelet_type='db4',
                                      forecast_basis_dim=forecast_basis_dim)
  def forward(self, x):
    return super(DB4WaveletV3, self).forward(x)

class DB10WaveletV3(WaveletV3):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None):
    super(DB10WaveletV3, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                       basis_offset, share_weights, activation, active_g,
                                       wavelet_type='db10',
                                       forecast_basis_dim=forecast_basis_dim)
  def forward(self, x):
    return super(DB10WaveletV3, self).forward(x)

class DB20WaveletV3(WaveletV3):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None):
    super(DB20WaveletV3, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                       basis_offset, share_weights, activation, active_g,
                                       wavelet_type='db20',
                                       forecast_basis_dim=forecast_basis_dim)
  def forward(self, x):
    return super(DB20WaveletV3, self).forward(x)

class Coif1WaveletV3(WaveletV3):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None):
    super(Coif1WaveletV3, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                        basis_offset, share_weights, activation, active_g,
                                        wavelet_type='coif1',
                                        forecast_basis_dim=forecast_basis_dim)
  def forward(self, x):
    return super(Coif1WaveletV3, self).forward(x)

class Coif2WaveletV3(WaveletV3):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None):
    super(Coif2WaveletV3, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                        basis_offset, share_weights, activation, active_g,
                                        wavelet_type='coif2',
                                        forecast_basis_dim=forecast_basis_dim)
  def forward(self, x):
    return super(Coif2WaveletV3, self).forward(x)

class Coif3WaveletV3(WaveletV3):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None):
    super(Coif3WaveletV3, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                        basis_offset, share_weights, activation, active_g,
                                        wavelet_type='coif3',
                                        forecast_basis_dim=forecast_basis_dim)
  def forward(self, x):
    return super(Coif3WaveletV3, self).forward(x)

class Coif10WaveletV3(WaveletV3):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None):
    super(Coif10WaveletV3, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                         basis_offset, share_weights, activation, active_g,
                                         wavelet_type='coif10',
                                         forecast_basis_dim=forecast_basis_dim)
  def forward(self, x):
    return super(Coif10WaveletV3, self).forward(x)

class Symlet2WaveletV3(WaveletV3):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None):
    super(Symlet2WaveletV3, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                          basis_offset, share_weights, activation, active_g,
                                          wavelet_type='sym2',
                                          forecast_basis_dim=forecast_basis_dim)
  def forward(self, x):
    return super(Symlet2WaveletV3, self).forward(x)

class Symlet3WaveletV3(WaveletV3):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None):
    super(Symlet3WaveletV3, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                          basis_offset, share_weights, activation, active_g,
                                          wavelet_type='sym3',
                                          forecast_basis_dim=forecast_basis_dim)
  def forward(self, x):
    return super(Symlet3WaveletV3, self).forward(x)

class Symlet10WaveletV3(WaveletV3):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None):
    super(Symlet10WaveletV3, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                           basis_offset, share_weights, activation, active_g,
                                           wavelet_type='sym10',
                                           forecast_basis_dim=forecast_basis_dim)
  def forward(self, x):
    return super(Symlet10WaveletV3, self).forward(x)

class Symlet20WaveletV3(WaveletV3):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None):
    super(Symlet20WaveletV3, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                           basis_offset, share_weights, activation, active_g,
                                           wavelet_type='sym20',
                                           forecast_basis_dim=forecast_basis_dim)
  def forward(self, x):
    return super(Symlet20WaveletV3, self).forward(x)


# ---------------------------------------------------------------------------
# WaveletV3AE — V3 wavelet blocks with AE bottleneck backbone (Option B)
# ---------------------------------------------------------------------------

class WaveletV3AE(AERootBlock):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False, wavelet_type='db3',
               forecast_basis_dim=None, latent_dim=5):
    """WaveletV3AE: orthonormal DWT basis block with AutoEncoder backbone (Option B).

    Structurally identical to WaveletV3 but inherits from AERootBlock instead of RootBlock,
    replacing the flat 4-FC-layer backbone with an encoder-decoder bottleneck backbone
    (backcast_length → units/2 → latent_dim → units/2 → units).  This reduces backbone
    trainable parameters significantly at large `units` values while preserving the frozen
    orthonormal DWT basis expansion for the projection heads.

    Args:
        units (int): Width of the fully connected layers in the AE backbone.
        backcast_length (int): Length of the historical input sequence.
        forecast_length (int): Length of the forecast horizon.
        basis_dim (int, optional): Number of DWT basis rows. Defaults to 32.
        basis_offset (int, optional): Row offset into SVD-ordered basis. Defaults to 0.
        share_weights (bool, optional): Share projection heads when dimensions match.
            Defaults to False.
        activation (str, optional): Activation for the AE backbone. Defaults to 'ReLU'.
        active_g (bool or str, optional): Apply activation after basis expansion.
            Defaults to False.
        wavelet_type (str, optional): PyWavelets wavelet family string. Defaults to 'db3'.
        forecast_basis_dim (int, optional): Override basis_dim for forecast path only.
            Defaults to None (both paths use basis_dim).
        latent_dim (int, optional): Dimensionality of the AE bottleneck. Defaults to 5.
    """
    super(WaveletV3AE, self).__init__(backcast_length, units, activation, latent_dim=latent_dim)
    self.wavelet_type = wavelet_type
    self.share_weights = share_weights
    self.active_g = active_g

    eff_back_dim = (backcast_length - min(basis_offset, backcast_length - 1)
                    if basis_dim is None
                    else min(basis_dim, backcast_length - min(basis_offset, backcast_length - 1)))
    fore_dim = forecast_basis_dim if forecast_basis_dim is not None else basis_dim
    eff_fore_dim = (forecast_length - min(basis_offset, forecast_length - 1)
                    if fore_dim is None
                    else min(fore_dim, forecast_length - min(basis_offset, forecast_length - 1)))

    self.backcast_linear = nn.Linear(units, eff_back_dim, bias=False)
    self.forecast_linear = nn.Linear(units, eff_fore_dim, bias=False)

    self.backcast_g = _WaveletGeneratorV3(backcast_length, wavelet_type=wavelet_type,
                                          basis_dim=eff_back_dim, basis_offset=basis_offset)
    self.forecast_g = _WaveletGeneratorV3(forecast_length, wavelet_type=wavelet_type,
                                          basis_dim=eff_fore_dim, basis_offset=basis_offset)

  def forward(self, x):
    x = super(WaveletV3AE, self).forward(x)
    backcast_thetas = self.backcast_linear(x)
    forecast_thetas = self.forecast_linear(x)
    backcast = self.backcast_g(backcast_thetas)
    forecast = self.forecast_g(forecast_thetas)
    if self.active_g:
      if self.active_g != 'forecast':
        backcast = self.activation(backcast)
      if self.active_g != 'backcast':
        forecast = self.activation(forecast)
    return backcast, forecast


# --- V3AE Wavelet subclasses (thin wrappers setting wavelet_type) ---

class HaarWaveletV3AE(WaveletV3AE):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None, latent_dim=5):
    super(HaarWaveletV3AE, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                         basis_offset, share_weights, activation, active_g,
                                         wavelet_type='haar',
                                         forecast_basis_dim=forecast_basis_dim, latent_dim=latent_dim)
  def forward(self, x):
    return super(HaarWaveletV3AE, self).forward(x)

class DB2WaveletV3AE(WaveletV3AE):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None, latent_dim=5):
    super(DB2WaveletV3AE, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                        basis_offset, share_weights, activation, active_g,
                                        wavelet_type='db2',
                                        forecast_basis_dim=forecast_basis_dim, latent_dim=latent_dim)
  def forward(self, x):
    return super(DB2WaveletV3AE, self).forward(x)

class DB3WaveletV3AE(WaveletV3AE):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None, latent_dim=5):
    super(DB3WaveletV3AE, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                        basis_offset, share_weights, activation, active_g,
                                        wavelet_type='db3',
                                        forecast_basis_dim=forecast_basis_dim, latent_dim=latent_dim)
  def forward(self, x):
    return super(DB3WaveletV3AE, self).forward(x)

class DB4WaveletV3AE(WaveletV3AE):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None, latent_dim=5):
    super(DB4WaveletV3AE, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                        basis_offset, share_weights, activation, active_g,
                                        wavelet_type='db4',
                                        forecast_basis_dim=forecast_basis_dim, latent_dim=latent_dim)
  def forward(self, x):
    return super(DB4WaveletV3AE, self).forward(x)

class DB10WaveletV3AE(WaveletV3AE):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None, latent_dim=5):
    super(DB10WaveletV3AE, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                         basis_offset, share_weights, activation, active_g,
                                         wavelet_type='db10',
                                         forecast_basis_dim=forecast_basis_dim, latent_dim=latent_dim)
  def forward(self, x):
    return super(DB10WaveletV3AE, self).forward(x)

class DB20WaveletV3AE(WaveletV3AE):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None, latent_dim=5):
    super(DB20WaveletV3AE, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                         basis_offset, share_weights, activation, active_g,
                                         wavelet_type='db20',
                                         forecast_basis_dim=forecast_basis_dim, latent_dim=latent_dim)
  def forward(self, x):
    return super(DB20WaveletV3AE, self).forward(x)


class Coif1WaveletV3AE(WaveletV3AE):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None, latent_dim=5):
    super(Coif1WaveletV3AE, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                          basis_offset, share_weights, activation, active_g,
                                          wavelet_type='coif1',
                                          forecast_basis_dim=forecast_basis_dim, latent_dim=latent_dim)
  def forward(self, x):
    return super(Coif1WaveletV3AE, self).forward(x)

class Coif2WaveletV3AE(WaveletV3AE):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None, latent_dim=5):
    super(Coif2WaveletV3AE, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                          basis_offset, share_weights, activation, active_g,
                                          wavelet_type='coif2',
                                          forecast_basis_dim=forecast_basis_dim, latent_dim=latent_dim)
  def forward(self, x):
    return super(Coif2WaveletV3AE, self).forward(x)

class Coif3WaveletV3AE(WaveletV3AE):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None, latent_dim=5):
    super(Coif3WaveletV3AE, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                          basis_offset, share_weights, activation, active_g,
                                          wavelet_type='coif3',
                                          forecast_basis_dim=forecast_basis_dim, latent_dim=latent_dim)
  def forward(self, x):
    return super(Coif3WaveletV3AE, self).forward(x)

class Coif10WaveletV3AE(WaveletV3AE):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None, latent_dim=5):
    super(Coif10WaveletV3AE, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                           basis_offset, share_weights, activation, active_g,
                                           wavelet_type='coif10',
                                           forecast_basis_dim=forecast_basis_dim, latent_dim=latent_dim)
  def forward(self, x):
    return super(Coif10WaveletV3AE, self).forward(x)

class Symlet2WaveletV3AE(WaveletV3AE):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None, latent_dim=5):
    super(Symlet2WaveletV3AE, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                            basis_offset, share_weights, activation, active_g,
                                            wavelet_type='sym2',
                                            forecast_basis_dim=forecast_basis_dim, latent_dim=latent_dim)
  def forward(self, x):
    return super(Symlet2WaveletV3AE, self).forward(x)

class Symlet3WaveletV3AE(WaveletV3AE):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None, latent_dim=5):
    super(Symlet3WaveletV3AE, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                            basis_offset, share_weights, activation, active_g,
                                            wavelet_type='sym3',
                                            forecast_basis_dim=forecast_basis_dim, latent_dim=latent_dim)
  def forward(self, x):
    return super(Symlet3WaveletV3AE, self).forward(x)

class Symlet10WaveletV3AE(WaveletV3AE):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None, latent_dim=5):
    super(Symlet10WaveletV3AE, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                             basis_offset, share_weights, activation, active_g,
                                             wavelet_type='sym10',
                                             forecast_basis_dim=forecast_basis_dim, latent_dim=latent_dim)
  def forward(self, x):
    return super(Symlet10WaveletV3AE, self).forward(x)

class Symlet20WaveletV3AE(WaveletV3AE):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None, latent_dim=5):
    super(Symlet20WaveletV3AE, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                             basis_offset, share_weights, activation, active_g,
                                             wavelet_type='sym20',
                                             forecast_basis_dim=forecast_basis_dim, latent_dim=latent_dim)
  def forward(self, x):
    return super(Symlet20WaveletV3AE, self).forward(x)


# ---------------------------------------------------------------------------
# WaveletV3AELG — V3 wavelet blocks with Learned-Gate AE backbone
# ---------------------------------------------------------------------------

class WaveletV3AELG(AERootBlockLG):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False, wavelet_type='db3',
               forecast_basis_dim=None, latent_dim=5):
    """WaveletV3AELG: orthonormal DWT basis block with Learned-Gate AE backbone.

    Structurally identical to WaveletV3AE but inherits from AERootBlockLG instead of
    AERootBlock, adding a learnable sigmoid gate vector at the latent bottleneck that
    allows the network to discover effective latent dimensionality during training.

    Args:
        units (int): Width of the fully connected layers in the LG-AE backbone.
        backcast_length (int): Length of the historical input sequence.
        forecast_length (int): Length of the forecast horizon.
        basis_dim (int, optional): Number of DWT basis rows. Defaults to 32.
        basis_offset (int, optional): Row offset into SVD-ordered basis. Defaults to 0.
        share_weights (bool, optional): Share projection heads when dimensions match.
            Defaults to False.
        activation (str, optional): Activation for the LG-AE backbone. Defaults to 'ReLU'.
        active_g (bool or str, optional): Apply activation after basis expansion.
            Defaults to False.
        wavelet_type (str, optional): PyWavelets wavelet family string. Defaults to 'db3'.
        forecast_basis_dim (int, optional): Override basis_dim for forecast path only.
            Defaults to None (both paths use basis_dim).
        latent_dim (int, optional): Dimensionality of the LG-AE bottleneck. Defaults to 5.
    """
    super(WaveletV3AELG, self).__init__(backcast_length, units, activation, latent_dim=latent_dim)
    self.wavelet_type = wavelet_type
    self.share_weights = share_weights
    self.active_g = active_g

    eff_back_dim = (backcast_length - min(basis_offset, backcast_length - 1)
                    if basis_dim is None
                    else min(basis_dim, backcast_length - min(basis_offset, backcast_length - 1)))
    fore_dim = forecast_basis_dim if forecast_basis_dim is not None else basis_dim
    eff_fore_dim = (forecast_length - min(basis_offset, forecast_length - 1)
                    if fore_dim is None
                    else min(fore_dim, forecast_length - min(basis_offset, forecast_length - 1)))

    self.backcast_linear = nn.Linear(units, eff_back_dim, bias=False)
    self.forecast_linear = nn.Linear(units, eff_fore_dim, bias=False)

    self.backcast_g = _WaveletGeneratorV3(backcast_length, wavelet_type=wavelet_type,
                                          basis_dim=eff_back_dim, basis_offset=basis_offset)
    self.forecast_g = _WaveletGeneratorV3(forecast_length, wavelet_type=wavelet_type,
                                          basis_dim=eff_fore_dim, basis_offset=basis_offset)

  def forward(self, x):
    x = super(WaveletV3AELG, self).forward(x)
    backcast_thetas = self.backcast_linear(x)
    forecast_thetas = self.forecast_linear(x)
    backcast = self.backcast_g(backcast_thetas)
    forecast = self.forecast_g(forecast_thetas)
    if self.active_g:
      if self.active_g != 'forecast':
        backcast = self.activation(backcast)
      if self.active_g != 'backcast':
        forecast = self.activation(forecast)
    return backcast, forecast


# --- V3AELG Wavelet subclasses (thin wrappers setting wavelet_type) ---

class HaarWaveletV3AELG(WaveletV3AELG):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None, latent_dim=5):
    super(HaarWaveletV3AELG, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                            basis_offset, share_weights, activation, active_g,
                                            wavelet_type='haar',
                                            forecast_basis_dim=forecast_basis_dim, latent_dim=latent_dim)
  def forward(self, x):
    return super(HaarWaveletV3AELG, self).forward(x)

class DB2WaveletV3AELG(WaveletV3AELG):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None, latent_dim=5):
    super(DB2WaveletV3AELG, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                           basis_offset, share_weights, activation, active_g,
                                           wavelet_type='db2',
                                           forecast_basis_dim=forecast_basis_dim, latent_dim=latent_dim)
  def forward(self, x):
    return super(DB2WaveletV3AELG, self).forward(x)

class DB3WaveletV3AELG(WaveletV3AELG):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None, latent_dim=5):
    super(DB3WaveletV3AELG, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                           basis_offset, share_weights, activation, active_g,
                                           wavelet_type='db3',
                                           forecast_basis_dim=forecast_basis_dim, latent_dim=latent_dim)
  def forward(self, x):
    return super(DB3WaveletV3AELG, self).forward(x)

class DB4WaveletV3AELG(WaveletV3AELG):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None, latent_dim=5):
    super(DB4WaveletV3AELG, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                           basis_offset, share_weights, activation, active_g,
                                           wavelet_type='db4',
                                           forecast_basis_dim=forecast_basis_dim, latent_dim=latent_dim)
  def forward(self, x):
    return super(DB4WaveletV3AELG, self).forward(x)

class DB10WaveletV3AELG(WaveletV3AELG):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None, latent_dim=5):
    super(DB10WaveletV3AELG, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                            basis_offset, share_weights, activation, active_g,
                                            wavelet_type='db10',
                                            forecast_basis_dim=forecast_basis_dim, latent_dim=latent_dim)
  def forward(self, x):
    return super(DB10WaveletV3AELG, self).forward(x)

class DB20WaveletV3AELG(WaveletV3AELG):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None, latent_dim=5):
    super(DB20WaveletV3AELG, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                            basis_offset, share_weights, activation, active_g,
                                            wavelet_type='db20',
                                            forecast_basis_dim=forecast_basis_dim, latent_dim=latent_dim)
  def forward(self, x):
    return super(DB20WaveletV3AELG, self).forward(x)


class Coif1WaveletV3AELG(WaveletV3AELG):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None, latent_dim=5):
    super(Coif1WaveletV3AELG, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                             basis_offset, share_weights, activation, active_g,
                                             wavelet_type='coif1',
                                             forecast_basis_dim=forecast_basis_dim, latent_dim=latent_dim)
  def forward(self, x):
    return super(Coif1WaveletV3AELG, self).forward(x)

class Coif2WaveletV3AELG(WaveletV3AELG):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None, latent_dim=5):
    super(Coif2WaveletV3AELG, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                             basis_offset, share_weights, activation, active_g,
                                             wavelet_type='coif2',
                                             forecast_basis_dim=forecast_basis_dim, latent_dim=latent_dim)
  def forward(self, x):
    return super(Coif2WaveletV3AELG, self).forward(x)

class Coif3WaveletV3AELG(WaveletV3AELG):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None, latent_dim=5):
    super(Coif3WaveletV3AELG, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                             basis_offset, share_weights, activation, active_g,
                                             wavelet_type='coif3',
                                             forecast_basis_dim=forecast_basis_dim, latent_dim=latent_dim)
  def forward(self, x):
    return super(Coif3WaveletV3AELG, self).forward(x)

class Coif10WaveletV3AELG(WaveletV3AELG):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None, latent_dim=5):
    super(Coif10WaveletV3AELG, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                              basis_offset, share_weights, activation, active_g,
                                              wavelet_type='coif10',
                                              forecast_basis_dim=forecast_basis_dim, latent_dim=latent_dim)
  def forward(self, x):
    return super(Coif10WaveletV3AELG, self).forward(x)

class Symlet2WaveletV3AELG(WaveletV3AELG):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None, latent_dim=5):
    super(Symlet2WaveletV3AELG, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                               basis_offset, share_weights, activation, active_g,
                                               wavelet_type='sym2',
                                               forecast_basis_dim=forecast_basis_dim, latent_dim=latent_dim)
  def forward(self, x):
    return super(Symlet2WaveletV3AELG, self).forward(x)

class Symlet3WaveletV3AELG(WaveletV3AELG):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None, latent_dim=5):
    super(Symlet3WaveletV3AELG, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                               basis_offset, share_weights, activation, active_g,
                                               wavelet_type='sym3',
                                               forecast_basis_dim=forecast_basis_dim, latent_dim=latent_dim)
  def forward(self, x):
    return super(Symlet3WaveletV3AELG, self).forward(x)

class Symlet10WaveletV3AELG(WaveletV3AELG):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None, latent_dim=5):
    super(Symlet10WaveletV3AELG, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                                basis_offset, share_weights, activation, active_g,
                                                wavelet_type='sym10',
                                                forecast_basis_dim=forecast_basis_dim, latent_dim=latent_dim)
  def forward(self, x):
    return super(Symlet10WaveletV3AELG, self).forward(x)

class Symlet20WaveletV3AELG(WaveletV3AELG):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None, latent_dim=5):
    super(Symlet20WaveletV3AELG, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                                basis_offset, share_weights, activation, active_g,
                                                wavelet_type='sym20',
                                                forecast_basis_dim=forecast_basis_dim, latent_dim=latent_dim)
  def forward(self, x):
    return super(Symlet20WaveletV3AELG, self).forward(x)


# ---------------------------------------------------------------------------
# WaveletV3VAE2 — V3 wavelet blocks with VAE2RootBlock (compact VAE) backbone
# ---------------------------------------------------------------------------

class WaveletV3VAE2(VAE2RootBlock):
  """WaveletV3VAE2: orthonormal DWT basis block with compact VAE2 backbone.

  Structurally identical to WaveletV3AE but uses VAE2RootBlock instead of
  AERootBlock, replacing the deeper encoder-decoder with the more compact
  2-step encoder (units→units//2→latent_dim*2) and 2-step decoder. The frozen
  orthonormal DWT basis expansion is applied to the projection heads as in
  WaveletV3 and WaveletV3AE.

  Args:
      units (int): Width of the fully connected layers in the VAE2 backbone.
      backcast_length (int): Length of the historical input sequence.
      forecast_length (int): Length of the forecast horizon.
      basis_dim (int, optional): Number of DWT basis rows. Defaults to 32.
      basis_offset (int, optional): Row offset into SVD-ordered basis. Defaults to 0.
      share_weights (bool, optional): Share projection heads when dims match. Defaults to False.
      activation (str, optional): Activation for the VAE2 backbone. Defaults to 'ReLU'.
      active_g (bool or str, optional): Apply activation after basis expansion. Defaults to False.
      wavelet_type (str, optional): PyWavelets wavelet family string. Defaults to 'db3'.
      forecast_basis_dim (int, optional): Override basis_dim for forecast path only. Defaults to None.
      latent_dim (int, optional): Effective latent dimension for the VAE2 backbone. Defaults to 5.
  """
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False, wavelet_type='db3',
               forecast_basis_dim=None, latent_dim=5):
    super(WaveletV3VAE2, self).__init__(backcast_length, units, activation, latent_dim=latent_dim)
    self.wavelet_type = wavelet_type
    self.share_weights = share_weights
    self.active_g = active_g

    eff_back_dim = (backcast_length - min(basis_offset, backcast_length - 1)
                    if basis_dim is None
                    else min(basis_dim, backcast_length - min(basis_offset, backcast_length - 1)))
    fore_dim = forecast_basis_dim if forecast_basis_dim is not None else basis_dim
    eff_fore_dim = (forecast_length - min(basis_offset, forecast_length - 1)
                    if fore_dim is None
                    else min(fore_dim, forecast_length - min(basis_offset, forecast_length - 1)))

    self.backcast_linear = nn.Linear(units, eff_back_dim, bias=False)
    self.forecast_linear = nn.Linear(units, eff_fore_dim, bias=False)

    self.backcast_g = _WaveletGeneratorV3(backcast_length, wavelet_type=wavelet_type,
                                          basis_dim=eff_back_dim, basis_offset=basis_offset)
    self.forecast_g = _WaveletGeneratorV3(forecast_length, wavelet_type=wavelet_type,
                                          basis_dim=eff_fore_dim, basis_offset=basis_offset)

  def forward(self, x):
    x = super(WaveletV3VAE2, self).forward(x)
    backcast_thetas = self.backcast_linear(x)
    forecast_thetas = self.forecast_linear(x)
    backcast = self.backcast_g(backcast_thetas)
    forecast = self.forecast_g(forecast_thetas)
    if self.active_g:
      if self.active_g != 'forecast':
        backcast = self.activation(backcast)
      if self.active_g != 'backcast':
        forecast = self.activation(forecast)
    return backcast, forecast


# --- V3VAE2 Wavelet subclasses (thin wrappers setting wavelet_type) ---

class HaarWaveletV3VAE2(WaveletV3VAE2):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None, latent_dim=5):
    super(HaarWaveletV3VAE2, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                           basis_offset, share_weights, activation, active_g,
                                           wavelet_type='haar',
                                           forecast_basis_dim=forecast_basis_dim, latent_dim=latent_dim)
  def forward(self, x):
    return super(HaarWaveletV3VAE2, self).forward(x)

class DB2WaveletV3VAE2(WaveletV3VAE2):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None, latent_dim=5):
    super(DB2WaveletV3VAE2, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                          basis_offset, share_weights, activation, active_g,
                                          wavelet_type='db2',
                                          forecast_basis_dim=forecast_basis_dim, latent_dim=latent_dim)
  def forward(self, x):
    return super(DB2WaveletV3VAE2, self).forward(x)

class DB3WaveletV3VAE2(WaveletV3VAE2):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None, latent_dim=5):
    super(DB3WaveletV3VAE2, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                          basis_offset, share_weights, activation, active_g,
                                          wavelet_type='db3',
                                          forecast_basis_dim=forecast_basis_dim, latent_dim=latent_dim)
  def forward(self, x):
    return super(DB3WaveletV3VAE2, self).forward(x)

class DB4WaveletV3VAE2(WaveletV3VAE2):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None, latent_dim=5):
    super(DB4WaveletV3VAE2, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                          basis_offset, share_weights, activation, active_g,
                                          wavelet_type='db4',
                                          forecast_basis_dim=forecast_basis_dim, latent_dim=latent_dim)
  def forward(self, x):
    return super(DB4WaveletV3VAE2, self).forward(x)

class DB10WaveletV3VAE2(WaveletV3VAE2):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None, latent_dim=5):
    super(DB10WaveletV3VAE2, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                           basis_offset, share_weights, activation, active_g,
                                           wavelet_type='db10',
                                           forecast_basis_dim=forecast_basis_dim, latent_dim=latent_dim)
  def forward(self, x):
    return super(DB10WaveletV3VAE2, self).forward(x)

class DB20WaveletV3VAE2(WaveletV3VAE2):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None, latent_dim=5):
    super(DB20WaveletV3VAE2, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                           basis_offset, share_weights, activation, active_g,
                                           wavelet_type='db20',
                                           forecast_basis_dim=forecast_basis_dim, latent_dim=latent_dim)
  def forward(self, x):
    return super(DB20WaveletV3VAE2, self).forward(x)

class Coif1WaveletV3VAE2(WaveletV3VAE2):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None, latent_dim=5):
    super(Coif1WaveletV3VAE2, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                            basis_offset, share_weights, activation, active_g,
                                            wavelet_type='coif1',
                                            forecast_basis_dim=forecast_basis_dim, latent_dim=latent_dim)
  def forward(self, x):
    return super(Coif1WaveletV3VAE2, self).forward(x)

class Coif2WaveletV3VAE2(WaveletV3VAE2):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None, latent_dim=5):
    super(Coif2WaveletV3VAE2, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                            basis_offset, share_weights, activation, active_g,
                                            wavelet_type='coif2',
                                            forecast_basis_dim=forecast_basis_dim, latent_dim=latent_dim)
  def forward(self, x):
    return super(Coif2WaveletV3VAE2, self).forward(x)

class Coif3WaveletV3VAE2(WaveletV3VAE2):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None, latent_dim=5):
    super(Coif3WaveletV3VAE2, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                            basis_offset, share_weights, activation, active_g,
                                            wavelet_type='coif3',
                                            forecast_basis_dim=forecast_basis_dim, latent_dim=latent_dim)
  def forward(self, x):
    return super(Coif3WaveletV3VAE2, self).forward(x)

class Coif10WaveletV3VAE2(WaveletV3VAE2):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None, latent_dim=5):
    super(Coif10WaveletV3VAE2, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                             basis_offset, share_weights, activation, active_g,
                                             wavelet_type='coif10',
                                             forecast_basis_dim=forecast_basis_dim, latent_dim=latent_dim)
  def forward(self, x):
    return super(Coif10WaveletV3VAE2, self).forward(x)

class Symlet2WaveletV3VAE2(WaveletV3VAE2):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None, latent_dim=5):
    super(Symlet2WaveletV3VAE2, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                              basis_offset, share_weights, activation, active_g,
                                              wavelet_type='sym2',
                                              forecast_basis_dim=forecast_basis_dim, latent_dim=latent_dim)
  def forward(self, x):
    return super(Symlet2WaveletV3VAE2, self).forward(x)

class Symlet3WaveletV3VAE2(WaveletV3VAE2):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None, latent_dim=5):
    super(Symlet3WaveletV3VAE2, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                              basis_offset, share_weights, activation, active_g,
                                              wavelet_type='sym3',
                                              forecast_basis_dim=forecast_basis_dim, latent_dim=latent_dim)
  def forward(self, x):
    return super(Symlet3WaveletV3VAE2, self).forward(x)

class Symlet10WaveletV3VAE2(WaveletV3VAE2):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None, latent_dim=5):
    super(Symlet10WaveletV3VAE2, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                               basis_offset, share_weights, activation, active_g,
                                               wavelet_type='sym10',
                                               forecast_basis_dim=forecast_basis_dim, latent_dim=latent_dim)
  def forward(self, x):
    return super(Symlet10WaveletV3VAE2, self).forward(x)

class Symlet20WaveletV3VAE2(WaveletV3VAE2):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32, basis_offset=0,
               share_weights=False, activation='ReLU', active_g: bool = False,
               forecast_basis_dim=None, latent_dim=5):
    super(Symlet20WaveletV3VAE2, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                               basis_offset, share_weights, activation, active_g,
                                               wavelet_type='sym20',
                                               forecast_basis_dim=forecast_basis_dim, latent_dim=latent_dim)
  def forward(self, x):
    return super(Symlet20WaveletV3VAE2, self).forward(x)
