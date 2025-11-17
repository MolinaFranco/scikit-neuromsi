#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Device configuration utilities for GPU/CPU selection.

This module provides utilities to configure and manage device selection
(CPU/GPU) for both JAX (training) and BrainPy (simulation) backends.

Referencias:
-----------
JAX documentation: https://jax.readthedocs.io/en/latest/
BrainPy documentation: https://brainpy.readthedocs.io/
"""

import warnings
from typing import Literal, Optional, List

# Tipo para el dispositivo permitido
DeviceType = Literal["cpu", "gpu", "auto"]


class DeviceConfig:
    """
    Configuration manager for computational devices.

    This class handles device selection and configuration for both
    JAX and BrainPy backends, providing a unified interface for
    CPU/GPU selection.

    Attributes
    ----------
    device : str
        Current device setting ('cpu', 'gpu', or 'auto')
    jax_available : bool
        Whether JAX is available
    jax_gpu_available : bool
        Whether JAX has GPU support
    brainpy_available : bool
        Whether BrainPy is available

    Examples
    --------
    >>> config = DeviceConfig(device='auto')
    >>> config.setup()
    >>> print(config.get_info())
    """

    def __init__(self, device: DeviceType = "auto"):
        """
        Initialize device configuration.

        Parameters
        ----------
        device : {'cpu', 'gpu', 'auto'}, default='auto'
            Device to use for computations:
            - 'cpu': Force CPU usage
            - 'gpu': Force GPU usage (will raise error if not available)
            - 'auto': Automatically select best available device
        """
        self.device = device
        self.jax_available = False
        self.jax_gpu_available = False
        self.brainpy_available = False
        self._jax_devices = []

        # Detect available libraries
        self._detect_libraries()

    def _detect_libraries(self):
        """Detectar qué librerías están disponibles."""
        # Detectar JAX
        try:
            import jax
            self.jax_available = True
            self._jax = jax
            self._jax_devices = jax.devices()

            # Verificar si hay GPU disponible en JAX
            gpu_devices = [
                d for d in self._jax_devices if d.platform == 'gpu'
            ]
            self.jax_gpu_available = len(gpu_devices) > 0

        except ImportError:
            self.jax_available = False
            warnings.warn(
                "JAX not available. Install with: pip install jax"
            )

        # Detectar BrainPy
        try:
            import brainpy as bp
            self.brainpy_available = True
            self._brainpy = bp
        except ImportError:
            self.brainpy_available = False
            warnings.warn(
                "BrainPy not available. Install with: pip install brainpy"
            )

    def setup(self) -> bool:
        """
        Configure devices according to the specified device setting.

        Returns
        -------
        success : bool
            True if configuration was successful, False otherwise

        Raises
        ------
        RuntimeError
            If GPU is requested but not available
        """
        if self.device == "auto":
            return self._setup_auto()
        elif self.device == "gpu":
            return self._setup_gpu()
        elif self.device == "cpu":
            return self._setup_cpu()
        else:
            raise ValueError(
                f"Invalid device: {self.device}. "
                f"Must be 'cpu', 'gpu', or 'auto'"
            )

    def _setup_auto(self) -> bool:
        """Configuración automática: usar GPU si está disponible."""
        if self.jax_gpu_available:
            return self._setup_gpu()
        else:
            return self._setup_cpu()

    def _setup_gpu(self) -> bool:
        """Configurar para usar GPU."""
        success = True

        # Configurar JAX para GPU
        if self.jax_available:
            if self.jax_gpu_available:
                try:
                    # Configurar dispositivo por defecto a GPU
                    gpu_devices = [
                        d for d in self._jax_devices if d.platform == 'gpu'
                    ]
                    self._jax.config.update(
                        'jax_default_device',
                        gpu_devices[0]
                    )
                    print(f"JAX configurado para GPU: {gpu_devices[0]}")
                except Exception as e:
                    warnings.warn(f"Error configurando JAX GPU: {e}")
                    success = False
            else:
                if self.device == "gpu":
                    # Si el usuario pidió GPU explícitamente, es un error
                    raise RuntimeError(
                        "GPU requested but JAX GPU support not available. "
                        "Install with: pip install --upgrade 'jax[cuda12]'"
                    )
                else:
                    warnings.warn(
                        "JAX GPU not available, falling back to CPU"
                    )
                    success = False

        # Configurar BrainPy para GPU
        if self.brainpy_available:
            try:
                self._brainpy.math.set_platform('gpu')
                current = self._brainpy.math.get_platform()
                if current == 'gpu':
                    print("BrainPy configurado para GPU")
                else:
                    warnings.warn(
                        "BrainPy no pudo cambiar a GPU, usando CPU"
                    )
                    if self.device == "gpu":
                        success = False
            except Exception as e:
                warnings.warn(f"Error configurando BrainPy GPU: {e}")
                if self.device == "gpu":
                    success = False

        return success

    def _setup_cpu(self) -> bool:
        """Configurar para usar CPU."""
        # Configurar JAX para CPU
        if self.jax_available:
            try:
                cpu_devices = [
                    d for d in self._jax_devices if d.platform == 'cpu'
                ]
                if cpu_devices:
                    self._jax.config.update(
                        'jax_default_device',
                        cpu_devices[0]
                    )
                print("JAX configurado para CPU")
            except Exception as e:
                warnings.warn(f"Error configurando JAX CPU: {e}")
                return False

        # Configurar BrainPy para CPU
        if self.brainpy_available:
            try:
                self._brainpy.math.set_platform('cpu')
                print("BrainPy configurado para CPU")
            except Exception as e:
                warnings.warn(f"Error configurando BrainPy CPU: {e}")
                return False

        return True

    def get_jax_devices(self) -> List:
        """
        Get list of available JAX devices.

        Returns
        -------
        devices : list
            List of JAX devices
        """
        if self.jax_available:
            return self._jax_devices
        return []

    def get_info(self) -> str:
        """
        Get information about current device configuration.

        Returns
        -------
        info : str
            Human-readable device configuration information
        """
        lines = ["Device Configuration:"]
        lines.append(f"  Requested device: {self.device}")
        lines.append(f"  JAX available: {self.jax_available}")

        if self.jax_available:
            lines.append(
                f"  JAX GPU available: {self.jax_gpu_available}"
            )
            lines.append(f"  JAX devices: {self._jax_devices}")
            lines.append(
                f"  JAX default backend: {self._jax.default_backend()}"
            )

        lines.append(f"  BrainPy available: {self.brainpy_available}")

        if self.brainpy_available:
            current = self._brainpy.math.get_platform()
            lines.append(f"  BrainPy platform: {current}")

        return "\n".join(lines)


# Instancia global de configuración
_global_config: Optional[DeviceConfig] = None


def configure_device(device: DeviceType = "auto") -> DeviceConfig:
    """
    Configure computational device globally.

    This function sets up the device configuration for all subsequent
    operations in the package.

    Parameters
    ----------
    device : {'cpu', 'gpu', 'auto'}, default='auto'
        Device to use for computations

    Returns
    -------
    config : DeviceConfig
        The configured device manager

    Examples
    --------
    >>> # Use GPU if available, otherwise CPU
    >>> config = configure_device('auto')
    >>> print(config.get_info())

    >>> # Force CPU usage
    >>> config = configure_device('cpu')

    >>> # Force GPU usage (will error if not available)
    >>> config = configure_device('gpu')
    """
    global _global_config
    _global_config = DeviceConfig(device=device)
    _global_config.setup()
    return _global_config


def get_device_config() -> Optional[DeviceConfig]:
    """
    Get current global device configuration.

    Returns
    -------
    config : DeviceConfig or None
        Current device configuration, or None if not configured
    """
    return _global_config
