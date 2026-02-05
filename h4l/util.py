# coding: utf-8

"""
Helpful utils.
"""

from __future__ import annotations

__all__ = ["IF_NANO_V9", "IF_NANO_V10"]

import re
import itertools
import time
from typing import Any, Hashable, Iterable, Callable
from functools import wraps, reduce, partial
import tracemalloc

import law

from columnflow.types import Any
from columnflow.columnar_util import ArrayFunction, deferred_column, get_ak_routes
from columnflow.util import maybe_import

np = maybe_import("numpy")
ak = maybe_import("awkward")
coffea = maybe_import("coffea")

_logger = law.logger.get_logger(__name__)

def build_2e2mu(muons_plus, muons_minus, electrons_plus, electrons_minus):
    mu1, mu2, e1, e2 = ak.unzip(
        ak.cartesian([muons_plus, muons_minus, electrons_plus, electrons_minus])
    )
    z_mu = mu1 + mu2
    z_e  = e1 + e2
    is_mu_closer = abs(z_mu.mass - 91.1876) < abs(z_e.mass - 91.1876)
    z1 = ak.where(is_mu_closer, z_mu, z_e)
    z2 = ak.where(is_mu_closer, z_e, z_mu)
    zz = z1 + z2
    return ak.zip({"z1": z1, "z2": z2, "zz": zz}, depth_limit=1)

def build_4sf(leptons_plus, leptons_minus):
    # build all distinct same-flavor OS pairings without jagged broadcasting issues
    plus_pairs = ak.combinations(leptons_plus, 2)
    minus_pairs = ak.combinations(leptons_minus, 2)
    pp, mm = ak.unzip(ak.cartesian([plus_pairs, minus_pairs], axis=1))

    z_a1 = pp["0"] + mm["0"]
    z_a2 = pp["1"] + mm["1"]
    z_b1 = pp["0"] + mm["1"]
    z_b2 = pp["1"] + mm["0"]

    z_mass = 91.1876
    is_a1_closer = abs(z_a1.mass - z_mass) < abs(z_a2.mass - z_mass)
    z1a = ak.where(is_a1_closer, z_a1, z_a2)
    z2a = ak.where(is_a1_closer, z_a2, z_a1)

    is_b1_closer = abs(z_b1.mass - z_mass) < abs(z_b2.mass - z_mass)
    z1b = ak.where(is_b1_closer, z_b1, z_b2)
    z2b = ak.where(is_b1_closer, z_b2, z_b1)

    z1 = ak.concatenate([z1a, z1b], axis=1)
    z2 = ak.concatenate([z2a, z2b], axis=1)
    zz = z1 + z2
    return ak.zip({"z1": z1, "z2": z2, "zz": zz}, depth_limit=1)


def masked_sorted_indices(mask: ak.Array, sort_var: ak.Array, ascending: bool = False) -> ak.Array:
  """
  Helper function to obtain the correct indices of an object mask
  """
  indices = ak.argsort(sort_var, axis=-1, ascending=ascending)
  return indices[mask[indices]]


def call_once_on_config(func=None, *, include_hash=False):
  """
  Parametrized decorator to ensure that function *func* is only called once for the config *config*.
  Can be used with or without parentheses.
  """
  if func is None:
    # If func is None, it means the decorator was called with arguments.
    def wrapper(f):
      return call_once_on_config(f, include_hash=include_hash)
    return wrapper

  @wraps(func)
  def inner(config, *args, **kwargs):
    tag = f"{func.__name__}_called"
    if include_hash:
      tag += f"_{func.__hash__()}"

    if config.has_tag(tag):
      return

    config.add_tag(tag)
    return func(config, *args, **kwargs)

  return inner

@deferred_column
def IF_NANO_V9(self, func: ArrayFunction) -> Any | set[Any]:
    return self.get() if func.config_inst.campaign.x.version == 9 else None


@deferred_column
def IF_NANO_V10(self, func: ArrayFunction) -> Any | set[Any]:
    return self.get() if func.config_inst.campaign.x.version >= 10 else None
