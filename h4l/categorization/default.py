# coding: utf-8

"""
H4L Categorization methods.
"""

from columnflow.categorization import Categorizer, categorizer
from columnflow.util import maybe_import

ak = maybe_import("awkward")


#
# categorizer functions used by categories definitions
#

@categorizer(uses={"event"})
def catid_incl(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
      mask_4e = (ak.num(events.Electron, axis=-1) == 4) & (ak.num(events.Muon, axis=-1) == 0)
      mask_4mu = (ak.num(events.Electron, axis=-1) == 0) & (ak.num(events.Muon, axis=-1) == 4)
      mask_2e2mu = (ak.num(events.Electron, axis=-1) == 2) & (ak.num(events.Muon, axis=-1) == 2)
      mask = mask_4e | mask_4mu | mask_2e2mu
      return events, mask

@categorizer(uses={"event"}, call_force=True)
def catid_4e(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
      mask = (ak.num(events.Electron, axis=-1) == 4) & (ak.num(events.Muon, axis=-1) == 0)
      return events, mask

# Task 1. Define categories for 4mu and 2e2mu
# Hint: modify also config/categories.py

@categorizer(uses={"event"}, call_force=True)
def catid_4mu(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
      mask = (ak.num(events.Electron, axis=-1) == 0) & (ak.num(events.Muon, axis=-1) == 4)
      return events, mask

@categorizer(uses={"event"}, call_force=True)
def catid_2e2mu(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
      mask = (ak.num(events.Electron, axis=-1) == 2) & (ak.num(events.Muon, axis=-1) == 2)
      return events, mask
