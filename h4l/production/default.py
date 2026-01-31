# coding: utf-8

"""
Column production methods related to higher-level features.
"""
import functools

from columnflow.production import Producer, producer
from columnflow.production.categories import category_ids
from columnflow.production.normalization import normalization_weights
from columnflow.production.util import attach_coffea_behavior
from columnflow.production.cms.seeds import deterministic_seeds
from columnflow.util import maybe_import

from columnflow.production.cms.electron import electron_weights
from columnflow.production.cms.muon import muon_weights


from h4l.production.invariant_mass import four_lep_invariant_mass

ak = maybe_import("awkward")
coffea = maybe_import("coffea")
maybe_import("coffea.nanoevents.methods.nanoaod")



@producer(
    uses={
        attach_coffea_behavior,
        deterministic_seeds,
        electron_weights, muon_weights,
        category_ids, normalization_weights,
        four_lep_invariant_mass,
        "process_id",
    },
    produces={
        attach_coffea_behavior,
        deterministic_seeds,
        electron_weights, muon_weights,
        category_ids, normalization_weights,
        four_lep_invariant_mass,
        "process_id"
    }
)
def default(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # Build categories
    events = self[category_ids](events, **kwargs)

    # deterministic seeds
    events = self[deterministic_seeds](events, **kwargs)

    if self.dataset_inst.is_mc:
        # normalization weights
        events = self[normalization_weights](events, **kwargs)
        
        # electron/muon scale factors using pT>15 masks
        events = self[electron_weights](events, electron_mask=(events.Electron.pt > 15), **kwargs)
        events = self[muon_weights](events, muon_mask=(events.Muon.pt > 15), **kwargs)

    events = self[four_lep_invariant_mass](events, **kwargs)

    return events
