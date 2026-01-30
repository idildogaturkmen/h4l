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
from columnflow.columnar_util import set_ak_column

from columnflow.production.cms.electron import electron_weights
from columnflow.production.cms.muon import muon_weights


from h4l.production.invariant_mass import four_lep_invariant_mass

ak = maybe_import("awkward")
coffea = maybe_import("coffea")
np = maybe_import("numpy")
maybe_import("coffea.nanoevents.methods.nanoaod")

set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


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
        
        #electrons
        high_pt_mask = events.Electron.pt > 15  # define mask for electrons
        events_high_pt = ak.with_field(events, events.Electron[high_pt_mask], "Electron")
        events_high_pt = self[electron_weights](events_high_pt, **kwargs)
        
        for postfix in ["", "_up", "_down"]:
              colname = f"{electron_weights.weight_name}{postfix}"
              if hasattr(events_high_pt, colname):
                sf_ele = getattr(events_high_pt, colname)
                events = set_ak_column(events, colname, sf_ele, value_type=np.float32)
        
        # Now repeat for muons
        high_pt_mask_mu = events.Muon.pt > 15  # define mask for muons
        events_high_pt_mu = ak.with_field(events, events.Muon[high_pt_mask_mu], "Muon")
        events_high_pt_mu = self[muon_weights](events_high_pt_mu, **kwargs)

        for postfix in ["", "_up", "_down"]:
              colname = f"{muon_weights.weight_name}{postfix}"
              if hasattr(events_high_pt_mu, colname):
                sf_mu = getattr(events_high_pt_mu, colname)
                events = set_ak_column(events, colname, sf_mu, value_type=np.float32)

    events = self[four_lep_invariant_mass](events, **kwargs)

    return events

