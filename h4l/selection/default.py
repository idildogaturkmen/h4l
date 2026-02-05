from operator import and_
from functools import reduce
from collections import defaultdict
from typing import Tuple

from columnflow.util import maybe_import

from columnflow.selection.stats import increment_stats
from columnflow.selection import Selector, SelectionResult, selector
from columnflow.selection.cms.met_filters import met_filters
from columnflow.selection.cms.json_filter import json_filter
from columnflow.selection.cms.jets import jet_veto_map

from columnflow.production.categories import category_ids
from columnflow.production.util import attach_coffea_behavior
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.cms.electron import electron_weights
from columnflow.production.cms.muon import muon_weights
from columnflow.production.processes import process_ids

from h4l.selection.lepton import electron_selection, muon_selection
from h4l.selection.trigger import trigger_selection

from h4l.util import build_2e2mu, build_4sf

np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector(
    uses={
        "event",
        category_ids,
        attach_coffea_behavior, json_filter, mc_weight,
        electron_weights, muon_weights,
        electron_selection, muon_selection,
        trigger_selection,
        increment_stats, process_ids
    },
    produces={
        category_ids,
        attach_coffea_behavior, json_filter, mc_weight,
        electron_weights, muon_weights,
        electron_selection, muon_selection,
        trigger_selection,
        increment_stats, process_ids,
    },
    # sandbox=dev_sandbox("bash::$CF_BASE/sandboxes/venv_columnar.sh"),
    exposed=True,
)
def default(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    # ensure coffea behaviors are loaded
    events = self[attach_coffea_behavior](events, **kwargs)
    events = self[category_ids](events, **kwargs)

    # add corrected mc weights
    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)

    # initialize `SelectionResult` object
    results = SelectionResult()

    # filter bad data events according to golden lumi mask
    if self.dataset_inst.is_data:
        events, json_filter_results = self[json_filter](events, **kwargs)
        results += json_filter_results

    # run trigger selection
    events, trigger_results = self[trigger_selection](events, call_force=True, **kwargs)
    results += trigger_results

    # run electron selection
    events, ele_results = self[electron_selection](events, call_force=True, **kwargs)
    results += ele_results

    # run muon selection
    events, muon_results = self[muon_selection](events, call_force=True, **kwargs)
    results += muon_results

    # get indices of selected leptons
    ele_idx = results.objects.Electron.Electron
    muon_idx = results.objects.Muon.Muon

    # select leptons
    electrons = events.Electron[ele_idx]
    muons = events.Muon[muon_idx]

    # add lepton SFs only for selected leptons to avoid out-of-range inputs
    if self.dataset_inst.is_mc:
        ele_local_idx = ak.local_index(events.Electron.pt)
        mu_local_idx = ak.local_index(events.Muon.pt)
        electron_mask = ak.any(ele_local_idx[..., None] == ele_idx[:, None, :], axis=-1)
        muon_mask = ak.any(mu_local_idx[..., None] == muon_idx[:, None, :], axis=-1)
        # SFs are only valid in a defined kinematic range; guard against out-of-bounds
        ele_sc_eta = abs(events.Electron.eta + events.Electron.deltaEtaSC)
        electron_mask = electron_mask & (events.Electron.pt >= 10.0) & (ele_sc_eta < 2.5)
        # muon SFs are typically valid only above ~15 GeV in UL; guard against low-pt
        muon_mask = muon_mask & (events.Muon.pt >= 15.0) & (abs(events.Muon.eta) < 2.4)
        events = self[electron_weights](events, electron_mask=electron_mask, **kwargs)
        events = self[muon_weights](events, muon_mask=muon_mask, **kwargs)

    # count selected leptons
    n_ele = ak.num(electrons, axis=1)
    n_muon = ak.num(muons, axis=1)

    # select events with at least four selected leptons
    results.steps["four_leptons"] = (n_ele + n_muon) >= 4

    # Task 2.
    # Implement official HZZ Selection
    
    # leading/subleading lepton pT requirement
    leptons = ak.concatenate([electrons, muons], axis=1)
    leptons = leptons[ak.argsort(leptons.pt, axis=1, ascending=False)]
    leptons = ak.pad_none(leptons, 2, axis=1)
    lead_pt = leptons.pt[:, 0]
    sublead_pt = leptons.pt[:, 1]
    # Bonus: Leading lepton must have pT > 20 GeV, subleading pT > 10 GeV
    results.steps["lepton_pt"] = ak.fill_none((lead_pt > 20) & (sublead_pt > 10), False)

    ele_plus = electrons[electrons.charge > 0]
    ele_minus = electrons[electrons.charge < 0]
    mu_plus = muons[muons.charge > 0]
    mu_minus = muons[muons.charge < 0]

    zz_2e2mu = build_2e2mu(mu_plus, mu_minus, ele_plus, ele_minus)
    zz_4e = build_4sf(ele_plus, ele_minus)
    zz_4mu = build_4sf(mu_plus, mu_minus)
    zz_inclusive = ak.concatenate([zz_2e2mu, zz_4e, zz_4mu], axis=1)

    # All Z candidates must have 12 < mll < 120 GeV
    z_mass_mask = (
        (zz_inclusive.z1.mass > 12) & (zz_inclusive.z1.mass < 120) &
        (zz_inclusive.z2.mass > 12) & (zz_inclusive.z2.mass < 120)
    )
    # The Z1 candidate must have mZ1 > 40 GeV
    z1_mass_mask = z_mass_mask & (zz_inclusive.z1.mass > 40)
    # The ZZ candidate must have mZZ > 70 GeV
    zz_mass_mask = z1_mass_mask & (zz_inclusive.zz.mass > 70)

    results.steps["z_candidate"] = ak.fill_none(ak.any(z_mass_mask, axis=1), False)
    results.steps["z1_mass"] = ak.fill_none(ak.any(z1_mass_mask, axis=1), False)
    results.steps["zz_mass"] = ak.fill_none(ak.any(zz_mass_mask, axis=1), False)

    # post selection build process IDs
    events = self[process_ids](events, **kwargs)

    # final event selection mask is AND of all selection steps
    results.event = reduce(and_, results.steps.values())
    results.event = ak.fill_none(results.event, False)

    weight_map = {
      "num_events": Ellipsis,
      "num_events_selected": results.event,
    }
    group_map = {}
    if self.dataset_inst.is_mc:
      weight_map = {
          **weight_map,
          # mc weight for all events
          "sum_mc_weight": (events.mc_weight, Ellipsis),
          "sum_mc_weight_selected": (events.mc_weight, results.event),
      }
      group_map = {
          # per process
          "process": {
              "values": events.process_id,
               "mask_fn": (lambda v: events.process_id == v),
          },
      }

    events, results = self[increment_stats](
        events,
        results,
        stats,
        weight_map=weight_map,
        group_map=group_map,
        **kwargs,
    )

    return events, results
