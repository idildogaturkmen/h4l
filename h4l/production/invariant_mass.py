# coding: utf-8

import functools
from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import EMPTY_FLOAT, set_ak_column
from columnflow.production.util import attach_coffea_behavior

# Task 3.
# Produce variables for ZZ, Z1, Z2
from h4l.util import build_2e2mu, build_4sf

np = maybe_import("numpy")
ak = maybe_import("awkward")

set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


@producer(
    uses=(
        {
            f"{field}.{var}"
            for field in ["Electron", "Muon"]
            for var in ["pt", "mass", "eta", "phi", "charge"]
        } | {
            attach_coffea_behavior,
        }
    ),
    produces={
        "m4l",
        "z1_mass",
        "z2_mass",
    },
)
def four_lep_invariant_mass(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Construct four-lepton invariant mass given the Electron and Muon arrays.
    """

    # attach coffea behavior for four-vector arithmetic
    events = self[attach_coffea_behavior](
        events,
        collections=["Electron", "Muon"],
        **kwargs,
    )

    electrons = events.Electron
    muons = events.Muon

    ele_plus = electrons[electrons.charge > 0]
    ele_minus = electrons[electrons.charge < 0]
    mu_plus = muons[muons.charge > 0]
    mu_minus = muons[muons.charge < 0]

    # Task 3.
    # Produce variables for ZZ, Z1, Z2
    # Hint: Build 2e2mu, 4e, 4mu separately and then zz_inclusive = ak.concatenate([zz_2e2mu, zz_4e, zz_4mu], axis=1)
    zz_2e2mu = build_2e2mu(mu_plus, mu_minus, ele_plus, ele_minus)
    zz_4e = build_4sf(ele_plus, ele_minus)
    zz_4mu = build_4sf(mu_plus, mu_minus)
    zz_inclusive = ak.concatenate([zz_2e2mu, zz_4e, zz_4mu], axis=1)

    # total number of leptons per event
    n_leptons = (
        ak.num(events.Electron, axis=1) +
        ak.num(events.Muon, axis=1)
    )

    # four-lepton mass, taking into account only events with at least four leptons,
    # and otherwise substituting a predefined EMPTY_FLOAT value
    # Task 3 Hint: ak.firsts(zz_inclusive.zz.mass) could be useful
    zz_mass = ak.firsts(zz_inclusive.zz.mass)
    z1_mass = ak.firsts(zz_inclusive.z1.mass)
    z2_mass = ak.firsts(zz_inclusive.z2.mass)

    fourlep_mass = ak.where(n_leptons >= 4, zz_mass, EMPTY_FLOAT)
    fourlep_mass = ak.fill_none(fourlep_mass, EMPTY_FLOAT)
    z1_mass = ak.fill_none(z1_mass, EMPTY_FLOAT)
    z2_mass = ak.fill_none(z2_mass, EMPTY_FLOAT)

    # write out the resulting mass to the `events` array,
    events = set_ak_column_f32(
        events,
        "m4l",
        fourlep_mass,
    )
    events = set_ak_column_f32(
        events,
        "z1_mass",
        z1_mass,
    )
    events = set_ak_column_f32(
        events,
        "z2_mass",
        z2_mass,
    )

    # return the events
    return events
