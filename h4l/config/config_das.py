# coding: utf-8

"""
Configuration of the HZZ4L DAS Exercise.
"""

from __future__ import annotations

import os
import re
from functools import partial
from typing import Set

import yaml
from scinum import Number
import order as od
import law

from columnflow.util import DotDict

from columnflow.production.cms.btag import BTagSFConfig

# from cmsdb.util import add_decay_process

from h4l.config.analysis_h4l import analysis_h4l
from h4l.config.categories import add_all_categories
from h4l.config.variables import add_variables

from columnflow.config_util import (
    get_root_processes_from_campaign, add_shift_aliases,
    get_shifts_from_sources,
    add_category, verify_config_processes,
)

thisdir = os.path.dirname(os.path.abspath(__file__))

def add_das_config(
    analysis: od.Analysis,
    campaign: od.Campaign,
    config_name: str | None = None,
    config_id: int | None = None,
    limit_dataset_files: int | None = None,
) -> od.Config:

    # TODO: Maybe add something for Run3 (pre/post EE/BPix)

    # get all root processes from campaign
    procs = get_root_processes_from_campaign(campaign)

    # create a config given the analysis
    cfg = analysis.add_config(campaign, name=config_name, id=config_id)

    # gather campaign data (year and specifics for pre/post VFP)
    year = campaign.x.year
    year_short = year%100

    corr_postfix = f"{campaign.x.vfp}VFP" if year == 2016 else ""

    # triggers required, sorted by primary dataset tag for recorded data
    cfg.x.trigger_matrix = [
        (
            "DoubleEG", {
                "Ele23_Ele12_CaloIdL_TrackIdL_IsoVL",
                "DoubleEle25_CaloIdL_MW",
            },
        ),
        (
            "DoubleMuon", {
                "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8",
            },
        ),
        (
            "MuonEG", {
                "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL",
                "Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
                "Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
                "DiMu9_Ele9_CaloIdL_TrackIdL_DZ",
            },
        ),
        (
            "SingleElectron", {
                "Ele32_WPTight_Gsf",
            },
        ),
        (
            "SingleMuon", {
                "IsoMu24",
            },
        ),
    ]

    # union of all triggers for use in MC
    cfg.x.all_triggers = {
        trigger
        for _, triggers in cfg.x.trigger_matrix
        for trigger in triggers
    }

    # add processes we are interested in
    process_names = [
        # data
        "data",
        # signals
        "h",
        "h_ggf_4l",  # GluGluHToZZTo4L_M125_TuneCP5_13TeV_powheg2_JHUGenV7011_pythia8
        "h_vbf_4l",  # VBF_HToZZTo4L_M125_TuneCP5_13TeV_powheg2_JHUGenV7011_pythia8
        "wph_4l",  # WplusH_HToZZTo4L_M125_TuneCP5_13TeV_powheg2-minlo-HWJ_JHUGenV7011_pythia8
        "wmh_4l",  # WminusH_HToZZTo4L_M125_TuneCP5_13TeV_powheg2-minlo-HWJ_JHUGenV7011_pythia8
        "zh_4l",  # ZH_HToZZ_4LFilter_M125_TuneCP5_13TeV_powheg2-minlo-HZJ_JHUGenV7011_pythia8
        "tth_4l",  # ttH_HToZZ_4LFilter_M125_TuneCP5_13TeV_powheg2_JHUGenV7011_pythia8
        "bbh_4l",  # bbH_HToZZTo4L_M125_TuneCP2_13TeV-jhugenv7011-pythia8
        "thq_4l",  # tqH_HToZZTo4L_M125_TuneCP5_13TeV-jhugenv7011-pythia8
        # backgrounds
        "zz_llll",  # ZZTo4L_TuneCP5_13TeV_powheg_pythia8
        "ggf",  # GluGluToContinToZZTo*_TuneCP5_13TeV-mcfm701-pythia8
        # TODO: dy, wz, tt
    ]

    # configuration of colors, labels, etc. can happen here
    process_styles = {
        # data
        "data": {
            "color1": "#000000",
            "label": r"Data",
        },
        # signals
        "h": {
            "color1": "#ff9999",
            "color2": "#555555",
            "label": r"H(125)",
        },
        "h_ggf_4l": {
            "color1": "#ff9999",
            "color2": "#555555",
            "label": r"gg $\rightarrow$ H $\rightarrow$ 4$\ell$",
        },
        # backgrounds
        "zz_llll": {
            "color1": "#99ccff",
            "color2": "#555555",
            "label": r"$q\overline{q}\rightarrow{ZZ}$, Z$\gamma^{*}$",
        },
        "ggf": {
            "color1": "#4b78ff",
            "color2": "#555555",
            "label": r"${gg}\rightarrow{ZZ}$, Z$\gamma^{*}$",
        },
        # TODO: dy, wz, tt (color: #669966, label: Z+X)
    }


    for process_name in process_names:
        # add the process
        proc = cfg.add_process(procs.get(process_name))

        # set style of some processes
        for attr in ("color1", "color2", "label"):
            val = process_styles.get(proc.name, {}).get(attr, None)
            if val is not None:
                setattr(proc, attr, val)

    # add datasets we need to study
    dataset_names = [
        # data
        "data_e_b",
        "data_e_c",
        "data_e_d",
        "data_e_e",
        "data_e_f",
        "data_double_egamma_b",
        "data_double_egamma_c",
        "data_double_egamma_d",
        "data_double_egamma_e",
        "data_double_egamma_f",
        "data_mu_b",
        "data_mu_c",
        "data_mu_d",
        "data_mu_e",
        "data_mu_f",
        "data_double_mu_b",
        "data_double_mu_c",
        "data_double_mu_d",
        "data_double_mu_e",
        "data_double_mu_f",
        "data_mu_egamma_b",
        "data_mu_egamma_c",
        "data_mu_egamma_d",
        "data_mu_egamma_e",
        "data_mu_egamma_f",
        # signals
        "h_ggf_4l_powheg",  # GluGluHToZZTo4L_M125_TuneCP5_13TeV_powheg2_JHUGenV7011_pythia8
        "h_vbf_4l_powheg",  # VBF_HToZZTo4L_M125_TuneCP5_13TeV_powheg2_JHUGenV7011_pythia8
        "wph_4l_powheg",  # WplusH_HToZZTo4L_M125_TuneCP5_13TeV_powheg2-minlo-HWJ_JHUGenV7011_pythia8
        "wmh_4l_powheg",  # WminusH_HToZZTo4L_M125_TuneCP5_13TeV_powheg2-minlo-HWJ_JHUGenV7011_pythia8
        "zh_4l_powheg",  # ZH_HToZZ_4LFilter_M125_TuneCP5_13TeV_powheg2-minlo-HZJ_JHUGenV7011_pythia8
        "tth_4l_powheg",  # ttH_HToZZ_4LFilter_M125_TuneCP5_13TeV_powheg2_JHUGenV7011_pythia8
        "bbh_4l_powheg",  # bbH_HToZZTo4L_M125_TuneCP2_13TeV-jhugenv7011-pythia8
        "thq_4l_powheg",  # tqH_HToZZTo4L_M125_TuneCP5_13TeV-jhugenv7011-pythia8
        # backgrounds
        "zz_llll_powheg",  # ZZTo4L_TuneCP5_13TeV_powheg_pythia8
        "ggf_4e_mcfm",  # GluGluToContinToZZTo4e_TuneCP5_13TeV-mcfm701-pythia8
        "ggf_4mu_mcfm",  # GluGluToContinToZZTo4mu_TuneCP5_13TeV-mcfm701-pythia8
        "ggf_4tau_mcfm",  # GluGluToContinToZZTo4tau_TuneCP5_13TeV-mcfm701-pythia8
        "ggf_2e2mu_mcfm",  # GluGluToContinToZZTo2e2mu_TuneCP5_13TeV-mcfm701-pythia8
        "ggf_2e2tau_mcfm",  # GluGluToContinToZZTo2e2tau_TuneCP5_13TeV-mcfm701-pythia8
        "ggf_2mu2tau_mcfm",  # GluGluToContinToZZTo2mu2tau_TuneCP5_13TeV-mcfm701-pythia8
    ]
    for dataset_name in dataset_names:
        # add the dataset
        dataset = cfg.add_dataset(campaign.get_dataset(dataset_name))

        # for testing purposes, limit the number of files
        if isinstance(limit_dataset_files, int) and limit_dataset_files > 0:
            for info in dataset.info.values():
                info.n_files = min(info.n_files, limit_dataset_files)

        # add tags for trigger selection
        if dataset_name.startswith("data_double_egamma_"):
            dataset.add_tag("DoubleEG")
        elif dataset_name.startswith("data_double_mu_"):
            dataset.add_tag("DoubleMuon")
        elif dataset_name.startswith("data_mu_egamma_"):
            dataset.add_tag("MuonEG")
        elif dataset_name.startswith("data_mu_"):
            dataset.add_tag("SingleMuon")
        elif dataset_name.startswith("data_e_"):
            dataset.add_tag("SingleElectron")

        # for each dataset, select which triggers to require
        # (and which to veto to avoid double counting events
        # in recorded data)
        if dataset.is_data:
            prev_triggers = set()
            for tag, triggers in cfg.x.trigger_matrix:
                if dataset.has_tag(tag):
                    dataset.x.require_triggers = triggers
                    dataset.x.veto_triggers = prev_triggers
                    break
                prev_triggers = prev_triggers | triggers

        elif dataset.is_mc:
            dataset.x.require_triggers = cfg.x.all_triggers

    # verify that the root process of all datasets is part of any of the registered processes
    verify_config_processes(cfg, warn=True)

    # default objects, such as calibrator, selector, producer, ml model, inference model, etc
    cfg.x.default_calibrator = "example" # TODO: Do we need any calibration?
    cfg.x.default_selector = "default"
    cfg.x.default_selector_steps = (
        "trigger",
        "four_leptons",
        "lepton_pt",
        "z_candidate",
        "z1_mass",
        "zz_mass",
    )
    cfg.x.default_producer = "default"
    cfg.x.default_hist_producer = "cf_default"
    cfg.x.default_ml_model = None
    cfg.x.default_inference_model = None
    cfg.x.default_categories = ("cat_incl",)
    cfg.x.default_variables = ("jet1_pt",)

    # process groups for conveniently looping over certain processs
    # (used in wrapper_factory and during plotting)
    cfg.x.process_groups = {}

    # dataset groups for conveniently looping over certain datasets
    # (used in wrapper_factory and during plotting)
    cfg.x.dataset_groups = {}

    # category groups for conveniently looping over certain categories
    # (used during plotting)
    cfg.x.category_groups = {}

    # variable groups for conveniently looping over certain variables
    # (used during plotting)
    cfg.x.variable_groups = {}

    # shift groups for conveniently looping over certain shifts
    # (used during plotting)
    cfg.x.shift_groups = {}

    # selector step groups for conveniently looping over certain steps
    # (used in cutflow tasks)
    cfg.x.selector_step_groups = {
        "default": [
            "trigger",
            "four_leptons",
            "lepton_pt",
            "z_candidate",
            "z1_mass",
            "zz_mass",
        ],
    }

    # selector step labels (for cutflow plots)
    cfg.x.selector_step_labels = {
        "four_leptons": r"4$\ell$",
        "lepton_pt": r"$p_T^{\ell 1}>20,\ p_T^{\ell 2}>10$",
        "z_candidate": r"$12<m_{\ell\ell}<120$",
        "z1_mass": r"$m_{Z1}>40$",
        "zz_mass": r"$m_{4\ell}>70$",
    }

    # custom method and sandbox for determining dataset lfns
    cfg.x.get_dataset_lfns = None
    cfg.x.get_dataset_lfns_sandbox = None

    # whether to validate the number of obtained LFNs in GetDatasetLFNs
    # (currently set to false because the number of files per dataset is truncated to 2)
    cfg.x.validate_dataset_lfns = False

    # lumi values in inverse pb
    # https://twiki.cern.ch/twiki/bin/view/CMS/LumiRecommendationsRun2?rev=2#Combination_and_correlations
    cfg.x.luminosity = Number(41480, {
        "lumi_13TeV_2017": 0.02j,
        "lumi_13TeV_1718": 0.006j,
        "lumi_13TeV_correlated": 0.009j,
    })

    # jec configuration
    # https://twiki.cern.ch/twiki/bin/view/CMS/JECDataMC?rev=201
    jerc_postfix = "APV" if year == 2016 and campaign.x.vfp == "post" else ""
    cfg.x.jec = DotDict.wrap({
        "campaign": f"Summer19UL{year_short}{jerc_postfix}",
        "version": {2016: "V7", 2017: "V5", 2018: "V5"}[year],
        "jet_type": "AK4PFchs",
        "levels": ["L1L2L3Res"],
        "levels_for_type1_met": ["L1FastJet"],
        "data_eras": sorted(filter(None, {
            d.x("jec_era", None)
            for d in cfg.datasets
            if d.is_data
        })),
        "uncertainty_sources": [
            "Total",
        ],
    })

    # JER
    # https://twiki.cern.ch/twiki/bin/view/CMS/JetResolution?rev=107
    cfg.x.jer = DotDict.wrap({
        "campaign": f"Summer19UL{year_short}{jerc_postfix}",
        "version": "JR" + {2016: "V3", 2017: "V2", 2018: "V2"}[year],
        "jet_type": "AK4PFchs",
    })

    # names of muon correction sets and working points
    # (used in the muon producer)
    cfg.x.muon_sf_names = ("NUM_TightRelIso_DEN_TightIDandIPCut", f"{year}_UL")

    # register shifts
    cfg.add_shift(name="nominal", id=0)

    # tune shifts are covered by dedicated, varied datasets, so tag the shift as "disjoint_from_nominal"
    # (this is currently used to decide whether ML evaluations are done on the full shifted dataset)
    cfg.add_shift(name="tune_up", id=1, type="shape", tags={"disjoint_from_nominal"})
    cfg.add_shift(name="tune_down", id=2, type="shape", tags={"disjoint_from_nominal"})

    # fake jet energy correction shift, with aliases flaged as "selection_dependent", i.e. the aliases
    # affect columns that might change the output of the event selection
    cfg.add_shift(name="jec_up", id=20, type="shape")
    cfg.add_shift(name="jec_down", id=21, type="shape")
    add_shift_aliases(
        cfg,
        "jec",
        {
            "Jet.pt": "Jet.pt_{name}",
            "Jet.mass": "Jet.mass_{name}",
            "MET.pt": "MET.pt_{name}",
            "MET.phi": "MET.phi_{name}",
        },
    )

    # event weights due to muon scale factors
    cfg.add_shift(name="mu_up", id=10, type="shape")
    cfg.add_shift(name="mu_down", id=11, type="shape")
    add_shift_aliases(cfg, "mu", {"muon_weight": "muon_weight_{direction}"})
    # event weights due to electron scale factors
    cfg.add_shift(name="e_up", id=12, type="shape")
    cfg.add_shift(name="e_down", id=13, type="shape")
    add_shift_aliases(cfg, "e", {"electron_weight": "electron_weight_{direction}"})
    # TODO #
    # Task 1.1
    # Add Electron SFs
    # Hint: modify event_weights below
    # Hint: modify production/default.py

    cfg.add_shift(name="murmuf_up", id=140, type="shape")
    cfg.add_shift(name="murmuf_down", id=141, type="shape")
    add_shift_aliases(
        cfg,
        "murmuf",
        {
            "murmuf_weight": "murmuf_weight_{direction}",
            "normalized_murmuf_weight": "normalized_murmuf_weight_{direction}",
        },
    )

    cfg.add_shift(name="ee_up", id=142, type="shape")
    cfg.add_shift(name="ee_down", id=143, type="shape")
    add_shift_aliases(
        cfg,
        "ee",
        {
            "ee_weight": "ee_weight_{direction}",
            "normalized_ee_weight": "normalized_ee_weight_{direction}",
        },
    )
    
    # external files
    json_mirror = "/afs/cern.ch/user/m/mrieger/public/mirrors/jsonpog-integration-c3be7e71"
    cfg.x.external_files = DotDict.wrap({
        # jet energy correction
        "jet_jerc": (f"{json_mirror}/POG/JME/{year}{corr_postfix}_UL/jet_jerc.json.gz", "v1"),

        # tau energy correction and scale factors
        "tau_sf": (f"{json_mirror}/POG/TAU/{year}{corr_postfix}_UL/tau.json.gz", "v1"),

        # electron scale factors
        "electron_sf": (f"{json_mirror}/POG/EGM/{year}{corr_postfix}_UL/electron.json.gz", "v1"),

        # muon scale factors
        "muon_sf": (f"{json_mirror}/POG/MUO/{year}{corr_postfix}_UL/muon_Z.json.gz", "v1"),

        # btag scale factor
        "btag_sf_corr": (f"{json_mirror}/POG/BTV/{year}{corr_postfix}_UL/btagging.json.gz", "v1"),

        # met phi corrector
        "met_phi_corr": (f"{json_mirror}/POG/JME/{year}{corr_postfix}_UL/met.json.gz", "v1"),

        # hh-btag repository (lightweight) with TF saved model directories
        "hh_btag_repo": ("https://github.com/hh-italian-group/HHbtag/archive/1dc426053418e1cab2aec021802faf31ddf3c5cd.tar.gz", "v1"),  # noqa
    })

    # external files with more complex year dependence
    if year == 2016:
        cfg.x.external_files.update(DotDict.wrap({
            # lumi files
            "lumi": {
                "golden": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions16/13TeV/Legacy_2016/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt", "v1"),  # noqa
                "normtag": ("/afs/cern.ch/user/l/lumipro/public/Normtags/normtag_PHYSICS.json", "v1"),
            },

            # https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJSONFileforData?rev=45#Pileup_JSON_Files_For_Run_II
            "pu": {
                "json": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions16/13TeV/PileUp/pileup_latest.txt", "v1"),  # noqa
                "mc_profile": ("https://raw.githubusercontent.com/cms-sw/cmssw/a65c2e1a23f2e7fe036237e2e34cda8af06b8182/SimGeneral/MixingModule/python/mix_2016_25ns_UltraLegacy_PoissonOOTPU_cfi.py", "v1"),  # noqa
                "data_profile": {
                    "nominal": (f"/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions16/13TeV/PileUp/UltraLegacy/PileupHistogram-goldenJSON-13tev-2016-{campaign.x.vfp}VFP-69200ub-99bins.root", "v1"),  # noqa
                    "minbias_xs_up": (f"/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions16/13TeV/PileUp/UltraLegacy/PileupHistogram-goldenJSON-13tev-2016-{campaign.x.vfp}VFP-72400ub-99bins.root", "v1"),  # noqa
                    "minbias_xs_down": (f"/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions16/13TeV/PileUp/UltraLegacy/PileupHistogram-goldenJSON-13tev-2016-{campaign.x.vfp}VFP-66000ub-99bins.root", "v1"),  # noqa
                },
            },
        }))
    elif year == 2017:
        cfg.x.external_files.update(DotDict.wrap({
            # lumi files
            "lumi": {
                "golden": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/Legacy_2017/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt", "v1"),  # noqa
                "normtag": ("/afs/cern.ch/user/l/lumipro/public/Normtags/normtag_PHYSICS.json", "v1"),
            },

            # https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJSONFileforData?rev=45#Pileup_JSON_Files_For_Run_II
            "pu": {
                "json": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/PileUp/UltraLegacy/pileup_latest.txt", "v1"),  # noqa
                "mc_profile": ("https://raw.githubusercontent.com/cms-sw/cmssw/435f0b04c0e318c1036a6b95eb169181bbbe8344/SimGeneral/MixingModule/python/mix_2017_25ns_UltraLegacy_PoissonOOTPU_cfi.py", "v1"),  # noqa
                "data_profile": {
                    "nominal": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/PileUp/UltraLegacy/PileupHistogram-goldenJSON-13tev-2017-69200ub-99bins.root", "v1"),  # noqa
                    "minbias_xs_up": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/PileUp/UltraLegacy/PileupHistogram-goldenJSON-13tev-2017-72400ub-99bins.root", "v1"),  # noqa
                    "minbias_xs_down": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/PileUp/UltraLegacy/PileupHistogram-goldenJSON-13tev-2017-66000ub-99bins.root", "v1"),  # noqa
                },
            },
        }))
    else:  # year 2018
        cfg.x.external_files.update(DotDict.wrap({
            # lumi files
            "lumi": {
                "golden": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions18/13TeV/Legacy_2018/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt", "v1"),  # noqa
                "normtag": ("/afs/cern.ch/user/l/lumipro/public/Normtags/normtag_PHYSICS.json", "v1"),
            },

            # https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJSONFileforData?rev=45#Pileup_JSON_Files_For_Run_II
            "pu": {
                "json": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions18/13TeV/PileUp/UltraLegacy/pileup_latest.txt", "v1"),  # noqa
                "mc_profile": ("https://raw.githubusercontent.com/cms-sw/cmssw/a65c2e1a23f2e7fe036237e2e34cda8af06b8182/SimGeneral/MixingModule/python/mix_2018_25ns_UltraLegacy_PoissonOOTPU_cfi.py", "v1"),  # noqa
                "data_profile": {
                    "nominal": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions18/13TeV/PileUp/UltraLegacy/PileupHistogram-goldenJSON-13tev-2018-69200ub-99bins.root", "v1"),  # noqa
                    "minbias_xs_up": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions18/13TeV/PileUp/UltraLegacy/PileupHistogram-goldenJSON-13tev-2018-72400ub-99bins.root", "v1"),  # noqa
                    "minbias_xs_down": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions18/13TeV/PileUp/UltraLegacy/PileupHistogram-goldenJSON-13tev-2018-66000ub-99bins.root", "v1"),  # noqa
                },
            },
        }))

    # target file size after MergeReducedEvents in MB
    cfg.x.reduced_file_size = 512.0

    # columns to keep after certain steps
    cfg.x.keep_columns = DotDict.wrap({
        "cf.ReduceEvents": {
            # general event info
            "run", "luminosityBlock", "event",
            # object info
            "Jet.btagDeepFlavB", "Jet.hadronFlavour",
            "Muon.pfRelIso04_all", "Muon.charge",
            "Electron.deltaEtaSC", "Electron.charge",
            "Electron.mvaFall17V2Iso", "Electron.mvaHZZIso",
            "MET.pt", "MET.phi", "MET.significance", "MET.covXX", "MET.covXY", "MET.covYY",
            "PV.npvs",
            # columns added during selection
            "deterministic_seed", "process_id", "mc_weight", "cutflow.*",
            "category_ids", "mc_weight", "pdf_weight*", "murmuf_weight*",
            "leptons_os", "single_triggered", "cross_triggered",
            "pu_weight*",
            "electron_weight*",
            "muon_weight*",
        } | {
            # four momenta information
            f"{field}.{var}"
            for field in ["Jet", "Muon", "Electron"]
            for var in ["pt", "eta", "phi", "mass", "e"]
        },
        "cf.MergeSelectionMasks": {
            "normalization_weight", "process_id", "category_ids", "cutflow.*",
        },
        "cf.UniteColumns": {
            "*",
        },
    })

    # names of electron correction sets and working points
    # (used in the electron_sf producer)
    cfg.x.electron_sf_names = ("UL-Electron-ID-SF", f"{year}{corr_postfix}", "wp80iso")

    # names of muon correction sets and working points
    # (used in the muon producer)
    cfg.x.muon_sf_names = ("NUM_TightRelIso_DEN_TightIDandIPCut", f"{year}{corr_postfix}_UL")

    # event weight columns as keys in an OrderedDict, mapped to shift instances they depend on
    # TODO: Add weights
    get_shifts = partial(get_shifts_from_sources, cfg)
    # TODO #
    # Task 1.1
    # Add Electron SFs
    # Hint: modify cfg L370
    # Hint: modify production/default.py
    cfg.x.event_weights = DotDict({
        "normalization_weight": [],
        "muon_weight": get_shifts("mu"),
        "electron_weight": get_shifts("e"),
    })

    # versions per task family, either referring to strings or to callables receving the invoking
    # task instance and parameters to be passed to the task family
    cfg.x.versions = {
        "cf.CalibrateEvents": "v0",
        # "cf.CalibrateEvents": "prod1",
        # "cf.SelectEvents": (lambda cls, inst, params: "prod1" if params.get("selector") == "default" else "dev1"),
        # ...
    }

    add_variables(cfg)
    add_all_categories(cfg)
