# coding: utf-8

"""
Definition of categories.

Categories are assigned a unique integer ID according to a fixed numbering
scheme, with digits/groups of digits indicating the different category groups:

TODO, document here.
"""

import law

from columnflow.util import maybe_import
from columnflow.config_util import create_category_combinations

from h4l.util import call_once_on_config

import order as od

logger = law.logger.get_logger(__name__)

np = maybe_import("numpy")
ak = maybe_import("awkward")

def name_fn(categories: dict[str, od.Category]):
  """Naming function for automatically generated combined categories."""
  return "__".join(cat.name for cat in categories.values() if cat)


def kwargs_fn(categories: dict[str, od.Category]):
  """Customization function for automatically generated combined categories."""
  return {
    "id": sum(cat.id for cat in categories.values()),
    "selection": [cat.selection for cat in categories.values()],
    "label": "\n".join(
      cat.label for cat in categories.values()
    ),
  }


def skip_fn(categories: dict[str, od.Category]):
  """Custom function for skipping certain category combinations."""
  return False  # don't skip

@call_once_on_config()
def add_all_categories(config: od.Config) -> None:
    add_incl_cat(config)
    add_lepton_categories(config)


@call_once_on_config()
def add_incl_cat(config: od.Config) -> None:
    cat_incl = config.add_category(  # noqa
      name="cat_incl",
      id=1,
      selection="catid_incl",
      label="Inclusive",
    )


@call_once_on_config()
def add_lepton_categories(config: od.Config) -> None:
    cat_4e = config.add_category(  # noqa
      name="4e",
      id=10,
      selection="catid_4e",
      label="4 Electrons",
    )

# Task 1.
# Define categories for 4mu and 2e2mu
# Hint: modify also categorization/default.py
    cat_4mu = config.add_category(
      name="4mu",
      id=20,
      selection="catid_4mu",
      label="4 Muons",
    )

    cat_2e2mu = config.add_category(
      name="2e2mu",
      id=30,
      selection="catid_2e2mu",
      label="2 Electrons 2 Muons",
    )