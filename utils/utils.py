import os
from pathlib import Path
from typing import Literal, Optional, List

import numpy as np
import math
import uproot as uproot
import pickle
import pandas as pd
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white"})
import matplotlib.colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import ticker
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
from matplotlib import gridspec
import seaborn as sns



# Cuts parameters

NU_SCORE_CUT = 0.5
BCFM_SCORE_CUT = 0.04

# n prims (>= / <=)
N_PRIM_PFP_MIN = 2
N_PRIM_PFP_MAX = 4
N_PRIM_TRKS_MIN = 1
N_PRIM_TRKS_MAX = 4
N_PRIM_SHWRS_CUT_MIN = 0
N_PRIM_SHWRS_CUT_MAX = 3

MIN_GAP = 1 #cm



# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

Operator = Literal[">", "<", ">=", "<=", "==", "!="]

@dataclass
class Cut:
    column: str | tuple[str,...]
    op: Operator
    value: float | int
    discrete: bool = False

    def apply(self, df: pd.DataFrame) -> pd.Series:
        ''' Returns a boolean series wether row passes the cut or not'''
        s = df[self.column]
        ops = {
            ">":  s.gt,
            "<":  s.lt,
            ">=": s.ge,
            "<=": s.le,
            "==": s.eq,
            "!=": s.ne,
        }
        return ops[self.op](self.value)


@dataclass
class RangeCut:
    column: str | tuple[str,...]

    lower: float | int
    upper: float | int

    lower_inclusive: bool = True
    upper_inclusive: bool = True
    discrete: bool = False

    def apply(self, df: pd.DataFrame) -> pd.Series:
        s = df[self.column]

        if self.lower_inclusive:
            lower_mask = s >= self.lower
        else:
            lower_mask = s > self.lower
        if self.upper_inclusive:
            upper_mask = s <= self.upper
        else:
            upper_mask = s < self.upper
        return lower_mask & upper_mask


@dataclass
class PlotConfig:
    var:         str
    label_col:   str              # columna que contiene la categoría
    bins:        int | np.ndarray = 30
    logy:        bool = False
    title:       Optional[str] = None
    xlabel:      Optional[str] = None
    ylabel:      str  = "Counts"
    weights_col: Optional[str] = None
    colors:      Optional[list] = None
    alpha:       float = 0.7
    discrete:    bool = False
    cut_val:     Optional[float | list[float]] = None 
    category_order: Optional[list] = None

def plot_stack(df: pd.DataFrame, cfg: PlotConfig, ax = None):
    if cfg.category_order:
        labels = [l for l in cfg.category_order if l in df[cfg.label_col].values]
    else:
        labels = df[cfg.label_col].unique().tolist()

    if cfg.colors and cfg.category_order:
        color_map = dict(zip(cfg.category_order, cfg.colors))
        colors = [color_map.get(l) for l in labels]
    else:
        colors = cfg.colors
        
    data = []
    weights = [] if cfg.weights_col else None

    for l in labels:
        mask = df[cfg.label_col] == l

        data.append(df.loc[mask, cfg.var].astype(float))

        if cfg.weights_col:
            weights.append(df.loc[mask, cfg.weights_col])

    if not ax:
        _, ax = plt.subplots()

    if cfg.discrete:
        vals = df[cfg.var].dropna().astype(int)
        bins = np.arange(vals.min(), vals.max() + 1.5) - 0.5
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_xticks(bins + 0.5)
    else:
        bins = cfg.bins

    if cfg.var == 'bcfm_score':
       bins = np.linspace(0,0.2,30)
    elif cfg.var == 'nu_score':
       bins = np.linspace(0,1,30)
        
    ax.hist(
        data,
        bins=bins,
        stacked=True,
        label=labels,
        weights=weights,
        alpha=cfg.alpha,
        #histtype="step",
        color=colors,
    )

    
    if cfg.cut_val is not None:
        cut_vals = np.atleast_1d(cfg.cut_val)  # funciona con float o list
        if cfg.discrete:
            cut_vals = cut_vals - 0.5
        for v in cut_vals:
            ax.axvline(v, color="black", linestyle="--", linewidth=1.5)

    ax.legend()
    ax.set_xlabel(cfg.xlabel or cfg.var)
    ax.set_ylabel(cfg.ylabel)

    if cfg.logy:
        ax.set_yscale('log')

    if cfg.title:
        ax.set_title(cfg.title)

    #plt.tight_layout()
    plt.show()


@dataclass
class AnalysisConfig:
    categories: dict = field(default_factory=lambda: {
        "signal" : ('is_lambda_signal', '', '', '', '', ''),
        # "no_hyperon": (('has_hyperon', '', '', '', '', ''), 0),
    })

    cuts: dict = field(default_factory=lambda: {
        'nu_score_cut' : Cut(('slc', 'nu_score', '', '', '', ''),'>', 0.5),
        'bcfm_cut' : Cut(('slc', 'barycenterFM', 'score', '', '', ''), '>', 0.04),
        'fv_cut' : Cut(('slc', 'in_fv', '','','',''), '==', 1, discrete=True),
        'prim_pfp_rcut' : RangeCut(('slc', 'n_prim_pfps', '','','',''), N_PRIM_PFP_MIN, N_PRIM_PFP_MAX,discrete=True),
        'prim_trks_rcut' : RangeCut(('slc', 'n_prim_trks', '','','',''), N_PRIM_TRKS_MIN, N_PRIM_TRKS_MAX,discrete=True),
        'prim_shwrs_rcut' : RangeCut(('slc', 'n_prim_shwrs', '','','',''), N_PRIM_SHWRS_CUT_MIN, N_PRIM_SHWRS_CUT_MAX,discrete=True),
        'gap_cut' : Cut(('slc', 'gap', '','','',''), '>', MIN_GAP),
    })
            
    palette: dict = field(default_factory=lambda: {
        "signal":       "#2ecc71",
        "background":   "#e74c3c",
        #"no_hyperon":   "#3498db",
        #"with_hyperon": "#e67e22",
    })

@dataclass
class AnalysisDriver:
    df:  pd.DataFrame
    cfg: AnalysisConfig

    def __post_init__(self):
        self.df = self.df.copy()
        self._build_categories()


    def get_final_df(self):
        return self._apply_cuts(self.cfg.cuts.values())

    def _build_categories(self):
        """Populate df['category'] from config masks."""
        #sig_col = self.cfg.signal_mask  # cambia signal_mask de query string a tupla/columna
    
        self.df["category"] = "background"
        #self.df.loc[self.df[sig_col] == 1, "category"] = "signal"
        not_used_mask = pd.Series(True, index=self.df.index)
        for label, (col, val) in self.cfg.categories.items():
            #bkg_mask = self.df[sig_col] != 1
            if isinstance(val, str):
                cat_mask = self.df.eval(f"`{col}`{val}")
            else:
                cat_mask = self.df[col] == val
            self.df.loc[cat_mask & not_used_mask, "category"] = label
            not_used_mask = not_used_mask & ~cat_mask

    def _apply_cuts(self, cuts) -> pd.DataFrame:
        df = self.df
        if not cuts:
            return df
        mask = pd.concat([c.apply(df) for c in cuts], axis=1).all(axis=1)
        return df[mask]
    
    def apply_cuts(self, cuts: dict | list | None = None) -> pd.DataFrame:
        if cuts is None:
            cuts = self.cfg.cuts.values()
        elif isinstance(cuts, dict):
            cuts = cuts.values()
        return self._apply_cuts(cuts)

    def plot(self, var: str, cuts: dict | None = None, **plot_kwargs):
        df_cut = self.apply_cuts(cuts)
        plot_cfg = PlotConfig(
            var=var,
            colors=[self.cfg.palette.get(c) for c in df_cut["category"].unique()],
            **plot_kwargs
        )
        plot_stack(df_cut, plot_cfg)

    def plot_cuts(self, **plot_kwargs):
        """Plot each cut variable before applying it, accumulating cuts."""
        active = []
        for name, cut in self.cfg.cuts.items():
            # plot the cut variable on the current (pre-cut) df
            active_cuts = [cut for _, cut in active]
            df_current = self.apply_cuts(active_cuts)
            var = cut.column  # asume que Cut tiene .column

            cut_val = None
            if isinstance(cut, Cut):
                cut_val = cut.value
            elif isinstance(cut, RangeCut):
                cut_val = [cut.lower, cut.upper]

            category_order=list(self.cfg.categories.keys())
                
            cfg = PlotConfig(
                var=var,
                label_col = 'category',
                xlabel = name,
                #title=f"Before '{name}'  (active cuts: {[n for n,_ in active] or 'none'})",
                colors=[self.cfg.palette.get(c) for c in category_order],
                discrete=cut.discrete,
                logy=True,
                cut_val=cut_val,
                #weights_col = '',
                category_order=category_order,
                **plot_kwargs
            )
            plot_stack(df_current, cfg)
            active.append((name, cut))
            

    def get_cuts_performance(self) -> pd.DataFrame:
        """
        Devuelve un DataFrame con la evolución de las métricas tras cada corte.
        """
    
        category_order = list(self.cfg.categories.keys())
        signal_category = "Signal"
    
        def get_amounts(df):
            sums = df.groupby("category")["scale"].sum()
            return {cat: sums.get(cat, 0.0) for cat in category_order}
    
        rows = []
    
        df_current = self.df.copy()
    
        initial_amounts = get_amounts(df_current)
        initial_signal = initial_amounts[signal_category]
        initial_bck = sum(
            amount
            for cat, amount in initial_amounts.items()
            if cat != signal_category
        )
    
        rows.append({
            "Stage": "Initial",
            "Cut value" : '[]',
            "Signal": initial_signal,
            **{
                cat: amount
                for cat, amount in initial_amounts.items()
                if cat != signal_category
            },
            "Cut efficiency (%)": 100, 
            "Total efficiency (%)": 100.0,
            "Purity (%)": (
                100.0 * initial_signal / (initial_signal + initial_bck)
                if initial_signal + initial_bck > 0
                else 0.0
            ),
            "Bck rejection (%)": 0.0,
        })

        prev_signal = initial_signal
    
        for name, cut in self.cfg.cuts.items():
            df_current = df_current[cut.apply(df_current)]
            
            amounts = get_amounts(df_current)
            signal = amounts[signal_category]
            bck = sum(
                amount
                for cat, amount in amounts.items()
                if cat != signal_category
            )

            c_val = cut.value if isinstance(cut, Cut) else None
            if isinstance(cut, Cut) and cut.discrete and cut.op == "<":
                c_val -= 1
    
            rows.append({
                "Stage": name,
                "Cut value" : str(c_val) if isinstance(cut, Cut) else f'[{cut.lower}, {cut.upper-1}]' ,
                "Signal": signal,
                **{
                    cat: amount
                    for cat, amount in amounts.items()
                    if cat != signal_category
                },
                "Cut efficiency (%)": (
                    100.0 * signal / prev_signal
                    if prev_signal > 0
                    else 0.0
                ),
                "Total efficiency (%)": (
                    100.0 * signal / initial_signal
                    if initial_signal > 0
                    else 0.0
                ),
                "Purity (%)": (
                    100.0 * signal / (signal + bck)
                    if signal + bck > 0
                    else 0.0
                ),
                "Bck rejection (%)": (
                    100.0 * (1 - bck / initial_bck)
                    if initial_bck > 0
                    else 0.0
                ),
            })

            prev_signal = signal
    
        return pd.DataFrame(rows)

