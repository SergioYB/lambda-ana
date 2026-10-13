from dataclasses import dataclass, field
from typing import List, Any

import matplotlib.pyplot as plt

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white"})

import matplotlib.colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import ticker
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
#from matplotlib import gridspecs
import pandas as pd
import numpy as np

from .cuts import Cut, RangeCut, CutConfig



def _plot_cut(cut: Cut, discrete: bool, ax: plt.Axes) -> None:
    val = cut.value

    if discrete and "=" not in cut.operator:
        if ">" in cut.operator:
            val += 0.5
        elif "<" in cut.operator:
            val -= 0.5
    elif discrete and "=" in cut.operator:
        if ">" in cut.operator:
            val -= 0.5
        elif "<" in cut.operator:
            val += 0.5
            
    ax.axvline(val, color="black", linestyle="--", linewidth=1.5)
    
    if ">" in cut.operator:
        ax.axvspan(val, ax.get_xlim()[1], alpha=0.2)
    elif "<" in cut.operator:
        ax.axvspan(ax.get_xlim()[0], val, alpha=0.2)

    if "==" in cut.operator:
        ax.axvspan(val-0.5, val+0.5, alpha=0.2)


def _plot_range_cut(rcut: RangeCut, discrete: bool, ax: plt.Axes) -> None:
    lower = rcut.lower
    upper = rcut.upper

    if not rcut.lower_inclusive:
        lower += 1
    if not rcut.upper_inclusive:
        upper -= 1

    if discrete:
        lower -= 0.5
        upper += 0.5
        
    ax.axvline(lower, color="black", linestyle="--", linewidth=1.5)
    ax.axvline(upper, color="black", linestyle="--", linewidth=1.5)

    ax.axvspan(lower, upper, alpha=0.2)


def _draw_plot(data: dict[str, Any], cfg: CutConfig, ax: plt.Axes) -> None:
    '''Auxiliary function to dray one subplot (called twice if split_signal is true'''
    
    if cfg.discrete:
        bins = np.arange(cfg.bins.min(), cfg.bins.max() + 1.5) - 0.5
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_xticks(bins + 0.5)
    else:
        bins = cfg.bins

    ax.hist(
        data['values'],
        bins=bins,
        stacked=True,
        label=data['categories'],
        weights=data['weights'],
        color=data['colors'],
        **cfg.plot_kwargs
    )

    
    # Draw cut thresholds
    if isinstance(cfg.cut, Cut):
        _plot_cut(cfg.cut, cfg.discrete, ax)
    elif isinstance(cfg.cut, RangeCut):
        _plot_range_cut(cfg.cut, cfg.discrete, ax)

    ax.set_xlabel(cfg.xlabel or cfg.cut.name)
    ax.set_ylabel(cfg.ylabel)

    if cfg.logy and len(data['categories']) > 1:
        ax.set_yscale('log')
        ax.set_ylim(bottom=5e-1)

    if cfg.cut.variable == 'InvariantMass':
        lambda_mass = 1115.68
        ax.axvline(lambda_mass, color="magenta", linestyle="--", linewidth=1.5, label=rf'$\Lambda$ mass = {lambda_mass} MeV')

    if cfg.title:
        ax.set_title(cfg.title)
    ax.legend()


def _draw_fom(signal_v, signal_w, bck_v, bck_w, bins, operator, ax):
    
    if "<" not in operator and ">" not in operator:
        return

    s_counts = []
    b_counts = []
    
    if "<" in operator:
        s_counts = counts = np.array([
            signal_w[signal_v < b].sum()
            for b in bins
        ])
        b_counts = counts = np.array([
            bck_w[bck_v < b].sum()
            for b in bins
        ])
        
    if ">" in operator:
        s_counts = counts = np.array([
            signal_w[signal_v > b].sum()
            for b in bins
        ])
        b_counts = counts = np.array([
            bck_w[bck_v > b].sum()
            for b in bins
        ])

    # figure of merit
    fom = np.zeros_like(s_counts, dtype=float)

    np.divide(
        s_counts**2,
        s_counts + b_counts,
        out=fom,
        where=(s_counts + b_counts) != 0
    )
    
    max_fom = np.max(fom)
    max_bin = bins[np.argmax(fom)]

    ax.plot(bins, fom, lw=2, color='black', label=f'Opt cut = {max_bin:.1f}')
    ax.set_xlabel('Cut value')
    ax.set_ylabel('FOM')
    ax.legend()
    
        

def plot_stack(df: pd.DataFrame, cfg: CutConfig) -> None:
    '''Given a DataFrame and a list of cut configurations it plots each cut variable and applies it'''
    
    categories = [categ for categ in cfg.categories.keys() if categ in df[cfg.categ_col].values]

    data = {}
    signal_data={}

    if cfg.split_signal:
        categories.remove('Signal')
        signal_data = {
            'categories': ['Signal'],
            'values': df.loc[df[cfg.categ_col] == 'Signal', cfg.cut.variable].values,
            'weights': df.loc[df[cfg.categ_col] == 'Signal', cfg.weights_col].values,
            'colors': cfg.categ_colors['Signal']
        }

    values = [
        df.loc[df[cfg.categ_col] == cat, cfg.cut.variable]
        for cat in categories
    ]

    weights = None
    if cfg.weights_col:
        weights = [
            df.loc[df[cfg.categ_col] == cat, cfg.weights_col]
            for cat in categories
        ]

    colors = [cfg.categ_colors.get(cat) for cat in categories]

    data = {
        'categories': categories,
        'values': values,
        'weights': weights,
        'colors': colors,
    }

    bck_v = np.concatenate([
        values
        for values, category in zip(data['values'], data['categories'])
        if category != 'Signal'
    ])
    
    bck_w = np.concatenate([
        w
        for w, category in zip(data['weights'], data['categories'])
        if category != 'Signal'
    ])

    signal_v = (
        np.concatenate([
            values
            for values, category in zip(data['values'], data['categories'])
            if category == 'Signal'
        ])
    ) if not cfg.split_signal else signal_data['values']

    signal_w = (
        np.concatenate([
            w
            for w, category in zip(data['weights'], data['categories'])
            if category == 'Signal'
        ])
    ) if not cfg.split_signal else signal_data['weights']

    

    axes = None
    if cfg.split_signal:
        _, axes = plt.subplots(1,3, figsize=(18,5))
        _draw_plot(signal_data, cfg, axes[1])
    else:
        _, axes = plt.subplots(1,2, figsize=(12,5))
        
    _draw_plot(data, cfg, axes[0])

    if isinstance(cfg.cut, Cut) and not cfg.discrete:
        _draw_fom(signal_v, signal_w, bck_v, bck_w, cfg.bins, cfg.cut.operator, axes[-1])
    
    plt.show()
    
    