from dataclasses import dataclass, field

import pandas as pd
import numpy as np

from .cuts import Cut, RangeCut, CutConfig
from .plotter import plot_stack 


@dataclass
class AnalysisDriver:
    df:  pd.DataFrame
    cfgs: list[CutConfig]

    def __post_init__(self):
        self.df = self.df.copy()
        self._build_categories()

    def _build_categories(self):
        """Populate df['category'] from config masks."""

        general_cfg = self.cfgs[0]
        self.df[general_cfg.categ_col] = "background"
        
        not_used_mask = pd.Series(True, index=self.df.index)
        for label, (col, val) in general_cfg.categories.items():
            #bkg_mask = self.df[sig_col] != 1
            if isinstance(val, str):
                cat_mask = self.df.eval(f"`{col}`{val}")
            else:
                cat_mask = self.df[col] == val
            self.df.loc[cat_mask & not_used_mask, general_cfg.categ_col] = label
            not_used_mask = not_used_mask & ~cat_mask

    def _apply_cuts(self, cuts) -> pd.DataFrame:
        df = self.df
        if not cuts:
            return df
        mask = pd.concat([c.apply(df) for c in cuts], axis=1).all(axis=1)
        return df[mask]

    
    def plot_cuts(self):
        """Plot each cut variable before applying it, accumulating cuts."""
        active_cuts = []
        for cfg in self.cfgs:
            # plot the cut variable on the current (pre-cut) df
            df_current = self._apply_cuts(active_cuts)
            plot_stack(df_current, cfg)
            active_cuts.append(cfg.cut)

            
    def get_final_df(self):
        cut_list = [cfg.cut for cfg in self.cfgs]
        return self._apply_cuts(cut_list)

    
    def get_cuts_performance(self, initial_signal = None, initial_bck = None, print_initial = False) -> pd.DataFrame: # HAY QUE ARREGLAR LA FUNCIÓN
        """
        Devuelve un DataFrame con la evolución de las métricas tras cada corte.
        """
        def _amount_format(value):
            if value // 1000 > 0:
                return f'{value:.1e}'
            else:
                return f'{value:.1f}'
    
        category_order = list(self.cfgs[0].categories.keys())
        signal_category = "Signal"
    
        def get_amounts(df):
            sums = df.groupby("category")["scale"].sum()
            return {cat: sums.get(cat, 0.0) for cat in category_order}
    
        rows = []
    
        df_current = self.df.copy()

        initial_amounts = get_amounts(df_current)
        local_initial_signal = initial_amounts[signal_category]
        local_initial_bck = sum(
            amount
            for cat, amount in initial_amounts.items()
            if cat != signal_category
        )

        if not initial_signal or not initial_bck:  
            initial_signal = local_initial_signal
            initial_bck = local_initial_bck

        if print_initial:
            rows.append({
                "Stage": "Initial",
                "Cut value" : '-',
                "Signal": _amount_format(initial_signal),
                **{
                    cat: _amount_format(amount)
                    for cat, amount in initial_amounts.items()
                    if cat != signal_category
                },
                "Cut efficiency (%)": 100, 
                "Total efficiency (%)": 100.0,
                "Purity (%)": (
                    f'{100.0 * initial_signal / (initial_signal + initial_bck):.2f}'
                    if initial_signal + initial_bck > 0
                    else 0.0
                ),
                "Cut bck rejection (%)": 0.0,
                "Total bck rejection (%)": 0.0,
            })

        prev_signal = local_initial_signal
        prev_back = local_initial_bck
    
        for cfg in self.cfgs:
            cut = cfg.cut
            df_current = df_current[cut.apply(df_current)]
            
            amounts = get_amounts(df_current)
            signal = amounts[signal_category]
            bck = sum(
                amount
                for cat, amount in amounts.items()
                if cat != signal_category
            )

    
            rows.append({
                "Stage": cut.name,
                "Cut value" : f'{cut.operator} {cut.value}' if isinstance(cut, Cut) else f'[{cut.lower}, {cut.upper-1}]' ,
                "Signal": _amount_format(signal),
                **{
                    cat: _amount_format(amount)
                    for cat, amount in amounts.items()
                    if cat != signal_category
                },
                "Cut efficiency (%)": (
                    f'{100.0 * signal / prev_signal:.2f}'
                    if prev_signal > 0
                    else 0.0
                ),
                "Total efficiency (%)": (
                    f'{100.0 * signal / initial_signal:.2f}'
                    if initial_signal > 0
                    else 0.0
                ),
                "Purity (%)": (
                    f'{100.0 * signal / (signal + bck):.2f}'
                    if signal + bck > 0
                    else 0.0
                ),
                "Cut bck rejection (%)": (
                    f'{100.0 * (1 - bck / prev_back):.2f}'
                    if initial_bck > 0
                    else 0.0
                ),
                "Total bck rejection (%)": (
                    f'{100.0 * (1 - bck / initial_bck):.4f}'
                    if initial_bck > 0
                    else 0.0
                ),
            })

            prev_signal = signal
            prev_back = bck
    
        return pd.DataFrame(rows)