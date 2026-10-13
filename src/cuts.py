from dataclasses import dataclass, field
from typing import Literal, Optional, List, Any

import pandas as pd
import numpy as np

Operator = Literal[">", "<", ">=", "<=", "==", "!="]

@dataclass
class Cut:
    name: str
    variable: str
    operator: Operator
    value: float | int

    def apply(self, df: pd.DataFrame) -> pd.Series:
        ''' Returns a boolean series wether row passes the cut or not'''
        column = df[self.variable]
        df_op = {
            ">":  column.gt,
            "<":  column.lt,
            ">=": column.ge,
            "<=": column.le,
            "==": column.eq,
            "!=": column.ne,
        }
        return df_op[self.operator](self.value)


@dataclass
class RangeCut:
    name: str
    variable: str
    lower: float | int
    upper: float | int
    lower_inclusive: bool = True
    upper_inclusive: bool = True

    def apply(self, df: pd.DataFrame) -> pd.Series:
        s = df[self.variable]

        if self.lower_inclusive:
            lower_mask = s >= self.lower
        else:
            lower_mask = s > self.lower
        if self.upper_inclusive:
            upper_mask = s <= self.upper
        else:
            upper_mask = s < self.upper
        return lower_mask & upper_mask


def create_cut(conf):
    if conf['cut_type'] == 'cut':
        return Cut(name = conf['name'],
                   variable = conf['variable'],
                   operator = conf['operator'],
                   value = conf['value'])
        
    elif conf['cut_type'] == 'range':
        return RangeCut(name = conf['name'],
                        variable = conf['variable'],
                        lower = conf['lower'],
                        upper = conf['upper'],
                        lower_inclusive = conf['lower_inclusive'],
                        upper_inclusive = conf['upper_inclusive'])
    else:
        raise ValueError(f"Unknown cut type: {cut_type}")


@dataclass
class CutConfig:
    cut:               Cut | RangeCut
    title:             str | None = None
    xlabel:            str | None = None
    ylabel:            str = "Counts"
    weights_col:       str | None = None
    categ_col:         str = 'Category'
    categories:        dict[str, Any] = field(default_factory=dict)   
    categ_colors:      dict[str, Any] = field(default_factory=dict)   
    logy:              bool = False
    bins:              int | List | tuple | np.array = 30
    discrete:          bool = False
    split_signal:      bool = False
    plot_kwargs:       dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.bins, List) or isinstance(self.bins, tuple):
            if len(self.bins) >= 2 and len(self.bins) <= 3:
                self.bins = np.arange(*self.bins)

    
