import numpy as np
import pandas as pd
import scipy.special
from . import geometry

__all__ = [
    "convert_table_to_fixed",
    "basic_query",
    "basic_eval",
    "rename_column",
    "object_cross_cleaning",
    "shift_2dvector",
    "object_cross_dphi",
    "mindphi",
    "weight_sigmoid",
]

def convert_table_to_fixed(path, *tables):
    """Simply read in a dataframe and output as a fixed table for quicker IO"""
    with pd.HDFStore(path) as store:
        for table in tables:
            if table in store:
                df = store.select(table)

                # event index and/or object ID is saved in columns. Keep the dataframe
                # index as a separate thing. Unsure the best way to go, but the output
                # is a different file to the input right now
                df.reset_index(drop=True).to_hdf(
                    path.replace(".h5", "_v2.h5"), table, format='fixed',
                    append=False, complib='zlib', complevel=9,
                )

def basic_query(path, *cfgs):
    """Apply a query string to a dataframe and output into the same file"""
    for cfg in cfgs:
        df = pd.read_hdf(path, cfg["input"])
        df.query(cfg["query"]).reset_index(drop=True).to_hdf(
            path, cfg["output"], format='fixed', append=False,
            complib='zlib', complevel=9,
        )

def basic_eval(path, *cfgs):
    for cfg in cfgs:
        df = pd.read_hdf(path, cfg["input"])
        for eval_str in cfg["evals"]:
            df.eval(eval_str, inplace=True)
        df.to_hdf(
            path, cfg["output"], format='fixed', append=False,
            complib='zlib', complevel=9,
        )

def rename_column(path, *cfgs):
    for cfg in cfgs:
        df = (
            pd.read_hdf(path, cfg["input"])
            .rename(columns=cfg["rename"])
        )
        df.to_hdf(
            path, cfg["output"], format='fixed', append=False,
            complib='zlib', complevel=9,
        )

def drop_column(path, *cfgs):
    for cfg in cfgs:
        df = (
            pd.read_hdf(path, cfg["input"])
            .drop(cfg["drop"], axis=1)
        )
        df.to_hdf(
            path, cfg["output"], format='fixed', append=False,
            complib='zlib', complevel=9,
        )

def object_cross_cleaning(path, *cfgs):
    """
    Remove objects from the input collection which match to any objects in any
    of the reference collections through a distance match in eta-phi space with
    a configurable cut.
    """
    for cfg in cfgs:
        # Retain the original dataframe for writing the skimmed version out
        df_orig = pd.read_hdf(path, cfg["input"])
        df = df_orig[[
            "parent_event", "object_id",
            cfg["columns"].format("eta"),
            cfg["columns"].format("phi"),
        ]].sort_values(["parent_event", "object_id"]).copy(deep=True)
        df.columns = ["parent_event", "object_id1", "eta1", "phi1"]
        df["matched"] = False

        for cfg_ref in cfg["references"]:
            df_ref = pd.read_hdf(path, cfg_ref["table"])[[
                "parent_event", "object_id",
                cfg_ref["columns"].format("eta"),
                cfg_ref["columns"].format("phi"),
            ]].sort_values(["parent_event", "object_id"])
            df_ref.columns = ["parent_event", "object_id2", "eta2", "phi2"]

            df_cross = df.merge(df_ref, on='parent_event', how='left')
            dr = geometry.deltar(
                df_cross.eval("eta1-eta2").values,
                df_cross.eval("phi1-phi2").values,
            )

            # Remove warnings by making sure nans fail. I.e. inf < 0.4 is always
            # true
            dr[np.isnan(dr)] = np.inf
            df_cross["matched"] = (df < cfg["distance"])

            # Default set to no match (False) and then take the logical OR with
            # any matched object of this particular reference
            df["matched"] = df["matched"] | (
                df_cross.groupby(["parent_event", "object_id1"])["matched"].any().values
            )

        df_orig.loc[df["matched"],:].reset_index(drop=True).to_hdf(
            path, cfg["output"], format='fixed', append=False,
            complib='zlib', complevel=9,
        )

def shift_2dvector(path, **kwargs):
    # Save original for output
    df_orig = pd.read_hdf(path, kwargs["input"])

    for cfg in kwargs["args"]:
        df = df_orig[[
            cfg["variables"]["pt"], cfg["variables"]["phi"],
        ]].copy(deep=True)
        df.columns = ["pt", "phi"]

        px, py = geometry.radial_to_cartesian2d(
            df["pt"].values, df["phi"].values,
        )

        for cfg_shift in cfg["shifters"]:
            df_shift = pd.read_hdf(path, cfg_shift["table"])
            df_shift["pt"] = df_shift.eval(cfg_shift["variables"]["pt"])
            df_shift["phi"] = df_shift.eval(cfg_shift["variables"]["phi"])
            df_shift = df_shift[["parent_event", "object_id", "pt", "phi"]].copy(deep=True)

            opx, opy = geometry.radial_to_cartesian2d(
                df_shift["pt"].values, df_shift["phi"].values,
            )
            df_shift["px"] = opx
            df_shift["py"] = opy

            # Not all events have the objects leading to a mismatch in the
            # index between df_shift and df. Fixed by the reindex
            df_shift = (
                df_shift.groupby("parent_event")[["px", "py"]].sum()
                .reindex(df.index, fill_value=0.)
            )

            px += df_shift["px"].values
            py += df_shift["py"].values

        pt, phi = geometry.cartesian_to_radial2d(px, py)

        if cfg.get("relative", False):
            pt -= df_orig[cfg["variables"]["pt"]]
            phi -= df_orig[cfg["variables"]["phi"]]

        df_orig[cfg["output_variables"]["pt"]] = pt
        df_orig[cfg["output_variables"]["phi"]] = phi

    df_orig.to_hdf(
        path, kwargs["output"], format='fixed', append=False,
        complib='zlib', complevel=9,
    )

def object_cross_dphi(path, *cfgs):
    for cfg in cfgs:
        df_orig = pd.read_hdf(path, cfg["left"]["table"])
        df = df_orig[[
            "parent_event", "object_id", cfg["left"]["variable"],
        ]].copy(deep=True)
        df.columns = ["parent_event", "object_id", "phi"]

        if cfg.get("shift_only", False):
            df["phi1"] = 0.
        else:
            df["phi1"] = df["phi"]

        df_right = pd.read_hdf(path, cfg["right"]["table"])[[
            cfg["right"]["variable"]
        ]]
        df_right = df_right.reset_index()
        df_right.columns = ["parent_event", "phi2"]

        df_merge = df.merge(df_right, how='left', on='parent_event')
        df_orig[cfg["output_variable"]] = geometry.bound_phi(
            df_merge.eval("phi1-phi2").values
        )

        df_orig.to_hdf(
            path, cfg["left"]["table"], format='fixed', append=False,
            complib='zlib', complevel=9,
        )

def mindphi(path, *cfgs):
    for cfg in cfgs:
        df_orig = pd.read_hdf(path, cfg["output"]["table"])

        df = pd.read_hdf(path, cfg["input"]["table"])[[
            "parent_event", "object_id", cfg["input"]["variable"]
        ]]
        df.columns = ["parent_event", "object_id", "variable"]

        mask = (df["object_id"]<cfg["nobj"])
        df_min = df.loc[mask,:].groupby("parent_event").min()

        df_min = df_min.reindex(df_orig.index)
        df_orig[cfg["output"]["variable"]] = df_min["variable"]

        df_orig.to_hdf(
            path, cfg["output"]["table"], format='fixed', append=False,
            complib='zlib', complevel=9,
        )

def weight_sigmoid(path, *cfgs):
    for cfg in cfgs:
        df_orig = pd.read_hdf(path, cfg["input"]["table"])

        varname = cfg["input"]["variable"]
        variations = cfg["input"]["variations"]

        columns = [varname] + [
            "{}_sigma{}Up".format(varname, variation)
            for variation in variations
        ] + [
            "{}_sigma{}Down".format(varname, variation)
            for variation in variations
        ]

        df = df_orig[columns].copy(deep=True)
        df.columns = [c.replace(varname, "x") for c in columns]

        for v in variations:
            up = df["x_sigma{}Up".format(v)].values
            do = df["x_sigma{}Down".format(v)].values
            mask = (up>=do)
            df["x_sigma{}Up".format(v)] = np.where(mask, up, do)
            df["x_sigma{}Down".format(v)] = np.where(~mask, up, do)

        df["x_sigmaTotalUp"] = df.eval("sqrt(({})**2)".format(")**2 + (".join([
            "x_sigma{}Up".format(v) for v in variations
        ])))
        df["x_sigmaTotalDown"] = df.eval("-1*sqrt(({})**2)".format(")**2 + (".join([
            "x_sigma{}Down".format(v) for v in variations
        ])))

        df["delta"] = df["x"] - cfg["cut"]
        df["sigma"] = np.abs(np.where(
            df["delta"]>=0., df["x_sigmaTotalDown"], df["x_sigmaTotalUp"],
            ))

        wname = "weight_{}{}".format(varname, int(cfg["cut"]))
        df_orig[wname] = 0.5*(
            1.+scipy.special.erf(df["delta"]/(np.sqrt(2)*df["sigma"]))
        )
        for v in variations:
            df_orig["{}_{}Up".format(wname, v)] = 0.5*(
                1.+scipy.special.erf((df.eval("delta+x_sigma{}Up".format(v)))/(np.sqrt(2)*df["sigma"]))
            )
            df_orig["{}_{}Down".format(wname, v)] = 0.5*(
                1.+scipy.special.erf((df.eval("delta+x_sigma{}Down".format(v)))/(np.sqrt(2)*df["sigma"]))
            )

        df_orig.to_hdf(
            path, cfg["output_table"], format='fixed', append=False,
            complib='zlib', complevel=9,
        )

