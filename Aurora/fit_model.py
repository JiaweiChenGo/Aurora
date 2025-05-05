from typing import List,Mapping,Optional
from anndata import AnnData

import os
import numpy as np
import pandas as pd
import scanpy as sc
from sparse import COO

import scipy.sparse
from sklearn.preprocessing import normalize

from .utils import config, logged,prod
from .model import MyModel

def aggregate_obs(
        adata: AnnData, 
        by: str, 
        X_agg: Optional[str] = "sum",
        obs_agg: Optional[Mapping[str, str]] = None,
        obsm_agg: Optional[Mapping[str, str]] = None,
        layers_agg: Optional[Mapping[str, str]] = None
) -> AnnData:
    obs_agg = obs_agg or {}
    obsm_agg = obsm_agg or {}
    layers_agg = layers_agg or {}

    by = adata.obs[by]
    agg_idx = pd.Index(by.cat.categories) \
        if pd.api.types.is_categorical_dtype(by) \
        else pd.Index(np.unique(by))
    agg_sum = scipy.sparse.coo_matrix((
        np.ones(adata.shape[0]), (
            agg_idx.get_indexer(by),
            np.arange(adata.shape[0])
        )
    )).tocsr()
    agg_mean = agg_sum.multiply(1 / agg_sum.sum(axis=1))

    agg_method = {
        "sum": lambda x: agg_sum @ x,
        "mean": lambda x: agg_mean @ x,
        "majority": lambda x: pd.crosstab(by, x).idxmax(axis=1).loc[agg_idx].to_numpy()
    }

    X = agg_method[X_agg](adata.X) if X_agg and adata.X is not None else None
    obs = pd.DataFrame({
        k: agg_method[v](adata.obs[k])
        for k, v in obs_agg.items()
    }, index=agg_idx.astype(str))
    obsm = {
        k: agg_method[v](adata.obsm[k])
        for k, v in obsm_agg.items()
    }
    layers = {
        k: agg_method[v](adata.layers[k])
        for k, v in layers_agg.items()
    }
    for c in obs:
        if pd.api.types.is_categorical_dtype(adata.obs[c]):
            obs[c] = pd.Categorical(obs[c], categories=adata.obs[c].cat.categories)
    return AnnData(
        X=X, obs=obs, var=adata.var,
        obsm=obsm, varm=adata.varm, layers=layers
    )

def estimate_balancing_weight(
        *adatas, 
        use_rep: str = None,
        resolution: float = 1.0, 
        cutoff: float = 0.5, 
        power: float = 4.0,
        key_added: str = "balancing_weight"
) -> None:
    if use_rep is None:
        raise ValueError("Missing required argument `use_rep`!")
    adatas_ = [
        AnnData(
            obs=adata.obs.copy(deep=False).assign(n=1),
            obsm={use_rep: adata.obsm[use_rep]}
        ) for adata in adatas
    ] 
    for adata_ in adatas_:
        sc.pp.neighbors(
            adata_, n_pcs=adata_.obsm[use_rep].shape[1],
            use_rep=use_rep, metric="cosine"
        )
        sc.tl.leiden(adata_, resolution=resolution)

    leidens = [
        aggregate_obs(
            adata, by="leiden", X_agg=None,
            obs_agg={"n": "sum"}, obsm_agg={use_rep: "mean"}
        ) for adata in adatas_
    ]
    us = [normalize(leiden.obsm[use_rep], norm="l2") for leiden in leidens]
    ns = [leiden.obs["n"] for leiden in leidens]
    cosines = []
    for i, ui in enumerate(us):
        for j, uj in enumerate(us[i + 1:], start=i + 1):
            cosine = ui @ uj.T
            if (cosine>cutoff).sum() == 0:
                cutoff_ = 0.15
            else:
                cutoff_ = cutoff
            cosine[cosine < cutoff_] = 0
            cosine = COO.from_numpy(cosine)
            cosine = np.power(cosine, power)
            key = tuple(
                slice(None) if k in (i, j) else np.newaxis
                for k in range(len(us))
            )  # To align axes
            cosines.append(cosine[key])
    joint_cosine = prod(cosines)

    for i, (adata, adata_, leiden, n) in enumerate(zip(adatas, adatas_, leidens, ns)):
        balancing = joint_cosine.sum(axis=tuple(
            k for k in range(joint_cosine.ndim) if k != i
        )).todense() / n
        balancing = pd.Series(balancing, index=leiden.obs_names)
        balancing = balancing.loc[adata_.obs["leiden"]].to_numpy()
        balancing /= balancing.sum() / balancing.size
        adata.obs[key_added] = balancing


@logged
def fit_model(
        adatas: Mapping[str, AnnData],
        features: List[str], 
        model: type = MyModel,
        project_name: str = "my_project",
) -> MyModel:
    fit_model.logger.info("Prepare Pre-train...")
    fit_kws = {"directory":project_name}
    pretrain_fit_kws = fit_kws.copy()
    pretrain_fit_kws.update({"align_burnin": np.inf, "safe_burnin": False})
    pretrain = model(adatas, sorted(features))
    pretrain.compile()
    fit_model.logger.info("Pre-train Prism model...")
    pretrain.fit(adatas, **pretrain_fit_kws)
    pretrain.save(os.path.join(pretrain_fit_kws["directory"], "pretrain.dill"))

    fit_model.logger.info("Prepare Fine-tuning...")
    for k, adata in adatas.items():
        adata.obsm[f"X_{config.TMP_PREFIX}"] = pretrain.encode_data(k, adata)
    estimate_balancing_weight(
        *adatas.values(), use_rep=f"X_{config.TMP_PREFIX}",
        key_added="balancing_weight"
    )
    for adata in adatas.values():
        adata.uns[config.ANNDATA_KEY]["use_dsc_weight"] = "balancing_weight"
        del adata.obsm[f"X_{config.TMP_PREFIX}"]
    prod.logger.info("Fine-tuning Prism model...")
    finetune = model(adatas, sorted(features))
    finetune.adopt_pretrained_model(pretrain)
    finetune.compile()
    finetune.fit(adatas, **fit_kws)
    finetune.save(os.path.join(fit_kws["directory"], "fine-tune.dill"))

    return finetune