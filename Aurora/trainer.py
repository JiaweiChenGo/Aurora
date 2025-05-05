import ignite
import torch
import numpy as np
import torch.distributions as D
import torch.nn.functional as F
import ignite.contrib.handlers.tensorboard_logger as tb
from torch.optim.lr_scheduler import ReduceLROnPlateau

import itertools
from itertools import chain
import os
from typing import Any, List, Mapping, Optional,Iterable
import pathlib
import shutil
import parse
import tempfile
from dataset import Dataset
from abc import abstractmethod

from .utils import config, logged

EPOCH_STARTED = ignite.engine.Events.EPOCH_STARTED
EPOCH_COMPLETED = ignite.engine.Events.EPOCH_COMPLETED
ITERATION_COMPLETED = ignite.engine.Events.ITERATION_COMPLETED
EXCEPTION_RAISED = ignite.engine.Events.EXCEPTION_RAISED
COMPLETED = ignite.engine.Events.COMPLETED
TERMINATE = ignite.engine.Events.TERMINATE

class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset: Dataset, **kwargs) -> None:
        super().__init__(dataset, **kwargs)
        self.collate_fn = self._collate
        self.shuffle = kwargs["shuffle"] if "shuffle" in kwargs else False

    def __iter__(self) -> "DataLoader":
        if self.shuffle:
            self.dataset.shuffle()  # Customized shuffling
        return super().__iter__()

    @staticmethod
    def _collate(batch):
        return tuple(map(lambda x: torch.cat(x, dim=0), zip(*batch)))


@logged    
class Tensorboard():
    def attach(
            self, net: torch.nn.Module, trainer,
            train_engine: ignite.engine.Engine,
            val_engine: ignite.engine.Engine,
            train_loader: Iterable,
            val_loader: Optional[Iterable],
            directory: pathlib.Path
    ) -> None:
        tb_directory = directory / "tensorboard"
        if tb_directory.exists():
            shutil.rmtree(tb_directory)

        tb_logger = tb.TensorboardLogger(
            log_dir=tb_directory,
            flush_secs=config.TENSORBOARD_FLUSH_SECS
        )
        tb_logger.attach(
            train_engine,
            log_handler=tb.OutputHandler(
                tag="train", metric_names=trainer.required_losses
            ), event_name=EPOCH_COMPLETED
        )
        if val_engine:
            tb_logger.attach(
                val_engine,
                log_handler=tb.OutputHandler(
                    tag="val", metric_names=trainer.required_losses
                ), event_name=EPOCH_COMPLETED
            )
        train_engine.add_event_handler(COMPLETED, tb_logger.close)

@logged
class EarlyStopping():
    def __init__(
            self,
            monitor: str, patience: int,
            burnin: int = 0, wait_n_lrs: int = 0
    ) -> None:
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.burnin = burnin
        self.wait_n_lrs = wait_n_lrs

    def attach(
            self, net: torch.nn.Module, trainer,
            train_engine: ignite.engine.Engine,
            val_engine: ignite.engine.Engine,
            train_loader: Iterable,
            val_loader: Optional[Iterable],
            directory: pathlib.Path
    ) -> None:
        for item in directory.glob("checkpoint_*.pt"):
            item.unlink()

        score_engine = val_engine if val_engine else train_engine
        score_function = lambda engine: -score_engine.state.metrics[self.monitor]
        event_filter = (
            lambda engine, event: event > self.burnin and engine.state.n_lrs >= self.wait_n_lrs
        ) if self.wait_n_lrs else (
            lambda engine, event: event > self.burnin
        )
        event = EPOCH_COMPLETED(event_filter=event_filter)  # pylint: disable=not-callable
        train_engine.add_event_handler(
            event, ignite.handlers.Checkpoint(
                {"net": net, "trainer": trainer},
                ignite.handlers.DiskSaver(
                    directory, atomic=True, create_dir=True, require_empty=False
                ), score_function=score_function,
                filename_pattern="checkpoint_{global_step}.pt",
                n_saved=config.CHECKPOINT_SAVE_NUMBERS,
                global_step_transform=ignite.handlers.global_step_from_engine(train_engine)
            )
        )
        train_engine.add_event_handler(
            event, ignite.handlers.EarlyStopping(
                patience=self.patience,
                score_function=score_function,
                trainer=train_engine
            )
        )

        @train_engine.on(COMPLETED | TERMINATE)
        def _(engine):
            nan_flag = any(
                not bool(torch.isfinite(item).all())
                for item in (engine.state.output or {}).values()
            )
            ckpts = sorted([
                parse.parse("checkpoint_{epoch:d}.pt", item.name).named["epoch"]
                for item in directory.glob("checkpoint_*.pt")
            ], reverse=True)
            if ckpts and nan_flag and train_engine.state.epoch == ckpts[0]:
                self.logger.warning(
                    "The most recent checkpoint \"%d\" can be corrupted by NaNs, "
                    "will thus be discarded.", ckpts[0]
                )
                ckpts = ckpts[1:]
            if ckpts:
                self.logger.info("Restoring checkpoint \"%d\"...", ckpts[0])
                loaded = torch.load(directory / f"checkpoint_{ckpts[0]}.pt")
                net.load_state_dict(loaded["net"])
                trainer.load_state_dict(loaded["trainer"])
            else:
                self.logger.info(
                    "No usable checkpoint found. "
                    "Skipping checkpoint restoration."
                )

class LRScheduler():
    def __init__(
            self, *optims: torch.optim.Optimizer, monitor: str = None,
            patience: int = None, burnin: int = 0
    ) -> None:
        super().__init__()
        self.monitor = monitor
        self.schedulers = [
            ReduceLROnPlateau(optim, patience=patience, verbose=True)
            for optim in optims
        ]
        self.burnin = burnin

    def attach(
            self, net: torch.nn.Module, trainer,
            train_engine: ignite.engine.Engine,
            val_engine: ignite.engine.Engine,
            train_loader: Iterable,
            val_loader: Optional[Iterable],
            directory: pathlib.Path
    ) -> None:
        score_engine = val_engine if val_engine else train_engine
        event_filter = lambda engine, event: event > self.burnin
        for scheduler in self.schedulers:
            scheduler.last_epoch = self.burnin
        train_engine.state.n_lrs = 0

        @train_engine.on(EPOCH_COMPLETED(event_filter=event_filter))  # pylint: disable=not-callable
        def _():
            update_flags = set()
            for scheduler in self.schedulers:
                old_lr = scheduler.optimizer.param_groups[0]["lr"]
                scheduler.step(score_engine.state.metrics[self.monitor])
                new_lr = scheduler.optimizer.param_groups[0]["lr"]
                update_flags.add(new_lr != old_lr)
            if len(update_flags) != 1:
                raise RuntimeError("Learning rates are out of sync!")
            if update_flags.pop():
                train_engine.state.n_lrs += 1
                self.logger.info("Learning rate reduction: step %d", train_engine.state.n_lrs)


@logged
class Trainer:
    BURNIN_NOISE_EXAG: float = 1.5
    
    def __init__(self, net,
                 lam_data: float = None, 
                 lam_kl: float = None,
                 lam_feature: float = None,
                 lam_align: float = None,
                 lam_joint_cross: float = None, 
                 lam_real_cross: float = None,
                 lam_cos: float = None, 
                 optim: str = None,
                 lr: float = None, 
                 **kwargs) -> None:
        self.net = net
        self.required_losses = ["f_kl","dsc_loss", "vae_loss", "gen_loss"]
        for k in self.net.keys:
            self.required_losses += [f"x_{k}_nll", f"x_{k}_kl", f"x_{k}_elbo"]
        self.earlystop_loss = "vae_loss"

        self.lam_data = lam_data
        self.lam_kl = lam_kl
        self.lam_feature = lam_feature
        self.lam_align = lam_align
        self.lr = lr
        self.vae_optim = getattr(torch.optim, optim)(
            itertools.chain(
                self.net.f2u.parameters(),
                self.net.x2z.parameters(),
                self.net.z2x.parameters()
            ), lr=self.lr, **kwargs
        )
        self.dsc_optim = getattr(torch.optim, optim)(
            self.net.du.parameters(), lr=self.lr, **kwargs
        )

        self.align_burnin: Optional[int] = None
        self.freeze_u = False
        self.lam_joint_cross = lam_joint_cross
        self.lam_real_cross = lam_real_cross
        self.lam_cos = lam_cos
        self.required_losses += ["joint_cross_loss", "real_cross_loss", "cos_loss"]

    @property
    def freeze_u(self) -> bool:
        return self._freeze_u

    @freeze_u.setter
    def freeze_u(self, freeze_u: bool) -> None:
        self._freeze_u = freeze_u
        for item in chain(self.net.x2z.parameters(), self.net.du.parameters()):
            item.requires_grad_(not self._freeze_u)

    def compute_losses(
            self, data, 
            epoch: int, 
            dsc_only: bool = False
    ) -> Mapping[str, torch.Tensor]:  # pragma: no cover
        net = self.net
        x, xalt, xlbl, xflag, pmsk = data

        z, l = {}, {}
        for k in net.keys:
            #print(x[k])
            #print(xalt[k])
            z[k] = net.x2z[k](x[k], xalt[k])
        zsamp = {k: z[k].rsample() for k in net.keys}
        prior = net.prior()

        z_cat = torch.cat([z[k].mean for k in net.keys])
        xflag_cat = torch.cat([xflag[k] for k in net.keys])

        anneal = max(1 - (epoch - 1) / self.align_burnin, 0) \
            if self.align_burnin else 0
        if anneal:
            noise = D.Normal(0, z_cat.std(axis=0)).sample((z_cat.shape[0], ))
            z_cat = z_cat + (anneal * self.BURNIN_NOISE_EXAG) * noise
        dsc_loss = F.cross_entropy(net.du(z_cat), xflag_cat, reduction="none")
        dsc_loss = dsc_loss.mean()
        if dsc_only:
            return {"dsc_loss": self.lam_align * dsc_loss}
        
        u,_ = net.f2u()
        usamp = u.rsample()

        f_kl = D.kl_divergence(u, prior).sum(dim=1).mean() / usamp.shape[0]

        x_nll = {
            k: -net.z2x[k](
                zsamp[k], usamp[getattr(net, f"{k}_idx")]
            ).log_prob(x[k]).mean()
            for k in net.keys
        }
        x_kl = {
            k: D.kl_divergence(
                D.Normal(z[k].mean[:,:-1], z[k].stddev[:,:-1]), prior
            ).sum(dim=1).mean() / x[k].shape[1]
            for k in net.keys
        }
        x_kl_ = {
            k: D.kl_divergence(
                D.Normal(z[k].mean[:,-1], z[k].stddev[:,-1]), D.Normal(xlbl[k]/10., torch.ones_like(xlbl[k]) * 0.2)
            ).mean() / x[k].shape[1]
            for k in net.keys
        }
        x_elbo = {
            k: x_nll[k] + self.lam_kl * (0.85 * x_kl[k] + 0.15 * x_kl_[k])
            for k in net.keys
        }
        x_elbo_sum = sum(x_elbo[k] for k in net.keys)

        pmsk = pmsk.T
        zsamp_stack = torch.stack([zsamp[k] for k in net.keys])
        pmsk_stack = pmsk.unsqueeze(2).expand_as(zsamp_stack)
        zsamp_mean = (zsamp_stack * pmsk_stack).sum(dim=0) / pmsk_stack.sum(dim=0)

        if self.lam_joint_cross:
            x_joint_cross_nll = {
                k: -net.z2x[k](
                    zsamp_mean[m], usamp[getattr(net, f"{k}_idx")]
                ).log_prob(x[k][m]).mean()
                for k, m in zip(net.keys, pmsk)
            }
            joint_cross_loss = sum(x_joint_cross_nll[k] for k in net.keys)
        else:
            joint_cross_loss = torch.as_tensor(0.0, device=net.device)

        if self.lam_real_cross:
            x_real_cross_nll = {}
            for k in net.keys:
                xk_real_cross_nll = 0
                for k_target, m in zip(net.keys, pmsk):
                    if k != k_target:
                        xk_real_cross_nll += -net.z2x[k_target](
                            zsamp[k][m], usamp[getattr(net, f"{k_target}_idx")]
                        ).log_prob(x[k_target][m]).mean()
                x_real_cross_nll[k] = xk_real_cross_nll
            real_cross_loss = sum(x_real_cross_nll[k] for k in net.keys)
        else:
            real_cross_loss = torch.as_tensor(0.0, device=net.device)

        if self.lam_cos:
            cos_loss = sum(
                1 - F.cosine_similarity(
                    zsamp_stack[i, m], zsamp_mean[m]
                ).mean()
                for i, m in enumerate(pmsk)
            )
        else:
            cos_loss = torch.as_tensor(0.0, device=net.device)

        vae_loss = self.lam_data * x_elbo_sum \
            + self.lam_feature * len(net.keys) * f_kl \
            + self.lam_joint_cross * joint_cross_loss \
            + self.lam_real_cross * real_cross_loss \
            + self.lam_cos * cos_loss
        gen_loss = vae_loss - self.lam_align * dsc_loss

        losses = {
            "dsc_loss": dsc_loss, 
            "vae_loss": vae_loss, 
            "gen_loss": gen_loss,
            "f_kl": f_kl,
            "joint_cross_loss": joint_cross_loss,
            "real_cross_loss": real_cross_loss,
            "cos_loss": cos_loss
        }
        for k in net.keys:
            losses.update({
                f"x_{k}_nll": x_nll[k],
                f"x_{k}_kl": x_kl[k],
                f"x_{k}_elbo": x_elbo[k]
            })
        return losses

    def format_data(self, data: List[torch.Tensor]):  # pragma: no cover
        device = self.net.device
        keys = self.net.keys
        K = len(keys)
        x, xalt, xlbl, pmsk = data[0:K], data[K:2*K], data[2*K:3*K], data[3*K]
        x = {
            k: x[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xalt = {
            k: xalt[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xlbl = {
            k: xlbl[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xflag = {
            k: torch.as_tensor(
                i, dtype=torch.int64, device=device
            ).expand(x[k].shape[0])
            for i, k in enumerate(keys)
        }
        pmsk = pmsk.to(device, non_blocking=True)
        return x, xalt, xlbl, xflag, pmsk
    
    @abstractmethod
    def train_step(
            self, 
            engine: ignite.engine.Engine, 
            data: List[torch.Tensor]
    ) -> Mapping[str, torch.Tensor]:
        self.net.train()
        data = self.format_data(data)
        epoch = engine.state.epoch

        if self.freeze_u:
            self.net.x2z.eval()
            self.net.du.eval()
        else:  
            # Discriminator step
            losses = self.compute_losses(data, epoch, dsc_only=True)
            self.net.zero_grad(set_to_none=True)
            losses["dsc_loss"].backward()  # Already scaled by lam_align
            self.dsc_optim.step()

        # Generator step
        losses = self.compute_losses(data, epoch)
        self.net.zero_grad(set_to_none=True)
        losses["gen_loss"].backward()
        self.vae_optim.step()

        return losses

    @abstractmethod
    def val_step(
            self, 
            engine: ignite.engine.Engine, 
            data: List[torch.Tensor]
    ) -> Mapping[str, torch.Tensor]:
        self.net.eval()
        data = self.format_data(data)
        return self.compute_losses(data, engine.state.epoch)

    def report_metrics(
            self, train_state: ignite.engine.State,
            val_state: Optional[ignite.engine.State]
    ) -> None:
        
        if train_state.epoch % 10:
            return
        train_metrics = {
            key: float(f"{val:.3f}")
            for key, val in train_state.metrics.items()
        }
        val_metrics = {
            key: float(f"{val:.3f}")
            for key, val in val_state.metrics.items()
        } if val_state else None
        self.logger.info(
            "[Epoch %d] train=%s, val=%s, %.1fs elapsed",
            train_state.epoch, train_metrics, val_metrics,
            train_state.times["EPOCH_COMPLETED"]  # Also includes validator time
        )

    def fit(self, 
            data,
            val_split: float = None,
            data_batch_size: int = None, 
            align_burnin: int = None, 
            safe_burnin: bool = True,
            max_epochs: int = None, 
            patience: Optional[int] = None,
            reduce_lr_patience: Optional[int] = None,
            wait_n_lrs: Optional[int] = None,
            random_seed: int = None, 
            directory: Optional[os.PathLike] = None,
    ) -> None:
        data.getitem_size = max(1, round(data_batch_size / 4))
        data_train, data_val = data.random_split([1 - val_split, val_split], random_state=random_seed)
        data_train.prepare_shuffle(num_workers=0, random_seed=random_seed)
        data_val.prepare_shuffle(num_workers=0, random_seed=random_seed)

        train_loader = DataLoader(
                data_train, batch_size=config.DATALOADER_FETCHES_PER_BATCH, shuffle=True,
                num_workers=config.DATALOADER_NUM_WORKERS,
                pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY,
                drop_last=len(data_train) > config.DATALOADER_FETCHES_PER_BATCH,
                generator=torch.Generator().manual_seed(random_seed),
                persistent_workers=False
            )
        val_loader = DataLoader(
                data_val, batch_size=config.DATALOADER_FETCHES_PER_BATCH, shuffle=True,
                num_workers=config.DATALOADER_NUM_WORKERS,
                pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY, drop_last=False,
                generator=torch.Generator().manual_seed(random_seed),
                persistent_workers=False
            )

        self.align_burnin = align_burnin

        default_plugins = [Tensorboard()]
        if reduce_lr_patience:
            default_plugins.append(LRScheduler(
                self.vae_optim, self.dsc_optim,
                monitor=self.earlystop_loss, patience=reduce_lr_patience,
                burnin=self.align_burnin if safe_burnin else 0
            ))
        if patience:
            default_plugins.append(EarlyStopping(
                monitor=self.earlystop_loss, patience=patience,
                burnin=self.align_burnin if safe_burnin else 0,
                wait_n_lrs=wait_n_lrs or 0
            ))
    
        plugins = default_plugins
        
        #interrupt_delayer = DelayedKeyboardInterrupt()
        directory = pathlib.Path(directory or tempfile.mkdtemp(prefix=config.TMP_PREFIX))
        self.logger.info("Using training directory: \"%s\"", directory)

        # Construct engines
        train_engine = ignite.engine.Engine(self.train_step)
        val_engine = ignite.engine.Engine(self.val_step) if val_loader else None
        # Exception handling
        train_engine.add_event_handler(ITERATION_COMPLETED, ignite.handlers.TerminateOnNan())

        # Compute metrics
        for item in self.required_losses:
            ignite.metrics.Average(
                output_transform=lambda output, item=item: output[item]
            ).attach(train_engine, item)
            if val_engine:
                ignite.metrics.Average(
                    output_transform=lambda output, item=item: output[item]
                ).attach(val_engine, item)

        if val_engine:
            @train_engine.on(EPOCH_COMPLETED)
            def _validate(engine):
                val_engine.run(
                    val_loader, max_epochs=engine.state.epoch
                ) 

        @train_engine.on(EPOCH_COMPLETED)
        def _report_metrics(engine):
            self.report_metrics(engine.state, val_engine.state if val_engine else None)

        for plugin in plugins or []:
            plugin.attach(
                net=self.net, trainer=self,
                train_engine=train_engine, val_engine=val_engine,
                train_loader=train_loader, val_loader=val_loader,
                directory=directory
            )

        # Start engines
        torch.manual_seed(random_seed)
        train_engine.run(train_loader, max_epochs=max_epochs)

        torch.cuda.empty_cache()  # Works even if GPU is unavailable
        data.clean()
        data_train.clean()
        data_val.clean()
        self.align_burnin = None
        self.enorm = None
        self.esgn = None
    
    def state_dict(self) -> Mapping[str, Any]:
        return {
            "vae_optim": self.vae_optim.state_dict(),
            "dsc_optim": self.dsc_optim.state_dict()
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.vae_optim.load_state_dict(state_dict.pop("vae_optim"))
        self.dsc_optim.load_state_dict(state_dict.pop("dsc_optim"))
