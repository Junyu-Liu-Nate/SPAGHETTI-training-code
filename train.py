from custom_types import *
import constants
from data_loaders import mesh_datasets
from options import Options
from utils import train_utils, files_utils, rotation_utils, mcubes_meshing
from models.occ_gmm import OccGen
# from models.occ_gmm import Spaghetti
from models import gm_utils, sdf_loss
from warmup_scheduler import GradualWarmupScheduler
import wandb

class Trainer:

    def between_epochs(self, train_dict, epoch):
        if train_dict['loss_occ'] < self.best_scores:
            self.model.save()
            self.best_scores = train_dict['loss_occ']
        if epoch > self.offset and (epoch - self.offset) % self.opt.lr_decay_every == 0:
            self.scheduler.step()

    def split_shape(self, mu: T) -> T:
        mu = mu.view(mu.shape[0], -1, mu.shape[-1])
        b, g, c = mu.shape
        rotation_mat = rotation_utils.get_random_rotation(b).to(self.device)
        mu_z: T = torch.einsum('bad,bgd->bga', rotation_mat, mu)[:, :, -1]
        random_down_top_order = mu_z.argsort(dim=-1)
        # split_index = g // 4 + torch.randint(g // 2, size=(b,), device=self.device).unsqueeze(-1)
        split_index = self.opt.min_split + torch.randint(self.opt.max_split - self.opt.min_split, size=(b,),
                                                         device=self.device).unsqueeze(-1)
        mask = random_down_top_order.gt(split_index)  # True- the gaussians we drop
        return mask

    @staticmethod
    def get_argmax_supports(supports):
        argmax_supports = supports[0].argmax(-1)
        for support in supports[1:]:
            num_splits = support.shape[-1]
            argmax_supports = argmax_supports * num_splits + support.argmax(-1)
        return argmax_supports

    def get_disentanglement_loss(self, gmms: List[TS], zh: T, samples: T, labels: T) -> T:
        with torch.no_grad():
            supports = self.gmm_criterion(gmms, samples, get_supports=True)[1]
            mask = self.split_shape(gmms[-1][0])
            vs_labels = self.get_argmax_supports(supports)
            outside = torch.gather(mask, 1, vs_labels).flatten()
            labels = labels.flatten()
            labels.masked_fill_(outside, 1.)
        gmms, samples = self.transform(gmms, samples)
        out = self.model.forward_b(samples, zh, gmms, mask)
        loss = self.criterion(out, labels)
        return loss

    def relabel(self, gmms, samples, mask, labels):

        def swap_samples(samples_):
            shape = samples_.shape
            samples_swap_ = samples_.view(b // 2, 2, *shape[1:])
            samples_swap_ = torch.stack((samples_swap_[:, 1], samples_swap_[:, 0]), dim=1).view(*shape)
            return samples_swap_

        b, n, d = samples.shape
        mask = mask.unsqueeze(1).repeat(1, 2, 1)
        mask[:, 1] = ~mask[:, 1]
        mask = mask.view(b, -1)
        supports_a = self.gmm_criterion([gmms], samples, get_supports=True)[1][0]
        vs_labels = supports_a.argmax(-1)
        outside = torch.gather(~mask, 1, vs_labels)
        labels.masked_fill_(outside, 1.)
        samples_swapped = swap_samples(samples)
        supports_b = self.gmm_criterion([gmms], samples_swapped, get_supports=True)[1][0]
        vs_labels = supports_b.argmax(-1)
        ignore_label = torch.gather(mask, 1, vs_labels)
        labels = swap_samples(labels)
        labels = labels.view(-1, n * 2)
        ignore_label = ignore_label.view(-1, n * 2)
        return labels, ignore_label, samples_swapped.view(-1, n * 2, d)

    def transform(self, gmms: List[TS], points):
        gmms_fixed, gmm_rot = gmms[:self.opt.num_levels], gmms[-1]
        with torch.no_grad():
            mu, p, phi, eigen = gmm_rot
            p: T = p.transpose(3, 4)
            b = mu.shape[0]
            # rot = self.get_rotation(b).to(mu.device)
            rot = torch.eye(3, device=self.device).unsqueeze(0).expand(b, 3, 3)
            scale = 1 + self.opt.augmentation_scale * (2 * torch.rand(b, 1, device=mu.device) - 1)
            translate = self.opt.augmentation_scale * (2 * torch.rand(b, 3, device=mu.device) - 1)
            rot = rot * scale[:, None, :]
            mu_r = torch.einsum('bad, bpnd->bpna', rot, mu) + translate[:, None, None, :]
            p_r = p * eigen[:, :, :, None, :]
            p_r = torch.einsum('bad, bpndc->bpnac', rot, p_r)
            eigen_r = p_r.norm(2, 3)
            p_r = p_r / eigen_r[:, :, :, None, :]
            points = torch.einsum('bad, bnd->bna', rot, points) + translate[:, None, :]
            p_r = p_r.transpose(3, 4)
            eigen_r = eigen_r * scale[:, :, None, None]
            gmm_rot = mu_r.detach(), p_r.detach(), phi.detach(), eigen_r.detach()
        return gmms_fixed + [gmm_rot], points

    def get_gmm_loss(self, gmms: TS, samples_gmm: T):
        return sum(self.gmm_criterion(gmms, samples_gmm))

    def prepare_data(self, data: TS) -> TS:
        return tuple(map(lambda x: x.to(self.device), data))

    def get_symmetric_loss(self, gmm: TS):
        mu, p, phi, eigen = gmm
        phi = phi.softmax(2)
        gmm = mu, p, phi, eigen
        gmms_r = rotation_utils.apply_gmm_affine(gmm, self.reflect)
        z_gmm_a = gm_utils.flatten_gmm(gmm, self.opt.as_tait_bryan)
        z_gmm_b = gm_utils.flatten_gmm(gmms_r, self.opt.as_tait_bryan)
        return nnf.l1_loss(z_gmm_a, z_gmm_b)

    def train_iter(self, data: TS, is_train: bool):
        samples, labels, items = self.prepare_data(data)
        samples_occ, samples_gmm,  = samples[:, :3].view(samples.shape[0], -1, 3), samples[:, 3],
        self.optimizer.zero_grad()
        zh, z, gmms = self.model.forward_a(items)
        # print(f'After self.model.forward_a(items), gmm[0] shape: {gmms[0]}') #TODO: Debug
        loss_gmm = self.get_gmm_loss(gmms, samples_gmm)
        gmms_b, samples_occ_b = self.transform(gmms, samples_occ)
        out = self.model.forward_b(samples_occ_b, zh, gmms_b)
        loss_reg = sdf_loss.reg_z_loss(z)
        loss_occ = self.criterion(out, labels)
        loss = loss_occ + self.opt.reg_weight * loss_reg + self.opt.gmm_weight * loss_gmm
        self.logger.stash_iter('loss_occ', loss_occ, 'loss_gmm', loss_gmm)
        if self.opt.disentanglement:
            loss_disentanglement = self.get_disentanglement_loss(gmms, zh, samples_occ, labels.view(labels.shape[0], -1))
            loss += self.opt.disentanglement_weight * loss_disentanglement
            self.logger.stash_iter('loss_disentanglement', loss_disentanglement)
        if self.reflect is not None:
            loss_reflect = self.get_symmetric_loss(gmms[0])
            loss += loss_reflect
            self.logger.stash_iter('loss_symm', loss_reflect)
            pass
        loss.backward()
        self.optimizer.step()
        self.warm_up_scheduler.step()

        # Log losses to W&B
        wandb.log({'loss_occ': loss_occ.item(), 
                'loss_gmm': loss_gmm.item(), 
                'total_loss': loss.item(),
                })

    def train_epoch(self, epoch: int, is_train: bool):
        self.model.train(is_train)
        self.logger.start(len(self.dl), tag=self.opt.tag + ' train' if is_train else ' val')
        for data in self.dl:
            self.train_iter(data, is_train)
            self.logger.reset_iter()
        return self.logger.stop()

    def reset_dl(self) -> DataLoader:
        # ds = DummyDs()
        ds = mesh_datasets.CacheDataset(self.opt.dataset_name, self.opt.num_samples, self.opt.data_symmetric)
        self.opt.dataset_size = len(ds)
        if self.opt.subset > 0:
            self.opt.dataset_size = self.opt.subset
            ds = Subset(ds, torch.arange(self.opt.subset))
        # ds = Subset(ds, torch.arange(1000))
        # if DEBUG and len(ds) > 32:
        #     ds = Subset(ds, torch.arange(32))
        dl = DataLoader(ds, batch_size=self.opt.batch_size, pin_memory=True,
                        num_workers=0 if constants.DEBUG else 4, shuffle=not constants.DEBUG, drop_last=True)
        return dl

    def train(self):
        for i in range(self.opt.epochs):
            train_dict = self.train_epoch(i, True)
            self.between_epochs(train_dict, i + 1)

    @property
    def device(self):
        return self.opt.device

    def __init__(self, opt: Options):
        self.opt = opt
        self.dl = self.reset_dl()
        self.offset = opt.warm_up // len(self.dl)
        model: Tuple[OccGen, Options] = train_utils.model_lc(opt)
        # model: Tuple[Spaghetti, Options] = train_utils.model_lc(opt)
        self.model, self.opt = model
        self.optimizer = Optimizer(self.model.parameters(), lr=1e-7)
        self.warm_up_scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1e2, total_epoch=opt.warm_up)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, opt.lr_decay)
        self.logger = train_utils.Logger()
        self.criterion = sdf_loss.occupancy_bce
        self.gmm_criterion = gm_utils.hierarchical_gm_log_likelihood_loss()
        self.fixed_items = torch.randint(low=0, high=self.opt.dataset_size, size=(opt.batch_size // 2,))
        self.best_scores = 100
        self.plot_scale = 1.
        self.get_rotation = rotation_utils.rand_bounded_rotation_matrix(100000, theta_range=self.opt.augmentation_rotation)
        if sum(opt.symmetric) == sum(opt.symmetric_loss) > 0:
            self.reflect = rotation_utils.get_reflection(opt.symmetric).to(self.device)
        else:
            self.reflect = None


if __name__ == '__main__':
    opt_ = Options().load()

    # Initialize W&B
    print(opt_.to_dict())
    wandb.init(project='spaghetti', config=opt_.to_dict())

    Trainer(opt_).train()
    exit(0)
