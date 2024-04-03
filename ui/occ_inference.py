import sys
sys.path.append("/home/amirh/projects/sdf_gmm")
import utils.rotation_utils
from custom_types import *
import constants
from data_loaders import mesh_datasets
from options import Options
from utils import train_utils, mcubes_meshing, files_utils, mesh_utils
from models.occ_gmm import OccGen
from models import models_utils

import os


class Inference:

    def split_shape(self, mu: T) -> T:
        b, g, c = mu.shape
        # rotation_mat = mesh_utils.get_random_rotation(b).to(self.device)
        # mu_z: T = torch.einsum('bad,bgd->bga', rotation_mat, mu)[:, :, -1]
        mask = []
        for i in range(b):
            axis = torch.randint(low=0, high=c, size=(1,)).item()
            random_down_top_order = mu[i, :, axis].argsort(dim=-1, descending = torch.randint(low=0, high=2, size=(1,)).item() == 0)
            split_index = g // 4 + torch.randint(g // 2, size=(1,), device=self.device)
            mask.append(random_down_top_order.lt(split_index))  # True- the gaussians we drop
        return torch.stack(mask, dim=0)

    def mix_z(self, gmms, zh):
        with torch.no_grad():
            mu = gmms[0].squeeze(1)[:self.opt.batch_size // 2, :]
            mask = self.split_shape(mu).float().unsqueeze(-1)
            zh_fake = zh[:self.opt.batch_size // 2] * mask + (1 - mask) * zh[self.opt.batch_size // 2:]
        return zh_fake

    def get_occ_fun(self, z: T, gmm: Optional[TS] = None):

        def forward(x: T) -> T:
            nonlocal z
            x = x.unsqueeze(0)
            out = self.model.occ_head(x, z, gmm)[0, :]
            if self.opt.loss_func == LossType.CROSS:
                out = out.softmax(-1)
                out = -1 * out[:, 0] + out[:, 2]
                # out = out.argmax(-1).float() - 1
            elif self.opt.loss_func == LossType.IN_OUT:
                out = 2 * out.sigmoid_() - 1
            else:
                out.clamp_(-.2, .2)
            # torch.nan_to_num_(out, nan=1)
            return out
        if z.dim() == 2:
            z = z.unsqueeze(0)
        return forward

    def get_mesh(self, z: T, res: int, gmm: TS, get_time=False) -> Optional[T_Mesh]:

        with torch.no_grad():
            # samples = torch.rand(50000, 3, device=self.device)
            # out = forward(samples)
            # mask = out.lt(0)
            # mesh = samples[mask]
            if get_time:
                time_a = self.meshing.occ_meshing(self.get_occ_fun(z, gmm), res=res, get_time=get_time)

                time_b = sdf_mesh.create_mesh_old(self.get_occ_fun(z, gmm), device=self.opt.device, scale=self.plot_scale, verbose=False,
                                                  res=res, get_time=get_time)

                return time_a, time_b
            else:
                mesh = self.meshing.occ_meshing(self.get_occ_fun(z, gmm), res=res)
                return mesh

    def plot_occ(self, z: Union[T, TS], gmms: Optional[List[TS]], prefix: str, res=200, verbose=False,
                  use_item_id: bool = False, fixed_items: Optional[Union[T, List[str]]] = None):
        gmms = gmms[-1]
        if type(fixed_items) is T:
            fixed_items = [f'{fixed_items[item].item():02d}' for item in fixed_items]
        for i in range(len(z)):
            gmm_ = [gmms[j][i].unsqueeze(0) for j in range(len(gmms))]
            mesh = self.get_mesh(z[i], res, [gmm_])
            if use_item_id and fixed_items is not None:
                name = f'_{fixed_items[i]}'
            elif len(z) == 1:
                name = ''
            else:
                name = f'_{i:02d}'
            if mesh is not None:
                # probs = gm_utils.gm_loglikelihood_loss((gmms,), mesh[0].unsqueeze(0), raw=True)[0][0]
                # colors = palette[probs.argmax(0)]
                files_utils.export_mesh(mesh, f'{self.opt.cp_folder}/occ_{prefix}/{prefix}{name}')
                # files_utils.save_np(mesh[1], f'{self.opt.cp_folder}/vox/{prefix}{name}')
                if gmms is not None:
                    files_utils.export_gmm(gmms, i, f'{self.opt.cp_folder}/gmms_{prefix}/{prefix}{name}')
            if verbose:
                print(f'done {i + 1:d}/{len(z):d}')

    def disentanglement_plot(self, item_a: int, item_b: int, z_in_a, z_in_b, a_inclusive: bool = True,
                             b_inclusive: bool = True):

        def merge_z(z_):
            nonlocal z_in_a, z_in_b, a_inclusive, b_inclusive
            masks = []
            for inds, inclusive in zip((z_in_a, z_in_b), (a_inclusive, b_inclusive)):
                mask_ = torch.zeros(z_.shape[1], dtype=torch.bool)
                mask_[torch.tensor(inds, dtype=torch.long)] = True
                if not inclusive:
                    mask_ = ~mask_
                masks.append(mask_.to(self.device))
            z_a = z_[0][masks[0]]
            z_b = z_[0][~masks[0]]
            if item_b >= 0:
                z_a = torch.cat((z_a, z_[1][masks[1]]), dim=0)
                z_b = torch.cat((z_b, z_[1][~masks[1]]), dim=0)
            return z_a, z_b

        bottom = False
        if item_a < 0 and item_b < 0:
            return
        elif item_a < 0:
            item_a, item_b, z_in_a, z_in_b = item_b, item_a, z_in_b, z_in_a
        suffix = '' if item_b < 0 else f'_{item_b}'
        z_in_a, z_in_b = list(set(z_in_a)), list(set(z_in_b))

        with torch.no_grad():
            if item_b < 0:
                items = torch.tensor([item_a], dtype=torch.int64, device=self.device)
            else:
                items = torch.tensor([item_a, item_b], dtype=torch.int64, device=self.device)
            z_items, z_init, zh, gmms = self.model.get_disentanglement(items)
            if bottom:
                z_in = [z_.unsqueeze(0) for z_ in merge_z(z_init)]
                z_in = [self.model.sdformer.forward_split(self.model.sdformer.forward_upper(z_))[0][0] for z_ in z_in]
            else:
                z_in = merge_z(zh)
            # self.plot_sdfs(zh, gmms, f'trial_a', verbose=True)
            # z_target = replace_half(zh, gmms)
            # z_in = torch.stack((zh[0], z_target), dim=0)

            # z, gmms = self.model_a.interpolate_higher(z_in, num_between=self.fixed_items.shape[0])

            self.plot_sdfs(z_in, None, f'dist_{item_a}{suffix}', verbose=True)

    def compose(self, items: List[int], parts: List[List[int]], inclusive: List[bool]):

        def merge_z() -> T:
            nonlocal inclusive, z_parts, zh
            z_ = []
            for i, (inds, inclusive) in enumerate(zip(z_parts, inclusive)):
                mask_ = torch.zeros(zh.shape[1], dtype=torch.bool)
                mask_[torch.tensor(inds, dtype=torch.long)] = True
                if not inclusive:
                    mask_ = ~mask_
                z_.append(zh[i][mask_])
            z_ = torch.cat(z_, dim=0).unsqueeze_(0)
            return z_
        name = '_'.join([str(item) for item in items])
        z_parts = [list(set(part)) for part in parts]

        with torch.no_grad():
            items = torch.tensor(items, dtype=torch.int64, device=self.device)
            z_items, z_init, zh, gmms = self.model.get_disentanglement(items)
            z_in = merge_z()
            self.plot_sdfs(z_in, None, f'compose_{name}', verbose=True, res=256)

    def load_file(self, info_path, disclude: Optional[List[int]] = None):
        info = files_utils.load_pickle(''.join(info_path))
        keys = list(info['ids'].keys())
        items = map(lambda x: int(x.split('_')[1]) if type(x) is str else x, keys)
        items = torch.tensor(list(items), dtype=torch.int64, device=self.device)
        zh, _, gmms_sanity, _ = self.model.get_embeddings(items)
        gmms = [item for item in info['gmm']]
        zh_ = []
        split = []
        gmm_mask = torch.ones(gmms[0].shape[2], dtype=torch.bool)
        counter = 0
        # gmms_ = [[] for _ in range(len(gmms))]
        for i, key in enumerate(keys):
            gaussian_inds = info['ids'][key]
            if disclude is not None:
                for j in range(len(gaussian_inds)):
                    gmm_mask[j + counter] = gaussian_inds[j] not in disclude
                counter += len(gaussian_inds)
                gaussian_inds = [ind for ind in gaussian_inds if ind not in disclude]
                info['ids'][key] = gaussian_inds
            gaussian_inds = torch.tensor(gaussian_inds, dtype=torch.int64)
            zh_.append(zh[i, gaussian_inds])
            split.append(len(split) + torch.ones(len(info['ids'][key]), dtype=torch.int64, device=self.device))
        zh_ = torch.cat(zh_, dim=0).unsqueeze(0).to(self.device)
        gmms = [item[:, :, gmm_mask].to(self.device) for item in info['gmm']]
        return zh_, gmms, split, info['ids']

    @models_utils.torch_no_grad
    def get_z_from_file(self, info_path):
        zh_, gmms, split, _ = self.load_file(info_path)
        zh_ = self.model.merge_zh_step_a(zh_, [gmms])
        zh, _ = self.model.affine_transformer.forward_with_attention(zh_)
        # gmms_ = [torch.cat(item, dim=1).unsqueeze(0) for item in gmms_]
        # zh, _ = self.model.merge_zh(zh_, [gmms])
        return zh, zh_, gmms, torch.cat(split)

    def plot_from_info(self, info_path, res):
        zh, zh_, gmms, split = self.get_z_from_file(info_path)
        mesh = self.get_mesh(zh[0], res, gmms)
        if mesh is not None:
            attention = self.get_attention_faces(mesh, zh, fixed_z=split)
        else:
            attention = None
        return mesh, attention

    def get_mesh_interpolation(self, z: T, res: int, mask:TN, alpha: T) -> Optional[T_Mesh]:

        def forward(x: T) -> T:
            nonlocal z, alpha
            x = x.unsqueeze(0)
            out = self.model.occ_head(x, z, mask=mask, alpha=alpha)[0, :]
            out = 2 * out.sigmoid_() - 1
            return out

        mesh = self.meshing.occ_meshing(forward, res=res)
        return mesh

    @staticmethod
    def combine_and_pad(zh_a: T, zh_b: T) -> Tuple[T, TN]:
        if zh_a.shape[1] == zh_b.shape[1]:
            mask = None
        else:
            pad_length = max(zh_a.shape[1], zh_b.shape[1])
            mask = torch.zeros(2, pad_length, device=zh_a.device, dtype=torch.bool)
            padding = torch.zeros(1, abs(zh_a.shape[1] - zh_b.shape[1]), zh_a.shape[-1], device=zh_a.device)
            if zh_a.shape[1] > zh_b.shape[1]:
                mask[1, zh_b.shape[1]:] = True
                zh_b = torch.cat((zh_b, padding), dim=1)
            else:
                mask[0, zh_a.shape[1]: ] = True
                zh_a = torch.cat((zh_a, padding), dim=1)
        return torch.cat((zh_a, zh_b), dim=0), mask

    @staticmethod
    def get_intersection_z(z_a: T, z_b: T) -> T:
        diff = (z_a[0, :, None, :] - z_b[0, None]).abs().sum(-1)
        diff_a = diff.min(1)[0].lt(.1)
        diff_b = diff.min(0)[0].lt(.1)
        if diff_a.shape[0] != diff_b.shape[0]:
            padding = torch.zeros(abs(diff_a.shape[0] - diff_b.shape[0]), device=z_a.device, dtype=torch.bool)
            if diff_a.shape[0] > diff_b.shape[0]:
                diff_b = torch.cat((diff_b, padding))
            else:
                diff_a = torch.cat((diff_a, padding))
        return torch.cat((diff_a, diff_b))

    def get_attention_points(self, vs: T, zh: T, mask: TN = None, alpha: TN = None):
        vs = vs.unsqueeze(0)
        attention = self.model.occ_head.forward_attention(vs, zh, mask=mask, alpha=alpha)
        attention = torch.stack(attention, 0).mean(0).mean(-1)
        attention = attention.permute(1, 0, 2).reshape(attention.shape[1], -1)
        attention_max = attention.argmax(-1)
        return attention_max

    @models_utils.torch_no_grad
    def get_attention_faces(self, mesh: T_Mesh, zh: T, mask: TN = None, fixed_z: TN = None, alpha: TN = None):
        coords = mesh[0][mesh[1]].mean(1).to(zh.device)
        # coords = mesh[0].unsqueeze(0).to(zh.device)
        attention_max = self.get_attention_points(coords, zh, mask, alpha)
        if fixed_z is not None:
            attention_select = fixed_z[attention_max].cpu()
        else:
            attention_select = attention_max
        # colors = torch.zeros(attention_select.shape[0], 3)
        # colors[~attention_select] = torch.tensor((1., 0, 0))
        return attention_select

    @models_utils.torch_no_grad
    def interpolate_from_files(self, item_a: Union[str, int], item_b: Union[str, int], num_mid: int, res: int,
                               counter: int = 0, logger: Optional[train_utils.Logger] = None):
        zh_a, zh_a_raw, _, _ = self.get_z_from_file(item_a)
        zh_b, zh_b_raw, _, _ = self.get_z_from_file(item_b)
        folder = files_utils.split_path(item_a)[0].split('/')[-1]
        fixed_z = self.get_intersection_z(zh_a_raw, zh_b_raw)
        zh, mask = self.combine_and_pad(zh_a, zh_b)
        if logger is None:
            logger = train_utils.Logger().start(num_mid)
        for i, alpha_ in enumerate(torch.linspace(0, 1, num_mid)):
            alpha = torch.tensor([1., 1.], device=self.device)
            alpha[0] = 1 - alpha_
            alpha[1] = alpha_
            out_path = f"{self.opt.cp_folder}/{folder}/ff_{counter + i:03d}.obj"
            if not files_utils.is_file(out_path):
                mesh = self.get_mesh_interpolation(zh, res, mask, alpha)
                colors = self.get_attention_faces(mesh, zh, mask, fixed_z, alpha)
                colors = (~colors).long()
                files_utils.export_mesh(mesh, f"{self.opt.cp_folder}/{folder}/ff_{counter + i:03d}")
                files_utils.export_list(colors.tolist(), f"{self.opt.cp_folder}/{folder}/ff_{counter + i:03d}_faces")
            logger.reset_iter()

    @models_utils.torch_no_grad
    def plot_folder(self, *folders, res: int = 256):
        logger = train_utils.Logger()
        for folder in folders:
            paths = files_utils.collect(folder, '.pkl')
            logger.start(len(paths))
            for path in paths:
                name = path[1]
                out_path = f"{self.opt.cp_folder}/from_ui/{name}"
                mesh, colors = self.plot_from_info(path, res)
                if mesh is not None:
                    files_utils.export_mesh(mesh, out_path)
                    files_utils.export_list(colors.tolist(), f"{out_path}_faces")
                logger.reset_iter()
            logger.stop()

    def get_samples_by_names(self, names: List[str]) -> T:
        ds = mesh_datasets.CacheDataset(self.opt.dataset_name, self.opt.num_samples, self.opt.data_symmetric)
        return torch.tensor([ds.get_item_by_name(name) for name in names], dtype=torch.int64)

    def get_names_by_samples(self, items):
        ds = mesh_datasets.CacheDataset(self.opt.dataset_name, self.opt.num_samples, self.opt.data_symmetric)
        return [ds.get_name(item) for item in items]

    def get_zh_from_idx(self, items: T):
        zh, _, gmms, __ = self.model.get_embeddings(items.to(self.device))
        zh, attn_b = self.model.merge_zh(zh, gmms)
        return zh, gmms

    @models_utils.torch_no_grad
    def plot(self, prefix: Union[str, int], verbose=False, interpolate: bool = False, res: int = 200, size: int = -1,
             names: Optional[List[str]] = None):
        if size <= 0:
            size = self.opt.batch_size // 2
        if names is not None:
            fixed_items = self.get_samples_by_names(names)
        elif self.model.opt.dataset_size < size:
            fixed_items = torch.arange(self.model.opt.dataset_size)
        else:
            if self.model.stash is None:
                # numbers = files_utils.collect(f"{constants.CHECKPOINTS_ROOT}/occ_gmm_tables_full/occ/", '.obj')
                # numbers = list(map(lambda x: int(x[1].split('_')[1]), numbers))
                fixed_items = torch.randint(low=0, high=self.opt.dataset_size, size=(size,))
                # fixed_items = torch.tensor(numbers)
            else:
                fixed_items = torch.randint(low=0, high=self.model.stash.shape[0], size=(size,))
        # if names is None and self.model.stash is None:
        #     names = self.get_names_by_samples(fixed_items)
        # else:
        #     names = ["random"] * len(fixed_items)
        # names = [f'{i:02d}_{name}' for i, name in zip(fixed_items, names)]
        fixed_items = torch.tensor([0, 100, 200, 300, 400, 500, 600, 700, 800, 900], dtype=torch.int64)
        # fixed_items = torch.arange(size) + 100
        # fixed_items = fixed_items.unique()
        # fixed_items = torch.tensor(fixed_items).long().to(self.device)
        if type(prefix) is int:
            prefix = f'{prefix:04d}'
        if interpolate:
            z, gmms = self.model.interpolate(fixed_items.to(self.device), num_between=8)
            use_item_id = False
        else:
            zh, _, gmms, attn_a = self.model.get_embeddings(fixed_items.to(self.device))
            zh, attn_b = self.model.merge_zh(zh, gmms)
            func = self.get_occ_fun(zh[0])
            x = torch.linspace(-1, 1, 27).to(self.device, dtype=torch.float32).view(-1, 3)
            out = func(x)
            # files_utils.save_pickle(out.detach().cpu(), f"/home/amirh/projects/spaghetti_private/assets/debug/out_188_gt")
            files_utils.save_pickle(out.detach().cpu(), f"./assets/debug/out_188_gt")
            use_item_id = True
            # return
        self.plot_occ(zh, gmms, prefix, verbose=verbose, use_item_id=use_item_id,
                       res=res, fixed_items=names)

    # @models_utils.torch_no_grad
    # def plot(self, folder_name: str, nums_sample: int, verbose=False, res: int = 200, tf_sample_dirname='', attn_weights_path=''):
    #     '''
    #     Saves reconstructions of training meshes
    #     '''
    #     attn_weights = None
    #     if attn_weights_path:
    #         assert not tf_sample_dirname # want standard quantized reconstruction process
    #         shape_samples = os.path.basename(attn_weights_path)[:-3].split("_")[-1]
    #         print(shape_samples)
    #         shape_samples = shape_samples.split("-")
    #         shape_samples = torch.tensor([int(x) for x in shape_samples]) # ints
    #         print("attn weight context shapes: ", shape_samples)
    #         attn_weights = torch.load(attn_weights_path) # [total_n_parts,]
    #         # linearly rescale attention weights to [0,1] range
    #         attn_weights = (attn_weights - torch.min(attn_weights))/(torch.max(attn_weights) - torch.min(attn_weights))
    #         attn_weights = attn_weights.view(shape_samples.shape[0], -1) # [n_priming_shapes, n_parts]
    #         print("scaled attn: ", attn_weights)
    #     elif self.model.opt.dataset_size < nums_sample:
    #         print("using ALL train data")
    #         shape_samples = torch.arange(self.model.opt.dataset_size)
    #     else:
    #         print('using rand train subset (if no TF samples specified)')
    #         shape_samples = torch.randint(low=0, high=self.opt.dataset_size, size=(nums_sample,))
    #     if tf_sample_dirname:
    #         print("using TF samples from ", tf_sample_dirname)
    #     zh_base, _, gmms = self.model.get_embeddings(shape_samples.to(self.device), folder_name, tf_sample_dirname) # NOTE: quantized code overrides internally
    #     zh, attn_b = self.model.merge_zh(zh_base, gmms)
    #     print("zh: ", zh.shape)

    #     # save mesh names
    #     from_quantized = False
    #     if not attn_weights_path: 
    #         if not os.path.exists(f'assets/checkpoints/spaghetti_airplanes/{folder_name}/codes/mesh_ids.npy'):
    #             np.save(f'assets/checkpoints/spaghetti_airplanes/{folder_name}/codes/mesh_ids.npy', shape_samples.detach().cpu().numpy())
    #         else:
    #             print('using existing mesh names')
    #             from_quantized = True
    #             shape_samples = np.load(f'assets/checkpoints/spaghetti_airplanes/{folder_name}/codes/mesh_ids.npy')
    #     # only reconstruct meshes for first n samples
    #     shape_samples = shape_samples[:nums_sample]
    #     self.plot_occ(zh, zh_base, gmms, shape_samples, folder_name, verbose=True, res=res, from_quantized=from_quantized, tf_sample_dirname=tf_sample_dirname, attn_weights=attn_weights)

    # 1617, 3148, 2175, 529, 1435, 660, 553, 719, 679

    def plot_mix(self):
        fixed_items = torch.randint(low=0, high=self.opt.dataset_size, size=(self.opt.batch_size,))
        with torch.no_grad():
            z, _, gmms = self.model.get_embeddings(fixed_items.to(self.device))
            z = self.mix_z(gmms, z)
            self.plot_occ(z, None, "mix", verbose=True, use_item_id=True, res=200, fixed_items=fixed_items)


    def interpolate_seq(self, num_mid: int, *seq: str, res=200):
        logger = train_utils.Logger().start((len(seq) - 1) * num_mid)
        for i in range(len(seq) - 1):
            self.interpolate_from_files(seq[i], seq[i + 1], num_mid, res, i * num_mid, logger=logger)
        logger.stop()

    def interpolate(self, item_a: Union[str, int], item_b: Union[str, int], num_mid: int, res: int = 200):
        if type(item_a) is str:
            self.interpolate_from_files(item_a, item_b, num_mid, res)
        else:
            zh = self.model.interpolate(item_a, item_b, num_mid)[0]
            for i in range(num_mid):
                mesh = self.get_mesh(zh[i], res)
                files_utils.export_mesh(mesh, f"{self.opt.cp_folder}/interpolate/{item_a}_{item_b}_{i}")
                print(f'done {i + 1:d}/{num_mid}')

    @property
    def device(self):
        return self.opt.device

    def measure_time(self, num_samples: int, *res: int):

        fixed_items = torch.randint(low=0, high=self.opt.dataset_size, size=(num_samples,))
        zh, _, gmms, attn_a = self.model.get_embeddings(fixed_items.to(self.device))
        zh, attn_b = self.model.merge_zh(zh, gmms)
        for res_ in res:
            print(f"\nmeasure {res_:d}")
            times_a, times_b = [], []
            for i in range(len(zh)):
                time_a, time_b = self.get_mesh(zh[i], res_, get_time=True)
                if i > 1:
                    times_a.append(time_a)
                    times_b.append(time_b)
            times_a = torch.tensor(times_a).float()
            times_b = torch.tensor(times_b).float()
            for times in (times_a, times_b):
                print(f"avg: {times.mean()}, std: {times.std()}, min: {times.min()}, , max: {times.max()}")

    @models_utils.torch_no_grad
    def random_stash(self, nums_sample: int, name: str):
        z = self.model.get_random_embeddings(nums_sample).detach().cpu()
        files_utils.save_pickle(z, f'{self.opt.cp_folder}/{name}')

    def load_random(self, name: str):
        z = files_utils.load_pickle(f'{self.opt.cp_folder}/{name}')
        self.model.stash_embedding(z)

    @models_utils.torch_no_grad
    def random_samples(self,  nums_sample, res=256):
        logger = train_utils.Logger().start(nums_sample)
        num_batches = nums_sample // self.opt.batch_size + int((nums_sample % self.opt.batch_size) != 0)
        counter = 0
        for batch in range(num_batches):
            if batch == num_batches - 1:
                batch_size = nums_sample - counter
            else:
                batch_size = self.opt.batch_size
            zh, gmms = self.model.random_samples(batch_size)
            for i in range(len(zh)):
                # gmm_ = [gmms[j][i].unsqueeze(0) for j in range(len(gmms))]
                mesh = self.get_mesh(zh[i], res, None)
                pcd = mesh_utils.sample_on_mesh(mesh, 2048, sample_s=mesh_utils.SampleBy.AREAS)[0]
                files_utils.save_np(pcd, f'{self.opt.cp_folder}/gen/pcd_{counter:04d}')
                files_utils.export_mesh(mesh, f'{self.opt.cp_folder}/gen/{counter:04d}')
                logger.reset_iter()
                counter += 1
        logger.stop()
            # self.plot_occ(zh, gmms, prefix, res=res, verbose=True)

    def get_plot_scale(self):
        scale = files_utils.load_pickle(f'{constants.CACHE_ROOT}{self.opt.dataset_name}/scale')
        if scale is None:
            return 1.
        return scale['global_max'] / scale['std']

    def plot_from_file(self, item_a: int, item_b: int):
        data = files_utils.load_pickle(f"{self.opt.cp_folder}/compositions/{item_a:d}_{item_b:d}")
        (item_a, gmms_id_a), (item_b, gmms_id_b) = data
        self.disentanglement_plot(item_a, item_b, gmms_id_a, gmms_id_b, b_inclusive=True)

    def plot_from_file_single(self, item_a: int, item_b: int, in_item: int):
        data = files_utils.load_pickle(f"{self.opt.cp_folder}/compositions/{item_a:d}_{item_b:d}")
        (item_a, gmms_id_a) = data[in_item]
        self.disentanglement_plot(item_a, -1, gmms_id_a, [], b_inclusive=True)

    def get_mesh_from_mid(self, gmm, included: T, res: int) -> Optional[T_Mesh]:
        if self.mid is None:
            return None
        gmm = [elem.to(self.device) for elem in gmm]
        included = included.to(device=self.device)
        mid_ = self.mid[included[:, 0], included[:, 1]].unsqueeze(0)
        zh = self.model.merge_zh(mid_, gmm)[0]
        # zh = self.model_a.merge_zh(self.mid, gmm, mask=mask)[0]
        # zh = zh[included[0]]
        mesh = self.get_mesh(zh[0], res, [gmm])
        return mesh

    def set_items(self, *items: int):
        items = torch.tensor(items, dtype=torch.int64, device=self.device)
        with torch.no_grad():
            self.mid = self.model.forward_a(items)[0]

    def save_light(self, root, gmms):
        gmms = self.sort_gmms(*gmms)
        save_dict = {'ids': {
            gmm.shape_id: [gaussian.gaussian_id[1] for gaussian in gmm.gmm if gaussian.included]
            for gmm in self.gmms if gmm.included},
            'gmm': gmms}
        path = f"{root}/{files_utils.get_time_name('light')}"
        files_utils.save_pickle(save_dict, path)

    @models_utils.torch_no_grad
    def mix_file(self, path: str, z_idx, replace_inds, num_samples: int, res=220):
        zh, gmms, _ , base_inds = self.load_file(path, disclude=z_idx)
        # select = torch.ones(zh.shape[1], dtype=torch.bool)
        # select[torch.tensor(z_idx, dtype=torch.int64)] = False
        replace_inds_t = torch.tensor(replace_inds, dtype=torch.int64, device=self.device)
        # select = select.to(self.device)

        # gmms = [item[0, :, select] for item in gmms]
        select_random = torch.rand(1000).argsort()[:num_samples]
        zh_new, _, gmms_new, _ = self.model.get_embeddings(select_random.to(self.device))
        zh_new = zh_new[:, replace_inds_t]
        gmms_new = [item[:, :, replace_inds_t] for item in gmms_new[0]]
        name = files_utils.get_time_name('light')
        for i in range(num_samples):
            zh_ = torch.cat((zh, zh_new[i].unsqueeze(0)), dim=1)
            gmms_ = [torch.cat((item, item_new[i].unsqueeze(0)), dim=2) for item, item_new in zip(gmms, gmms_new)]
            zh_ = self.model.merge_zh_step_a(zh_, [gmms_])
            zh_, _ = self.model.affine_transformer.forward_with_attention(zh_)
            mesh = self.get_mesh(zh_[0], res, None)
            files_utils.export_mesh(mesh, f"{self.opt.cp_folder}/occ_mix/mix_{i:02d}")
            save_dict = {'ids': base_inds | {select_random[i].item(): replace_inds},
                         'gmm': [item.cpu() for item in gmms_]}
            path = f"{self.opt.cp_folder}/occ_mix_light/{name}_{i:02d}"
            files_utils.save_pickle(save_dict, path)
            # files_utils.export_gmm()


    def random_gt(self):
        ds_train = mesh_datasets.CacheDataset(self.opt.dataset_name, self.opt.num_samples, self.opt.data_symmetric)
        ds_test = mesh_datasets.CacheDataset(self.opt.dataset_name.replace('train', 'test'), self.opt.num_samples, self.opt.data_symmetric)
        num_test = min(500, len(ds_test))
        num_train = 1000 - num_test
        cls = self.opt.dataset_name.split('_')[1]
        counter = 0
        for num_items, ds in zip((num_test, num_train), (ds_test, ds_train)):
            select = np.random.choice(len(ds), num_items, replace=False)
            for item in select:
                mesh_name = f'{constants.Shapenet_WT}/{cls}/{ds.get_name(item)}'
                mesh = files_utils.load_mesh(mesh_name)
                mesh = mesh_utils.to_unit_sphere(mesh, scale=.9)
                pcd = mesh_utils.sample_on_mesh(mesh, 2048, sample_s=mesh_utils.SampleBy.AREAS)[0]
                files_utils.save_np(pcd, f'{constants.CACHE_ROOT}/evaluation/generation/{cls}/gt/pcd_{counter:04d}')
                counter += 1

    def plot_single(self, *names: str, res: int = 200):
        gmms = []
        items = []
        paths = []
        included = []
        for name in names:
            paths.append(f'{self.opt.cp_folder}/single_edit/{name}')
            phi, mu, eigen, p, include = files_utils.load_gmm(paths[-1], device=self.device)
            gmms.append([item.unsqueeze(0) for item in (mu, p, phi, eigen)])
            items.append(torch.tensor([int(name.split('_')[0])], device=self.device, dtype=torch.int64))
            included.append(include)
        zh, _, _, attn_a = self.model.forward_a(torch.cat(items, dim=0))
        gmms = [torch.stack(elem) for elem in zip(*gmms)]
        included = torch.stack(included)
        zh = self.model.merge_zh(zh, gmms, mask=~included)
        for z, path, include in zip(zh, paths, included):
            z = z[include]
            mesh = self.get_mesh(z, res)
            files_utils.export_mesh(mesh, path)

    def __init__(self, opt: Options):
        self.opt = opt
        model: Tuple[OccGen, Options] = train_utils.model_lc(opt)
        self.model, self.opt = model
        self.model.eval()
        self.temperature = 1.
        self.plot_scale = self.get_plot_scale()
        self.mid: Optional[T] = None
        self.gmms: Optional[TN] = None
        self.get_rotation = utils.rotation_utils.rand_bounded_rotation_matrix(100000)
        # self.load_random("chairs_slides")
        self.meshing = mcubes_meshing.MarchingCubesMeshing(self.device, scale=self.plot_scale, max_num_faces=20000)


def attention_to_image(attn: T):
    attn = attn.view(1, 1, *attn.shape)
    attn = nnf.interpolate(attn, scale_factor=16)[0,0]
    image = image_utils.to_heatmap(attn) * 255
    image = image.numpy().astype(np.uint8)
    return image


def look_on_attn():
    attn = files_utils.load_pickle(f'../assets/checkpoints/occ_gmm_chairs_sym/attn/samples_5901')
    attn_b = attn["attn_b"]
    attn_a = attn["attn_a"]
    for j in range(1):
        attn_b_ = torch.stack(attn_b).max(-1)[0][:, 0]
        for i in range(4):
            image = attn_b_[i].float()
            image_ = image.gt(image.mean() + image.std()).float()
            for_print = (image_ - torch.eye(16)).relu().bool()
            inds = torch.where(for_print)
            print(inds[0].tolist())
            print(inds[1].tolist())
            image = attention_to_image(image)
            files_utils.save_image(image, f'../assets/checkpoints/occ_gmm_chairs_sym/attn/b_{i}_5901.png')
    return


def beautify():
    path = f"{constants.RAW_ROOT}/ui_export01"
    mesh = files_utils.load_mesh(path)
    mesh = mesh_utils.decimate_igl(mesh, 50000)
    mesh = mesh_utils.trimesh_smooth(mesh, iterations=40)
    files_utils.export_mesh(mesh, f"{constants.RAW_ROOT}smooth/ui_export01_smooth2")


if __name__ == '__main__':
    from utils import image_utils
    # beautify()
    # look_on_attn()
    opt_ = Options(device=CUDA(0), tag='chairs_sym_hard', model_name='occ_gmm').load()
    inference = Inference(opt_)
    # inference.load_random("chairs_slides")
    # inference.plot()
    # inference.random_stash(1000, f"chairs_slides")
    # inference.random_gt()
    # names= ['cace287f0d784f1be6fe3612af521500', 'c953d7b4f0189fe6a5838970f9c2180d',
    #         'ca01fd0de2534323c594a0e804f37c1a', 'cb1986dd3e968310664b3b9b23ddfcbc']
    # names = ['cb714c518e3607b8ed4feff97703cf03', 'c9d5ff677600b0a1a01ae992b62200ab', 'c93a696c4b6f843122963ea1e168015d',
    #          'cbbbb3aebaf2e112ca07b3f65fc99919', 'ca2294ffc664a55afab1bffbdecd7709',
    #          'cb17f1916ade6e005d5f1108744f02f1', 'c88eb06478809180f7628281ecb18112', 'c93113c742415c76cffd61677456447e',
    #          'c976cb3eac6a89d9a0aa42b42238537d', 'c9fa3d209a43e7fd38b39a90ee80e328', 'ca84b42ab1cfc37be25dfc1bbeae5325',
    #          'ca9f1525342549878ad57b51c4441549', 'c833ef6f882a4b2a14038d588fd1342f', 'c92721a95fe44b018039b09dacd0f1a7',
    #          'cb867c64ea2ecb408043364ed41c1a79', 'cbbf0aacb76a1ed17b20cb946bceb58f', 'c9f5c127b44d0538cb340854b82a069f',
    #          'c98e1a3e61caec6a67d783b4714d4324', 'c95e8fc2cf96b9349829306a513f9466', 'c98c12e85a3f70a28ddc51277f2e9733']

    # names = ['cf911e274fb613fbbf3143b1cb6076a', 'cf93f33b52900c64bbf3143b1cb6076a', 'cfaff76a1503d4f562b600da24e0965', 'cfb555a4d82a600aca8607f540cc62ba', 'cfd42bf49322e79d8deb28944f3f72ef', 'd01da87ecbc2deea27e33924ab17ba05', 'd0e517321cdcd165939e75b12f2e5480', 'd0ee4253d406b3f05e9e2656aff7dd5b', 'd1a887a47991d1b3bc0909d98a1ff2b4', 'd1b28579fde95f19e1873a3963e0d14', 'd1cdd239dcbfd018bbf3143b1cb6076a', 'd1d308692cb4b6059a6e43b878d5b335', 'd1df81e184c71e0f26360e1e29a956c7']
    # # names = ['d0e517321cdcd165939e75b12f2e5480']
    names = files_utils.collect(f'{constants.DATA_ROOT}ui_export/occ_gmm_chairs_sym_hard_05_20-14_48', '.pkl')
    names = [''.join(item) for item in names][3:]
    inference.interpolate_seq(32, *names, res=256)

    # inference.mix_file(f"{opt_.cp_folder}/occ_mix_light_seat/light_01_25-18_00_30",
    #                    (2, 6, 7, 10, 14, 15), (2, 6, 7, 10, 14, 15), num_samples=30, res=220)
    # inference.plot_folder(f"{opt_.cp_folder}/occ_mix_light/", res=220)
    # inference.plot_folder(f"{constants.DATA_ROOT}ui_export/occ_gmm_tables_no_dis_split_01_27-22_45", res=220)


    # inference.plot("samples", verbose=True, interpolate=False, res=220, size=50)
# "\begin{figure}
# \centering
# % \footnotesize
#     \begin{overpic}[width=1\columnwidth,tics=10, trim=0mm 0 0mm 0,clip]{figures/coarse_diagram_v2_hor-01.png}
#
#      \put(-.8,14){$\mathbf{\za}$}
#
#     \put(34.6,13.4){$\mathbf{\zB}$}
#
#      \put(67.9,13.4){$\mathbf{\zC}$}
#
#      \put(18.1,-2.2){$\netA$}
#      \put(52,-2.2){$\netB$}
#      \put(84.5,-2.2){$\netC$}
#
#     %  \put(-2,-2){{$\mathbf{\za}$: shape embedding}}
#     %  \put(33,-2){{$\mathbf{\zB}$: part embedding}}
#     %  \put(66,-2){{$\mathbf{\zC}$: contextual embedding}}
#
#     \end{overpic}
#
# \caption{Method overview. Our implicit shape generative model learns to decompose a shape embedding ${\za}$ into distinct embeddings $\zB$ that correspond to distinct 3D parts. Then, a mixing network outputs contextual embeddings $\zC$. Finally the implicit shape is given by a third occupancy network $\netC$ that is conditioned on the contextual part embeddings.}
#
#
# \label{fig:overview}
#
#
# \end{figure}
# "
