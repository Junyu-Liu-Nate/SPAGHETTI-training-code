
import models.models_utils
from custom_types import *
from utils import files_utils, mesh_utils
import constants
from utils import train_utils
from threading import Thread
import make_data
import abc
# import h5py
import options
import zipfile
import os


class OccDataset(Dataset, abc.ABC):

    def __len__(self):
        return len(self.paths)

    @abc.abstractmethod
    def get_samples_(self, item: int, total: float) -> TS:
        raise NotImplemented

    def get_samples(self, item: int, total: float = 5e5) -> TS:
        if self.data[item] is None:
            self.data[item] = self.get_samples_(item, total)
        return self.data[item]

    @staticmethod
    def shuffle_(points: T, labels: Optional[T] = None) -> Union[T, TS]:
        order = torch.rand(points.shape[0], device=points.device).argsort()
        if labels is None:
            return points[order]
        return points[order], labels[order]

    @abc.abstractmethod
    def shuffle(self, item: int, *args):
        raise NotImplemented

    def sampler(self, item, points, labels):
        points_ = points[:, self.counter[item] * self.num_samples: (1 + self.counter[item]) * self.num_samples]
        labels_ = labels[:, self.counter[item] * self.num_samples: (1 + self.counter[item]) * self.num_samples]
        self.counter[item] += 1
        if (self.counter[item] + 1) * self.num_samples > points.shape[1]:
            self.counter[item] = 0
            self.shuffle(item)
        return points_, labels_

    def get_large_batch(self, item: int, num_samples: int):
        points, labels = self.get_samples(item)
        select = torch.rand(points.shape[1]).argsort()[:num_samples]
        points, labels = points[:, select], labels[:, select]
        return points, labels

    def __getitem__(self, item: int):
        points, labels = self.get_samples(item)
        points, labels = self.sampler(item, points, labels)
        for axis in self.symmetric_axes:
            points_ = points.clone()
            points_[:, :, axis] = -points_[:, :, axis]
            points = torch.cat((points, points_), dim=1)
            labels = torch.cat((labels, labels), dim=1)
        return points, labels, item

    @staticmethod
    @abc.abstractmethod
    def collect(ds_name: str) -> List[List[str]]:
        raise NotImplemented

    def filter_paths(self, paths: List[List[str]]) -> List[List[str]]:
        if self.split_path is not None:
            names = files_utils.load_json(self.split_path)["ShapeNetV2"]
            names = list(names.items())[0][1]
            paths = list(filter(lambda x: x[1] in names, paths))
        return paths

    def __init__(self, ds_name: str, num_samples: int, symmetric: Tuple[bool, bool, bool], split_path: Optional[str] = None):
        self.split_path = split_path
        self.num_samples = num_samples
        paths = self.collect(ds_name)
        self.paths = self.filter_paths(paths)
        self.data: List[TSN] = [None] * len(self)
        self.counter = [0] * len(self)
        self.symmetric_axes = [i for i in range(len(symmetric)) if symmetric[i]]


class MeshDataset(OccDataset):

    def load_mesh(self, item: int):
        mesh = files_utils.load_mesh(''.join(self.paths[item]))
        mesh = mesh_utils.triangulate_mesh(mesh)[0]
        mesh = mesh_utils.to_unit_sphere(mesh)
        return mesh

    def get_samples_(self, item: int, total: float) -> TS:
        mesh = self.load_mesh(item)
        on_surface_points = mesh_utils.sample_on_mesh(mesh, int(total * self.split[0]))[0]
        if self.split[1] > 0:
            near_points = on_surface_points + torch.randn_like(on_surface_points) * .01
            random_points = torch.rand(int(total * self.split[1]), 3) * 2 - 1
            all_points = torch.cat((on_surface_points, near_points, random_points), dim=0)
            labels_near = mesh_utils.get_inside_outside(near_points, mesh)
            labels_random = mesh_utils.get_inside_outside(random_points, mesh)
            labels = torch.cat((torch.zeros(on_surface_points.shape[0]), labels_near, labels_random), dim=0)
        else:
            all_points = on_surface_points
            labels = self.labels
        shuffle = torch.argsort(torch.rand(int(total)))
        return all_points[shuffle], labels[shuffle]

    @staticmethod
    def collect(ds_name: str) -> List[List[str]]:
        return files_utils.collect(f'{constants.RAW_ROOT}{ds_name}/', '.obj', '.off')

    def __init__(self, ds_name: str, num_samples: int, flow: int):
        super(MeshDataset, self).__init__(ds_name, num_samples)
        if flow == 1:
            self.split = (.4, .4, .2)
        else:
            self.split = (1., .0, .0)
            self.labels = torch.zeros(int(5e5))


class SingleMeshDataset(OccDataset):

    def __getitem__(self, item: int):
        points, labels, _ = super(SingleMeshDataset, self).__getitem__(0)
        return points, labels, 0

    def __len__(self):
        return self.single_labels.shape[1] // self.num_samples

    def get_samples_(self, _: int, __: float) -> TS:
        return self.single_points, self.single_labels

    def shuffle(self, item: int, *args):
        all_points, labels = self.single_points, self.single_labels
        shuffled = [self.shuffle_(all_points[i], labels[i]) for i in range(labels.shape[0])]
        all_points = torch.stack([item[0] for item in shuffled], dim=0)
        labels = torch.stack([item[1] for item in shuffled], dim=0)
        self.single_points, self.single_labels = all_points, labels

    @staticmethod
    def init_samples(mesh_name: str, symmetric, device: D) -> TS:
        mesh_path = f'{constants.DATA_ROOT}singles/{mesh_name}'
        if not files_utils.is_file(mesh_path + '.npz'):
            sampler = make_data.MeshSampler(mesh_path, CUDA(0), (make_data.ScaleType.Sphere, None, 1.), inside_outside=True,
                                  symmetric=symmetric, num_samples=5e6)
            points, labels = sampler.points, sampler.labels
            data = {'points': points.cpu().numpy(), 'labels': labels.cpu().numpy()}
            if not sampler.error:
                files_utils.save_np(data, mesh_path)
            else:
                print(f'error: {mesh_path}')
        else:
            data: Dict[str, ARRAY] = np.load(mesh_path + '.npz')
        all_points, labels = torch.from_numpy(data['points']), torch.from_numpy(data['labels'])
        return all_points.to(device), labels.to(device)

    @staticmethod
    def collect(mesh_name: str) -> List[List[str]]:
        return []

    def __init__(self,  mesh_name: str, num_samples: int, symmetric: Tuple[bool, bool, bool], device: D):
        self.device = device
        self.single_points, self.single_labels = self.init_samples(mesh_name, symmetric, device)
        super(SingleMeshDataset, self).__init__(mesh_name, num_samples, symmetric)


class CacheDataset(OccDataset):

    # def sampler(self, item, points, labels):
    #     if self.sampler_by is None:
    #         return super(CacheDS, self).sampler(item, points, labels)
    #     surface_inds = np.random.choice(self.sampler_data[item][0], self.num_samples // 2, replace=False)
    #     random_inds = np.random.choice(self.sampler_data[item][1], self.num_samples - self.num_samples // 2, replace=False)
    #     points = torch.cat((points[surface_inds], points[random_inds]), dim=0)
    #     labels = torch.cat((labels[surface_inds], labels[random_inds]))
    #     return points, labels

    def get_name_mapper(self) -> Dict[str, int]:
        if self.name_mapper is None:
            self.name_mapper = {self.get_name(item): item for item in range(len(self))}
        return self.name_mapper

    def get_item_by_name(self, name: str) -> int:
        return self.get_name_mapper()[name]

    def get_name(self, item: int):
        return self.paths[item][1]

    def shuffle(self, item: int, *args):
        return
    # def shuffle(self, item: int, *args):
    #     all_points, labels = self.data[item]
    #     shuffled = [super(CacheDataset, self).shuffle_(all_points[i], labels[i]) for i in range(labels.shape[0])]
    #     all_points = torch.stack([item[0] for item in shuffled], dim=0)
    #     labels = torch.stack([item[1] for item in shuffled], dim=0)
    #     self.data[item] = all_points, labels

    def get_samples_(self, item: int, _) -> TS:
        path = f'{self.root}{self.get_name(item)}'
        all_points = np.load(f"{path}_pts.npy")
        labels = np.load(f"{path}_lbl.npy")
        # data: Dict[str, ARRAY] = np.load(''.join(self.paths[item]))
        # all_points, labels = torch.from_numpy(data['points']), torch.from_numpy(data['labels'])
            # if self.sampler_by is None:
            #     shuffle = torch.argsort(torch.rand(int(labels.shape[0])))
            #     all_points, labels = all_points[shuffle], labels[shuffle]
            # on_surface_ind = labels.abs().lt(0.02)
            # near_surface_ind = torch.where(~on_surface_ind)[0].numpy()
            # on_surface_ind = torch.where(on_surface_ind)[0].numpy()
            # self.sampler_data[item] = (on_surface_ind, near_surface_ind)
            # self.data[item] = all_points, labels
        return torch.from_numpy(all_points), torch.from_numpy(labels)


    @staticmethod
    def collect(ds_name: str) -> List[List[str]]:
        files = files_utils.collect(f'{constants.CACHE_ROOT}{ds_name}/', '.npy')
        files = [file for file in files if '_pts' in file[1]]
        files = [[file[0], file[1][:-4], file[2]] for file in files]
        return files

    def __init__(self, ds_name: str, num_samples: int, symmetric: Tuple[bool, bool, bool],
                 split_path: Optional[str] = None):
        self.root = f'{constants.CACHE_ROOT}{ds_name}/'
        super(CacheDataset, self).__init__(ds_name, num_samples, symmetric, split_path)
        self.sampler_data: List[TSN] = [None] * len(self)
        self.name_mapper: Optional[Dict[str, int]] = None


class CacheImNet(CacheDataset):

    def __getitem__(self, item):
        points, labels, _ = super(CacheImNet, self).__getitem__(item)
        vox = np.load(f"{self.im_net_root}{self.get_name(item)}.npy")
        vox = 1 - 2 * vox
        # try:
        #
        # except ValueError:
        #     files_utils.delete_single(f"{self.im_net_root}{self.get_name(item)}.npy")
        #     raise BaseException
        vox = torch.from_numpy(vox).view(1, 64, 64, 64).float()
        return points, labels, vox, item

    def filter_paths(self, paths: List[List[str]]) -> List[List[str]]:
        names_im_net = set(map(lambda x: x[1], files_utils.collect(self.im_net_root, '.npy')))
        paths = filter(lambda x: x[1] in names_im_net, paths)
        return list(paths)

    def __init__(self, cls: str, ds_name: str, num_samples: int, symmetric: Tuple[bool, bool, bool]):
        self.im_net_root = f'{constants.CACHE_ROOT}im_net/{cls}/'
        super(CacheImNet, self).__init__(ds_name, num_samples, symmetric)


class CacheInOutDataset(CacheDataset):

    def shuffle(self, item: int, *args):
        inside_points = self.data[item][0][3]
        inside_points = self.shuffle_(inside_points).unsqueeze_(0)
        super(CacheInOutDataset, self).shuffle(item, *args)
        all_points, labels = self.data[item]
        all_points = torch.cat((all_points, inside_points), dim=0)
        self.data[item] = all_points, labels

    @staticmethod
    def collect(ds_name: str, split_path: Optional[str] = None) -> List[List[str]]:
        return files_utils.collect(f'{constants.CACHE_ROOT}inside_outside/{ds_name}/', '.npz')


class CacheH5Dataset(CacheDataset):

    def get_name(self, item: int):
        return self.names[item]

    def get_samples(self, item: int, total: float = 5e5) -> TS:
        dataset = self.dataset[0]
        for i in range(len(self.lengths)):
            if item < self.lengths[i]:
                dataset = self.dataset[i]
                break
            item -= self.lengths[i]
        return torch.from_numpy(dataset['points'][item]).float(), torch.from_numpy(dataset['labels'] [item]).float()

    def shuffle(self, item: int, *args):
        return

    def __len__(self):
        return sum(self.lengths)

    @staticmethod
    def collect(ds_name: str) -> List[List[str]]:
        return files_utils.collect(f'{constants.CACHE_ROOT}inside_outside/{ds_name}/', '.hdf5')

    def __init__(self, ds_name: str, num_samples: int, symmetric: Tuple[bool, bool, bool]):
        paths = files_utils.collect(f'{constants.CACHE_ROOT}inside_outside/{ds_name}/', '.hdf5')
        self.dataset = [h5py.File(''.join(path), "r") for path in paths]
        self.lengths = [dataset['points'].shape[0] for dataset in self.dataset]
        super(CacheH5Dataset, self).__init__(ds_name, num_samples, symmetric)
        self.names = files_utils.load_pickle(f'{constants.CACHE_ROOT}inside_outside/{ds_name}/all_data_names')


class SimpleDataLoader:

    @staticmethod
    def tensor_reducer(items: TS) -> T:
        return torch.stack(items, dim=0)

    @staticmethod
    def number_reducer(dtype) -> Callable[[List[float]], T]:
        def reducer_(items: List[float]) -> T:
            return torch.tensor(items, dtype=dtype)
        return reducer_

    @staticmethod
    def default_reducer(items: List[Any]):
        return items

    def __iter__(self):
        self.counter = 0
        self.order = torch.rand(len(self.ds)).argsort()
        return self

    def __len__(self):
        return self.length

    def collate_fn(self, raw_data):
        batch = [self.reducers[i](items) for i, items in enumerate(zip(*raw_data))]
        return batch

    def init_reducers(self, sample):
        reducers = []
        for item in sample:
            if type(item) is T:
                reducers.append(self.tensor_reducer)
            elif type(item) is int:
                reducers.append(self.number_reducer(torch.int64))
            elif type(item) is float:
                reducers.append(self.number_reducer(torch.float32))
            else:
                reducers.append(self.default_reducer)
        return reducers

    def __next__(self):
        if self.counter < len(self):
            start = self.counter * self.batch_size
            indices = self.order[start:  min(start + self.batch_size, len(self.ds))].tolist()
            raw_data = [self.ds[ind] for ind in indices]
            self.counter += 1
            return self.collate_fn(raw_data)
        else:
            raise StopIteration

    def __init__(self, ds: Dataset, batch_size: int = 1):
        self.ds = ds
        self.batch_size = batch_size
        self.counter = 0
        self.length = len(ds) // self.batch_size + int(len(self.ds) % self.batch_size != 0)
        self.order: Optional[T] = None
        self.reducers = self.init_reducers(self.ds[0])


def save_np_mesh_thread(name: str, items: T, logger: train_utils.Logger):
    ds = MeshDataset(name, 512, 1)
    root = f'{constants.RAW_ROOT}shapenet_numpy/{name}/'
    items = items.tolist()
    for i in items:
        item_id = ds.paths[i][0].split('/')[-3]
        save_path = f'{root}{item_id}'
        if not files_utils.is_file(save_path + '.npy'):
            mesh = ds.load_mesh(i)
            np_mesh = (mesh[0][mesh[1]]).numpy()
            files_utils.save_np(np_mesh, save_path)
        logger.reset_iter()


def save_np_mesh(name):
    ds = MeshDataset(name, 512, 1)
    logger = train_utils.Logger().start(len(ds))
    num_threads = 4
    split_size = len(ds) // num_threads
    threads = []
    for i in range(num_threads):
        if i == num_threads - 1:
            split_size_ = len(ds) - (num_threads - 1) * split_size
        else:
            split_size_ = split_size
        items = torch.arange(split_size_) + i * split_size
        threads.append(Thread(target=save_np_mesh_thread, args=(name, items, logger)))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    logger.stop()


def export_points():
    opt = options.OptionsSingle()
    ds = SingleMeshDataset(opt.dataset_name, opt.num_samples,
                            opt.symmetric, opt.device)
    points = ds.single_points
    # points =  np.load('/home/amir/projects/sdf_gmm/assets/singles/MalteseFalconSolid.npz')['points']
    colors = [(1., 0, 0), (0, 1., 0), (0, 0, 1.), (1, .5, 0)]
    for i in range(4):
        select = torch.rand(points.shape[1]).argsort()[:10000]
        colors_ = torch.tensor(colors[i]).unsqueeze(0).expand(10000, 3)
        pts = points[i, select.numpy()]
        files_utils.export_mesh(pts, f"{constants.DATA_ROOT}/tmp/{i}", colors=colors_)


def to_npy():
    root = f"/data/amir/cache/inside_outside/shapenet_guitars_wm_sphere/"
    paths = files_utils.collect(root, ".h5py")
    counter = 0
    logger = train_utils.Logger()
    names = files_utils.load_pickle(f"{root}/all_data_names.pkl")
    for path in paths:
        dataset = h5py.File(''.join(path), "r")
        points = dataset['points']
        labels = dataset['labels']
        logger.start(points.shape[0])
        for i in range(points.shape[0]):
            points_, labels_ = V(points[i]), V(labels[i])
            files_utils.save_np(points_, f"{root}/{names[counter]}_pts")
            files_utils.save_np(labels_,
                                f"{root}/{names[counter]}_lbl")
            counter += 1
            logger.reset_iter()
        logger.stop()
        return
    # df.to_pickle(f"/data/amir/cache/inside_outside/shapenet_chairs_wm_sphere_sym_train/all_data.pd")

    return


def npz_to_npy():
    root = f"{constants.CACHE_ROOT}/inside_outside/shapenet_tables_wm_sphere_sym/"
    paths = files_utils.collect(root, ".npz")
    logger = train_utils.Logger().start(len(paths))
    for path in paths:
        try:
            data = np.load(''.join(path))
        except ValueError:
            logger.reset_iter()
            continue
        name = path[1]
        if not files_utils.is_file(f"{root}/{name}_pts.npy"):
            points = data['points']
            labels = data['labels']
            files_utils.save_np(points, f"{root}/{name}_pts")
            files_utils.save_np(labels, f"{root}/{name}_lbl")
        logger.reset_iter()
    logger.stop()
    return


def get_split_names(split_name: str, suffix: str) -> List[str]:
    split_path = f"{constants.DATA_ROOT}splits/{split_name}_{suffix}.json"
    names = files_utils.load_json(split_path)["ShapeNetV2"]
    names = list(names.items())[0][1]
    return names


def split_npy(ds_name, split_name: str):
    root = f'{constants.CACHE_ROOT}inside_outside/{ds_name}/'
    for suffix in ("test", "train"):
        export_root = f'{constants.CACHE_ROOT}inside_outside/{ds_name}_{suffix}/'
        files_utils.init_folders(export_root)
        names = get_split_names(split_name, suffix)
        for name in names:
            if files_utils.is_file(f'{root}{name}_pts.npy'):
                files_utils.move_file(f'{root}{name}_pts.npy', f'{export_root}{name}_pts.npy')
                files_utils.move_file(f'{root}{name}_lbl.npy', f'{export_root}{name}_lbl.npy')


def create_split_file(ds_name, ratio_test=0.25):

    root = f'{constants.CACHE_ROOT}inside_outside/{ds_name}/'
    names = files_utils.collect(root, '.npy')
    names = [name[1][:-4] for name in names if 'pts' in name[1]]
    split = torch.rand(len(names)).argsort()
    split = split[:6000]
    names_test = [names[split[i]] for i in range(0, int(len(split) * ratio_test))]
    names_train = [names[split[i]] for i in range(int(len(split) * ratio_test), len(split))]
    for suffix, item in zip(("test", "train"), (names_test, names_train)):
        split_path = f"{constants.DATA_ROOT}splits/shapenet_tables_{suffix}.json"
        data = {"ShapeNetV2": {"1234": item}}
        files_utils.save_json(data, split_path)
    return

