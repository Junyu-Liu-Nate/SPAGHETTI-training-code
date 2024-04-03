from custom_types import *
import constants
from utils import mesh_utils, files_utils, train_utils
from threading import Thread
import os
import multiprocessing as mp
import random
import json

DATASET_PATH = "/Volumes/DataSSDLJY/Data/Research/dataset/"
PROJECT_PATH = "/Volumes/DataSSDLJY/Data/Research/project/BVC/ITSS/"

class MeshSampler(Dataset):

    def __len__(self):
        return self.labels.shape[0] * self.labels.shape[1]

    def __getitem__(self, item):
        i = 0
        while item >= self.labels.shape[1]:
            item -= self.labels.shape[1]
            i += 1
        return self.points[i, item], self.labels[i, item]

    def get_surface_points(self, num_samples: int):
        counter = 0
        points = []
        while counter < num_samples:
            points_ = mesh_utils.sample_on_mesh(self.mesh, num_samples)[0]
            points.append(points_)
            counter += points_.shape[0]
        points = torch.cat(points, dim=0)
        if counter > num_samples:
            points = points[:num_samples]
        return points

    def get_labels(self, points: T) -> T:
        return torch.from_numpy(mesh_utils.get_fast_inside_outside(self.mesh, points.numpy()))

    def init_in_out(self, total=6e5, max_trials=50,  symmetric=(0, 0, 0)) -> TS:
        split_size = int(total // 4)
        on_surface_points = self.get_surface_points(split_size)
        near_points_a = on_surface_points + torch.randn(on_surface_points.shape) * .007
        near_points_b = on_surface_points + torch.randn(on_surface_points.shape) * .02
        inside_points_ = (torch.rand(split_size, 3) * 2 - 1) * self.global_scale
        for i in range(3):
            if symmetric[i] == 1:
                inside_points_[:, i] = inside_points_[:, i].abs()
        random_points = inside_points_
        labels_near_a = self.get_labels(near_points_a)
        labels_near_b = self.get_labels(near_points_b)
        labels_inside = labels_random = self.get_labels(random_points)
        inside_points = [near_points_b[~labels_near_b]]
        counter_inside = inside_points[-1].shape[0]
        trial = 0
        while counter_inside < split_size and trial < max_trials:
            if trial == max_trials - 1:
                self.error = True
                return torch.zeros(1), torch.zeros(1)
            inside_points.append(inside_points_[~labels_inside])
            counter_inside += inside_points[-1].shape[0]
            if counter_inside < split_size:
                inside_points_ = self.mesh_bb[0][None, :] + self.mesh_bb[1][None, :] * torch.rand(split_size, 3)
                for i in range(3):
                    if symmetric[i] == 1:
                        inside_points_[:, i] = inside_points_[:, i].abs()
                labels_inside = self.get_labels(inside_points_)
            trial += 1
        inside_points = torch.cat(inside_points, dim=0)[:split_size]
        inside_points = inside_points[torch.rand(inside_points.shape[0]).argsort()]
        all_points = torch.stack((near_points_a, near_points_b, random_points, inside_points), dim=0)
        labels = torch.stack((labels_near_a, labels_near_b, labels_random), dim=0)
        return all_points, labels

    def __init__(self, path, symmetric, num_samples=6e5):
        self.error = False
        mesh = files_utils.load_mesh(path)
        self.global_scale = 1
        mesh = mesh_utils.to_unit_sphere(mesh, scale=.90)
        mesh_bb = mesh[0].min(0)[0], mesh[0].max(0)[0]
        self.mesh_bb = mesh_bb[0], mesh_bb[1] - mesh_bb[0]
        self.mesh = mesh
        self.split = (2, 2, 2)
        self.points, self.labels = self.init_in_out(total=num_samples, symmetric=symmetric)


def save_np_samples(paths, items: T, out_root, logger: train_utils.Logger, symmetric):
    shape_counter = 0
    finish_counter = 0
    # print(paths[0])
    # print(items)
    for i in items:
        path = paths[i]
        # print(path)
        item_id = path[1]
        # shape_num, shape_cat, shapenet_id = shapes[i]
        shapenet_id = path[1]
        mesh_file = ''.join(paths[i])
        np_path = f'{out_root}/{item_id}'
        if files_utils.is_file(mesh_file):
            shape_counter += 1
            if not files_utils.is_file(np_path + '_pts.npy'):
                try:
                    # sampler = MeshSampler(symmetric, mesh_file)
                    #TODO: May need to specify the total sampling num -> should be 2e7 to match the paper
                    sampler = MeshSampler(mesh_file, symmetric)
                    points, labels = sampler.points.cpu().numpy(), sampler.labels.cpu().numpy()
                    if not sampler.error:
                        files_utils.save_np(points, f"{np_path}_pts")
                        files_utils.save_np(labels, f"{np_path}_lbl")
                        finish_counter += 1
                    else:
                        print(f'error: {shapenet_id}')
                except BaseException:
                    # break
                    print(f'Exception: {shapenet_id}')
            else:
                finish_counter += 1
        logger.reset_iter()
    # print(f'mesh files: {shape_counter} / {len(items)} {float(shape_counter) / len(items)}')
    # print(f'np files: {finish_counter} / {shape_counter} {float(finish_counter) / shape_counter}')


def get_std(paths: List[List[str]]) -> TS:
    vs = []
    for path in paths:
        mesh = files_utils.load_mesh(''.join(path))
        vs_ = mesh_utils.sample_on_mesh(mesh, 500, sample_s=mesh_utils.SampleBy.AREAS)[0]
        vs.append(mesh_utils.to_center(vs_))
    vs = torch.cat(vs, dim=0)
    global_max = vs.abs().max()
    return vs.std(), global_max


def get_num_to_cat() -> Dict[str, str]:
    num2cat = {}
    with open(f'{constants.Shapenet}/num2cat.txt', 'r') as f:
        for line in f:
            raw = line.strip().split(' ')
            if len(raw) == 2:
                num2cat[raw[0]] = raw[1]
    return num2cat


def main_cat(name, symmetric=(0, 0, 0)):
    root = f'{constants.Shapenet_WT}/{name}/'
    out = f'{constants.CACHE_ROOT}/{name}'
    paths = files_utils.collect(root, '.obj', '.off')
    logger = train_utils.Logger().start(len(paths))
    num_threads = 1
    split_size = len(paths) // num_threads
    threads = []
    target = save_np_samples

    for i in range(num_threads):
        if i == num_threads - 1:
            split_size_ = len(paths) - (num_threads - 1) * split_size
        else:
            split_size_ = split_size
        items = torch.arange(split_size_) + i * split_size
        # args = (paths, out, items, logger, symmetric)
        args = (paths, items, out, logger, symmetric)
        if num_threads == 1:
            target(*args)
        else:
            threads.append(Thread(target=target, args=args))
    if num_threads > 1:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    logger.stop()

def main_cat_presplit(name, symmetric=(0, 0, 0)):
    root = f'{constants.Shapenet_WT}/{name}/'
    out = f'{constants.CACHE_ROOT}/{name}'

    presplit_data_file = os.path.join() #TODO:fill up
    paths = []
    
    logger = train_utils.Logger().start(len(paths))
    num_threads = 1
    split_size = len(paths) // num_threads
    threads = []
    target = save_np_samples

    for i in range(num_threads):
        if i == num_threads - 1:
            split_size_ = len(paths) - (num_threads - 1) * split_size
        else:
            split_size_ = split_size
        items = torch.arange(split_size_) + i * split_size
        args = (paths, items, out, logger, symmetric)
        if num_threads == 1:
            target(*args)
        else:
            threads.append(Thread(target=target, args=args))
    if num_threads > 1:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    logger.stop()

def run_watertight_script(in_path, out_path, nul_output, use_plus: bool):
    os.system(f"{constants.MANIFOLD_SCRIPT}/manifold {in_path} {out_path}{nul_output}")


def to_wm(in_path: str, out_path: str, num_faces: Optional[int], use_plus: bool, verbose: bool = False):
    # in_path, out_path = files_utils.add_suffix(in_path, '.obj'), files_utils.add_suffix(out_path, '.obj')
    nul_output = f'{"" if verbose else f" > /dev/null 2>&1"}'
    extension = files_utils.split_path(in_path)[-1]
    if extension != '.obj':
        mesh = files_utils.load_mesh(in_path)
        files_utils.export_mesh(mesh, "./temp.obj")
        in_path = './temp.obj'
    if not os.path.isdir(constants.MANIFOLD_SCRIPT):
        raise NotADirectoryError(f'{constants.MANIFOLD_SCRIPT} not found')
    if num_faces is not None:
        run_watertight_script(in_path, "./temp.obj", nul_output, use_plus)
        # offset_surface("./temp.obj", 0.002)
        os.system(f"{constants.MANIFOLD_SCRIPT}/simplify -i ./temp.obj -o {out_path} -f {num_faces}{nul_output}")
    else:
        run_watertight_script(in_path, out_path, nul_output, use_plus)


# def get_shapenet13_shapes(train):
#     txt_path = f'{constants.Shapenet}/all_vox256_img_{"train" if train else "test"}.txt'
#     shapes = []
#     num2cat = get_num_to_cat()
#     # histogram = {}
#     with open(txt_path, 'r') as f:
#         for line in f:
#             cat_num, shape_mum = line.strip().split('/')
#             cat_name = num2cat[cat_num]
#             # if cat_name not in histogram:
#             #     histogram[cat_name] = 0
#             # histogram[cat_name] += 1
#             shapes.append((cat_num, cat_name, shape_mum))
#     return shapes

### Hardcode the train/test split as there is no all_vox256_img file in ShapeNet v2
def get_shapenet13_shapes(train):
    cat_num = '02828884'
    cat_name = 'bench'
    shapes = []

    shape_path = os.path.join(DATASET_PATH, 'ShapeNetCore_v2', cat_num)
    all_shape_names = [d for d in os.listdir(shape_path) if os.path.isdir(os.path.join(shape_path, d))]
    random.shuffle(all_shape_names)
    split_index = int(len(all_shape_names) * 0.8)
    train_names = all_shape_names[:split_index]
    test_names = all_shape_names[split_index:]

    ### Save train-test splits for once
    tain_test_split_save_path = os.path.join(PROJECT_PATH, 'generated', 'Spaghetti', 'new_splits')
    if not os.path.exists(tain_test_split_save_path):
        os.makedirs(tain_test_split_save_path)
    train_split_path = os.path.join(tain_test_split_save_path, cat_name + '_' + cat_num + '_train.json') 
    test_split_path = os.path.join(tain_test_split_save_path, cat_name + '_' + cat_num + '_test.json')
    if not os.path.exists(train_split_path):
        with open(train_split_path, 'w') as f:
            json.dump(train_names, f)
    if not os.path.exists(test_split_path):
        with open(test_split_path, 'w') as f:
            json.dump(test_names, f) 

    if train:
        for train_name in train_names:
            shapes.append((cat_num, cat_name, train_name))
    else:
        for test_name in test_names:
            shapes.append((cat_num, cat_name, test_name))

    return shapes


def watertight_process(shapes, use_logger, process_id):
    shapenet_root = constants.Shapenet
    shapenet_wt_root_ = constants.Shapenet_WT
    print(f'process {process_id}: start')
    if use_logger:
        logger = train_utils.Logger().start(len(shapes))
    for i, (shape_num, shape_cat, shapenet_id) in enumerate(shapes):
        out_file = f'{shapenet_wt_root_}/{shape_cat}s/{shapenet_id}.obj'
        if not files_utils.is_file(out_file):
            files_utils.init_folders(out_file)
            in_file = f'{shapenet_root}/{shape_num}/{shapenet_id}/models/model_normalized.obj'
            if files_utils.is_file(in_file):
                to_wm(in_file, out_file, None, False)
        if use_logger:
            logger.reset_iter()
        elif (i + 1) % 100 == 0:
            print(f'process {process_id}: done {i}')
    if use_logger:
        logger.stop()
    print(f'process {process_id}: end')


def watertight():
    num_processes = 1
    shapes_base = get_shapenet13_shapes(True)
    shapes = []
    shape_stack = []
    last_found = True
    for i, (shape_num, shape_cat, shapenet_id) in enumerate(shapes_base):
        out_file = f'{constants.Shapenet_WT}/{shape_cat}/{shapenet_id}.obj'
        if not files_utils.is_file(out_file):
            if last_found:
                shape_stack.append((shape_num, shape_cat, shapenet_id))
            else:
                shapes.append((shape_num, shape_cat, shapenet_id))
                if len(shape_stack) > 0:
                    shapes.append(shape_stack[0])
                shape_stack = []
            last_found = False
        else:
            shape_stack = []
            last_found = True
    if num_processes == 1:
        watertight_process(shapes, True, 0)
    else:
        processes = []
        batch_size = len(shapes) // num_processes
        for i in range(num_processes):
            start = i * batch_size
            end = (i + 1) * batch_size if i < num_processes - 1 else len(shapes)
            shape_process = shapes[start: end]
            processes.append(mp.Process(target=watertight_process, args=(shape_process, False, i)))
        for process in processes:
            process.start()
        for process in processes:
            process.join()


if __name__ == "__main__":
    watertight()
    main_cat('benchs', symmetric=(1, 0, 0))
