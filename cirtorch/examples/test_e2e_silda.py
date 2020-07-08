import argparse
import os
from os import path as osp
import time
import pickle
import pdb

import numpy as np

import torch
from torch.utils.model_zoo import load_url
from torchvision import transforms

from cirtorch.networks.imageretrievalnet import init_network, extract_vectors
from cirtorch.datasets.testdataset import configdataset
from cirtorch.utils.download import download_train, download_test
from cirtorch.utils.evaluate import compute_map_and_print, compute_map_and_print_top_k
from cirtorch.utils.general import get_data_root, htime

PRETRAINED = {
    'rSfM120k-tl-resnet50-gem-w': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet50-gem-w-97bf910.pth',
    'rSfM120k-tl-resnet101-gem-w': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet101-gem-w-a155e54.pth',
    'rSfM120k-tl-resnet152-gem-w': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet152-gem-w-f39cada.pth',
    'gl18-tl-resnet50-gem-w': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet50-gem-w-83fdc30.pth',
    'gl18-tl-resnet101-gem-w': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet101-gem-w-a4d43db.pth',
    'gl18-tl-resnet152-gem-w': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet152-gem-w-21278d5.pth',
}

parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Testing End-to-End')

# test options
parser.add_argument('--data-path', default='/data/datasets/SILDa')
parser.add_argument('--network', '-n', default='gl18-tl-resnet152-gem-w')
parser.add_argument('--image-size', '-imsize', default=1024, type=int, metavar='N',
                    help="maximum size of longer image side used for testing (default: 1024)")
parser.add_argument('--multiscale', '-ms', metavar='MULTISCALE', default='[1, 2**(1/2), 1/2**(1/2)]',
                    help="use multiscale vectors for testing, " +
                         " examples: '[1]' | '[1, 1/2**(1/2), 1/2]' | '[1, 2**(1/2), 1/2**(1/2)]' (default: '[1]')")
parser.add_argument('--top-n', default=20, type=int)

# GPU ID
parser.add_argument('--gpu-id', '-g', default='0', metavar='N',
                    help="gpu id used for testing (default: '0')")


def main():
    args = parser.parse_args()

    # setting up the visible GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # loading network
    # pretrained networks (downloaded automatically)
    print(">> Loading network:\n>>>> '{}'".format(args.network))
    state = load_url(PRETRAINED[args.network], model_dir=os.path.join(get_data_root(), 'networks'))
    # parsing net params from meta
    # architecture, pooling, mean, std required
    # the rest has default values, in case that is doesnt exist
    net_params = {}
    net_params['architecture'] = state['meta']['architecture']
    net_params['pooling'] = state['meta']['pooling']
    net_params['local_whitening'] = state['meta'].get('local_whitening', False)
    net_params['regional'] = state['meta'].get('regional', False)
    net_params['whitening'] = state['meta'].get('whitening', False)
    net_params['mean'] = state['meta']['mean']
    net_params['std'] = state['meta']['std']
    net_params['pretrained'] = False
    # network initialization
    net = init_network(net_params)
    net.load_state_dict(state['state_dict'])

    print(">>>> loaded network: ")
    print(net.meta_repr())

    # setting up the multi-scale parameters
    ms = list(eval(args.multiscale))
    print(">>>> Evaluating scales: {}".format(ms))

    # moving network to gpu and eval mode
    net.cuda()
    net.eval()

    # set up the transform
    normalize = transforms.Normalize(
        mean=net.meta['mean'],
        std=net.meta['std']
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # evaluate on test datasets
    start = time.time()

    print('>> {}: Extracting...')

    img_path = osp.join(args.data_path, "silda-images")
    fnames_db = []
    with open(osp.join(args.data_path, "silda-train-poses.txt"), "r") as f:
        for line in f:
            fnames_db.append(osp.join(img_path, line.strip().split(" ")[0]))

    fnames_q = []
    with open(osp.join(args.data_path, "query_imgs.txt"), "r") as f:
        for line in f:
            fnames_q.append(osp.join(img_path, line.strip().split(" ")[0]))

    # extract vectors
    db_vec_fname = osp.join(args.data_path, "db_feat_rad.pkl")
    q_vec_fname = osp.join(args.data_path, "q_feat_rad.pkl")
    vecs_db, vecs_q = None, None
    if osp.isfile(db_vec_fname):
        with open(db_vec_fname, "rb") as f:
            vecs_db = pickle.load(f)
    else:
        vecs_db = extract_vectors(net, fnames_db, args.image_size, transform, ms=ms)
        with open(db_vec_fname, "wb") as f:
            pickle.dump(vecs_db, f)

    if osp.isfile(q_vec_fname):
        with open(q_vec_fname, "rb") as f:
            vecs_q = pickle.load(f)
    else:
        vecs_q = extract_vectors(net, fnames_q, args.image_size, transform, ms=ms)
        with open(q_vec_fname, "wb") as f:
            pickle.dump(vecs_db, f)

    print('>> {}: Evaluating...')

    # convert to numpy
    vecs_db = vecs_db.numpy()
    vecs_q = vecs_q.numpy()

    # search, rank, and print
    scores_db = np.dot(vecs_db.T, vecs_db)
    scores_q = np.dot(vecs_db.T, vecs_q)
    ranks_db = np.argsort(-scores_db, axis=0)
    ranks_q = np.argsort(-scores_q, axis=0)

    print(ranks_q.shape, ranks_db.shape)

    images_db = [fname_db[len(img_path) + 1:] for fname_db in fnames_db]
    images_q = [fname_q[len(img_path) + 1:] for fname_q in fnames_q]
    pairs_db = []
    for q_id in range(len(images_db)):
        img_q = images_db[q_id]
        pairs_per_q = [" ".join([img_q, images_db[db_id]]) for db_id in list(ranks_db[1:args.top_n + 1, q_id])]
        pairs_db += pairs_per_q

    for q_id in range(len(images_q)):
        img_q = images_q[q_id]
        pairs_per_q = [" ".join([img_q, images_db[db_id]]) for db_id in list(ranks_q[:args.top_n, q_id])]
        pairs_db += pairs_per_q

    with open(osp.join(args.data_path, "image_pairs_silda_top_" + str(args.top_n) + ".txt"), "w") as f:
        for pair in pairs_db:
            f.write(pair + "\n")

    print("Done")


if __name__ == '__main__':
    main()
