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

datasets_names = ['Madrid_Metropolis', 'Gendarmenmarkt', 'Tower_of_London', "Roman_Forum", "Alamo", "Cornell"]

parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Testing End-to-End')

# test options
parser.add_argument('--data-path', default='/ssd/data/sfm-evaluation/data')
parser.add_argument('--network', '-n', default='gl18-tl-resnet152-gem-w')
parser.add_argument('--scenes', '-s', default='Roman_Forum,Alamo,Cornell')
parser.add_argument('--image-size', '-imsize', default=1024, type=int, metavar='N',
                    help="maximum size of longer image side used for testing (default: 1024)")
parser.add_argument('--multiscale', '-ms', metavar='MULTISCALE', default='[1, 2**(1/2), 1/2**(1/2)]',
                    help="use multiscale vectors for testing, " +
                         " examples: '[1]' | '[1, 1/2**(1/2), 1/2]' | '[1, 2**(1/2), 1/2**(1/2)]' (default: '[1]')")
parser.add_argument('--top-n', default='[20, 50, 100]')

# GPU ID
parser.add_argument('--gpu-id', '-g', default='0', metavar='N',
                    help="gpu id used for testing (default: '0')")


def main():
    args = parser.parse_args()

    # check if there are unknown datasets
    for scene in args.scenes.split(','):
        if scene not in datasets_names:
            raise ValueError('Unsupported or unknown scene: {}!'.format(scene))

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
    scenes = args.scenes.split(',')
    for scene in scenes:
        start = time.time()

        print('>> {}: Extracting...'.format(scene))
        img_path = osp.join(args.data_path, scene, "images")
        images = [osp.join(img_path, fname) for fname in os.listdir(img_path) if fname[-3:].lower() in ['jpg', 'png']]

        # extract vectors
        vecs = extract_vectors(net, images, args.image_size, transform, ms=ms)

        print('>> {}: Evaluating...'.format(scene))

        # convert to numpy
        vecs = vecs.numpy()

        # search, rank, and print
        scores = np.dot(vecs.T, vecs)
        ranks = np.argsort(-scores, axis=0)

        images = [img.split('/')[-1] for img in images]
        for top_k in list(eval(args.top_n)):
            pairs = []
            for q_id in range(len(images)):
                img_q = images[q_id]
                pairs_per_q = [" ".join([img_q, images[db_id]]) for db_id in list(ranks[1:top_k+1, q_id])]
                pairs += pairs_per_q
            with open(osp.join(args.data_path, scene, "image_pairs_" + str(top_k) + ".txt"), "w") as f:
                for pair in pairs:
                    f.write(pair + "\n")

        print('>> {}: elapsed time: {}'.format(scene, htime(time.time() - start)))


if __name__ == '__main__':
    main()
