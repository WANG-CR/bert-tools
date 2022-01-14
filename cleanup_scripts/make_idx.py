from subprocess import call
import os
import argparse
from tqdm import tqdm


def make_idx_files(tfrecord_root, idx_file_root):
    if not os.path.isdir(idx_file_root):
        os.makedirs(idx_file_root)
    for f in tqdm(os.listdir(tfrecord_root), desc='Writing idx'):
        tfrecord_path = os.path.join(tfrecord_root, f)
        idx_file_path = os.path.join(idx_file_root, f'{f}.idx')
        call(['tfrecord2idx', tfrecord_path, idx_file_path])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfrecord_train', default='./tf_records/train')
    parser.add_argument('--tfrecord_eval', default='./tf_records/eval')
    parser.add_argument('--tfrecord_output',default = './idx_files')
    args = parser.parse_args()

    print('Processing train:')
    make_idx_files(args.tfrecord_train,
                   os.path.join(args.tfrecord_output, 'train'))
    print('Processing validation:')
    make_idx_files(args.tfrecord_eval,
                   os.path.join(args.tfrecord_output, 'validation'))


if __name__ == '__main__':
    main()
