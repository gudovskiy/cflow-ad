from __future__ import print_function
import argparse, os
import numpy as np

RESULT_DIR = './results'


def get_args():
    parser = argparse.ArgumentParser(description='CFLOW-AD')
    parser.add_argument('--dataset', default='mvtec', type=str, metavar='D',
                        help='dataset name: mvtec/stc/video (default: mvtec)')
    parser.add_argument('-enc', '--enc-arch', default='wide_resnet50_2', type=str, metavar='A',
                        help='feature extractor: wide_resnet50_2/resnet18/mobilenet_v3_large (default: wide_resnet50_2)')
    parser.add_argument('-dec', '--dec-arch', default='freia-cflow', type=str, metavar='A',
                        help='normalizing flow model (default: freia-cflow)')
    parser.add_argument('-pl', '--pool-layers', default=3, type=int, metavar='L',
                        help='number of layers used in NF model (default: 3)')
    parser.add_argument('-cb', '--coupling-blocks', default=8, type=int, metavar='L',
                        help='number of layers used in NF model (default: 8)')
    parser.add_argument('-runs', '--run-count', default=4, type=int, metavar='C',
                        help='number of runs (default: 4)')
    parser.add_argument('-inp', '--input-size', default=512, type=int, metavar='C',
                        help='image resize dimensions (default: 512)')

    args = parser.parse_args()
    
    return args

def main(c):
    runs = c.run_count
    if c.dataset == 'mvtec':
        class_names = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
                    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                    'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    elif c.dataset == 'stc':
        class_names = ['01', '02', '03', '04', '05', '06', 
                    '07', '08', '09', '10', '11', '12'] #, '13']
    else:
        raise NotImplementedError('{} is not supported dataset!'.format(c.dataset))
    #
    metrics = ['DET_AUROC', 'SEG_AUROC', 'SEG_AUPRO']
    results = np.zeros((len(metrics), len(class_names), runs))
    # loop
    result_files = os.listdir(RESULT_DIR)
    for class_idx, class_name in enumerate(class_names):
        for run in range(runs):
            if class_name in ['cable', 'capsule', 'hazelnut', 'metal_nut', 'pill']:  # AUROC
                input_size = 256
            elif class_name == 'transistor':
                input_size = 128
            else:
                input_size = c.input_size
            
            #input_size = c.input_size

            c.model = "{}_{}_{}_pl{}_cb{}_inp{}_run{}_{}".format(
                c.dataset, c.enc_arch, c.dec_arch, c.pool_layers, c.coupling_blocks, input_size, run, class_name)
            #
            result_file = list(filter(lambda x: x.startswith(c.model), result_files))
            if len(result_file) == 0:
                raise NotImplementedError('{} results are not found!'.format(c.model))
            elif len(result_file) > 1:
                raise NotImplementedError('{} duplicate results are found!'.format(result_file))
            else:
                result_file = result_file[0]
            #
            fp = open(os.path.join(RESULT_DIR, result_file), 'r')
            lines = fp.readlines()
            rline = lines[0].split(' ')[0].split(',')
            result = np.array([float(r) for r in rline])
            fp.close()
            results[:, class_idx, run] = result
    #
    for i, m in enumerate(metrics):
        print('\n{}:'.format(m))
        for j, class_name in enumerate(class_names):
            print(r"{:.2f}\tiny$\pm${:.2f} for class {}".format(np.mean(results[i, j]), np.std(results[i, j]), class_name))
        # \tiny$\pm$
        means = np.mean(results[i], 0)
        #print(results[i].shape, means.shape)
        print(r"{:.2f} for average".format(np.mean(means)))


if __name__ == '__main__':
    c = get_args()
    main(c)

