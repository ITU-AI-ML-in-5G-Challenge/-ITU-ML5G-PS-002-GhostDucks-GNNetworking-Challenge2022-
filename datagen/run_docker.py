import argparse
import yaml
import os


def docker_cmd(dataset_path):
    path = os.path.abspath(dataset_path)
    raw_cmd = f"docker run --rm --mount type=bind,src={path},dst=/data bnnupc/netsim:v0.1"

    print("Superuser privileges are required to run docker. Run the following command from terminal")
    raw_cmd = "sudo " + raw_cmd
    print(raw_cmd)
    return raw_cmd


def save_docker_config(dataset_path, args):
    dataset_path = os.path.abspath(dataset_path)
    if not hasattr(args, 'dataset_name'):
        args.dataset_name = os.path.basename(dataset_path)

    conf_parameters = {
        "threads": args.threads,
        "dataset_name": args.dataset_name,
        "samples_per_file": args.samples_per_file,
        "rm_prev_results": args.rm_prev_results,
    }

    conf_file = os.path.join(dataset_path, 'conf.yml')
    with open(conf_file, 'w') as fd:
        yaml.dump(conf_parameters, fd)


def main(args):
    save_docker_config(args.dataset_path)
    raw_cmd = docker_cmd(args.dataset_path)
    print('')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--threads', default=6, type=int,
                        help='Number of threads to use')
    parser.add_argument('-d', '--dataset-path', default='./training',
                        help='path to dataset root')
    parser.add_argument('-name', '--dataset-name',
                        help='Name of the dataset. by default uses directory name')
    parser.add_argument('-spf', '--samples-per-file', default=10, type=int,
                        help='Number of samples per compressed file')
    parser.add_argument('-rm', '--rm-prev-results', default='n', choices=['y', 'n'],
                        help='If "y" is selected and the results folder already exists, the folder is removed.')

    args = parser.parse_args()
    main(args)


