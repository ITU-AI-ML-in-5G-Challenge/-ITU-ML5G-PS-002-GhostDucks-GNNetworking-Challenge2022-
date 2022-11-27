import argparse
from custom_train import main


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-path', help='files with samples list', required=True)
    parser.add_argument('-t', '--test-data-path', default='/mnt/ext/shared/Projects/GNNetworkingChallenge/validation_dataset_pkl')
    parser.add_argument('-n', '--name', default='exp')
    parser.add_argument('-v', '--val-steps', type=int, default=130)
    args = parser.parse_args()

    args.task_name = args.name
    args.save_path = '.'
    args.sample_train_mode = 'file'
    args.sample_val = args.data_path
    args.data_dir = '/mnt/ext/shared/Projects/GNNetworkingChallenge/generated_datasets_pkl'
    args.save_best_only = False
    args.load_from_ckpt = True
    args.ckpt_weights_dir = '/mnt/ext/shared/Projects/GNNetworkingChallenge/RouteNet_Fermi/initial_weights/initial_weights'
    main(args, args.data_dir, final_evaluation=False, val_steps_during_train=args.val_steps,
         epochs=20, steps_per_epoch=2000, check_size=False, test_path=args.test_data_path, log_sample_loss=True,
         shuffle_train=False)
