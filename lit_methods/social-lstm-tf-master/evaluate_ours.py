import numpy as np
import tensorflow as tf

import os
import pickle
import argparse
import ipdb

from utils_ours import SocialDataLoader
from social_model import SocialModel
from grid import getSequenceGridMask
# from social_train import getSocialGrid, getSocialTensor


def get_mean_and_final_error(predicted_traj, true_traj, observed_length, maxNumPeds):
    '''
    Function that computes the mean euclidean distance error between the
    predicted and the true trajectory
    params:
    predicted_traj : numpy matrix with the points of the predicted trajectory
    true_traj : numpy matrix with the points of the true trajectory
    observed_length : The length of trajectory observed
    '''
    # The data structure to store all errors
    error = np.zeros(len(true_traj) - observed_length)
    # For each point in the predicted part of the trajectory
    for i in range(observed_length, len(true_traj)):
        # The predicted position. This will be a maxNumPeds x 3 matrix
        pred_pos = predicted_traj[i, :]
        # The true position. This will be a maxNumPeds x 3 matrix
        true_pos = true_traj[i, :]
        timestep_error = 0
        counter = 0
        for j in range(maxNumPeds):
            if true_pos[j, 0] == 0:
                # Non-existent ped
                continue
            else:
                timestep_error += np.linalg.norm(true_pos[j, [1, 2]] - pred_pos[j, [1, 2]])
                counter += 1
        if counter != 0:
            error[i - observed_length] = timestep_error / counter

        # The euclidean distance is the error
        # error[i-observed_length] = np.linalg.norm(true_pos - pred_pos)

    # Return the mean error
    return np.mean(error), error[-1]


def main():
    parser = argparse.ArgumentParser()
    # Observed length of the trajectory parameter
    parser.add_argument('--obs_length', type=int, default=8,
                        help='Observed length of the trajectory')
    # Predicted length of the trajectory parameter
    parser.add_argument('--pred_length', type=int, default=12,
                        help='Predicted length of the trajectory')
    # Model
    parser.add_argument('--model_path', type=str, default='/retrained',
                        help='Relative path (w.r.t. save directory) of model to test with')
    # Test dataset
    parser.add_argument('--test_dataset', type=int, default=range(6),
                        help='Dataset to be tested on')
    parser.add_argument('--dtype', type=str, default='test',
                        help='Part of dataset to be tested on (train/val/test)')

    # Parse the parameters
    sample_args = parser.parse_args()

    # Define the path for the config file for saved args
    with open(os.path.join('save'+sample_args.model_path, 'social_config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)

    # Create a SocialModel object with the saved_args and infer set to true
    model = SocialModel(saved_args, True)
    # Initialize a TensorFlow session
    sess = tf.InteractiveSession()
    # Initialize a saver
    saver = tf.train.Saver()

    # Get the checkpoint state for the model
    ckpt = tf.train.get_checkpoint_state('save'+sample_args.model_path)
    print ('loading model: ', ckpt.model_checkpoint_path)

    # Restore the model at the checkpoint
    saver.restore(sess, ckpt.model_checkpoint_path)

    # Dataset to get data from
    dset = 1
    dataset = sample_args.test_dataset

    # Create a SocialDataLoader object with batch_size 1 and seq_length equal to observed_length + pred_length
    data_loader = SocialDataLoader(sample_args.dtype, 1, sample_args.pred_length + sample_args.obs_length, saved_args.maxNumPeds, dset, dataset, True)

    # Reset all pointers of the data_loader
    data_loader.reset_batch_pointer()

    # Variable to maintain total error
    # mean_error = np.empty(data_loader.num_batches)
    # final_error = np.empty(data_loader.num_batches)
    mean_error = 0
    final_error = 0
    # error = np.empty((data_loader.num_batches, sample_args.pred_length))
    # eidx = 0
    # For each batch
    for b in range(data_loader.num_batches):
        # Get the source, target and dataset data for the next batch
        x, y, d = data_loader.next_batch()

        # Batch size is 1
        x_batch, y_batch, d_batch = x[0], y[0], d[0]

        dimensions = [1280, 720]

        grid_batch = getSequenceGridMask(x_batch, dimensions, saved_args.neighborhood_size, saved_args.grid_size)

        obs_traj = x_batch[:sample_args.obs_length]
        obs_grid = grid_batch[:sample_args.obs_length]
        # obs_traj is an array of shape obs_length x maxNumPeds x 3

        complete_traj = model.sample(sess, obs_traj, obs_grid, dimensions, x_batch, sample_args.pred_length)

        # ipdb.set_trace()
        # complete_traj is an array of shape (obs_length+pred_length) x maxNumPeds x 3
        # error[eidx, :] = get_mean_and_final_error(complete_traj, x[0], sample_args.obs_length, saved_args.maxNumPeds)
        # eidx += 1
        total_error = get_mean_and_final_error(complete_traj, x[0], sample_args.obs_length, saved_args.maxNumPeds)
        mean_error += total_error[0]
        final_error += total_error[1]

        print "Processed trajectory number : ", b, "out of ", data_loader.num_batches, " trajectories"

    # Print the mean error across all the batches
    # error_per_pred = np.mean(error, axis=0)
    # print "Error of the model per timestep are\n", error_per_pred
    # print "Mean error of the model is ", np.mean(error_per_pred)
    # print "Final error of the model is ", error_per_pred[-1]
    print "Mean error of the model is ", mean_error/data_loader.num_batches
    print "Final error of the model is ", final_error/data_loader.num_batches

if __name__ == '__main__':
    main()
