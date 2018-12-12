import numpy as np
from utils import preprocess_data, rotMat, process_gt_data, compute_action_model
from em import ActionMapper, _norm_angle
import os
import pdb
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--root_dir', type=str, default='./')

args = parser.parse_args()

if __name__ == '__main__':
	sensor_data = preprocess_data(os.path.join(args.root_dir, '2d_asami_data.txt'))
	gt_data = process_gt_data(os.path.join(args.root_dir, 'state_log.txt'))

	expected_action_model = ActionMapper(cmd_size=40, n_action=3)

	cmds = [d.command for d in sensor_data]
	
	gt_action_model = compute_action_model(gt_data, cmds)
	for i in range(0, 40):
		with np.printoptions(formatter={'float': '{: 10.3f}'.format}):
			print('Action {:2d} - {}'.format(i, expected_action_model.gt_mus[i]))
			print('Mean -      {}'.format(gt_action_model['mean'][i]))
			# print('Cov')
			# print('{}'.format(gt_action_model['cov'][i]))
			# print('===================')
			# print('===================')
