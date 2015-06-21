
import numpy as np
import scipy.io


def parse_data(patient_info):
	signals = scipy.io.loadmat(patient_info[0])
	signals = signals['val']

	patient_data = {}
	patient_data['signals'] = signals
	patient_data['alarm'] = patient_info[1]
	patient_data['label'] = patient_info[2]
	return patient_data



def extract_all_features(patient_data):
	signals = patient_data['signals']
	print signals







	out = {}
	out['A'] = 'yes'
	out['b'] = patient_data['label']
	return out
