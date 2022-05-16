#!/usr/bin/env python3

import random


# Parametrization for the generation of test data:
MAX_STR_CNT = 25  # max number of characters in the smaller string
MAX_STR_DELTA_CNT = 5  # max difference between the number of characters in the compared strings
MAX_ALTERATIONS_CNT = 2  # max number of random alterations


def generate_test_data(fpath, count):
	"""
	Generates a CSV file :fpath: with :count: number of pairs of random strings to compare.
	:param str fpath: path of the CSV file to store the examples in
	:param int count: number of pairs of strings to generate
	:returns: None
	"""

	pairs = []
	for i in range(count):
		s1_len = random.randint(1, MAX_STR_CNT)
		s1_s2_delta_len = random.randint(0, MAX_STR_DELTA_CNT)
		diffs = random.randint(0, MAX_ALTERATIONS_CNT)
		s1 = ''.join(chr(ord('a') + i) for i in range(s1_len))
		extension = ''.join(chr(ord('a') + i) for i in range(s1_s2_delta_len))
		if 1 == random.choice((0, 1)):
			s2 = s1 + extension
		else:
			s2 = extension + s1

		for _ in range(diffs):
			alter_idx = random.randint(0, len(s2))
			s2 = s2[:alter_idx] + str(alter_idx % 10) + s2[alter_idx + 1:]

		pairs.append((s1, s2))

	with open(fpath, "wt") as fh:
		fh.write('\n'.join((f'{s1}\t{s2}' for (s1, s2) in pairs)))

	print(f'Test data generated\n\tpath: {fpath}\n\texamples: {count}')


if __name__ == "__main__":
	generate_test_data('levenshtein_test_data.csv', 1000)
