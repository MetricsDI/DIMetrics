#!/usr/bin/env python3

import itertools
import random


# Parametrization for the generation of textual test data:
MAX_STR_CNT = 25  # max number of characters in the smaller string
MAX_STR_DELTA_CNT = 5  # max difference between the number of characters in the compared strings
MAX_ALTERATIONS_CNT = 2  # max number of random alterations

# Parametrization for the generation of items test data:
ITEMS_MAX_STR_LEN = 15
ITEMS_MAX_STR_DELTA_CNT = 3
ITEMS_MAX_ITEM_DELTA_CNT = 1


def generate_textual_test_data(count):
	"""
	Generates a :count: number of pairs of random strings to compare.
	:param int count: number of pairs of strings to generate
	:returns: :count:-long list of pairs of strings.
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

	return pairs


def generate_items_test_data(item_count, itemset_count):
	"""
	Generates a predefined number (:itemset_count:) of sets of items with :item_count: elements each.
	:param item_count: number of elements in item set
	:param itemset_count: number of sets of items
	:return: list of pairs of dictionaries with items
	"""

	items = generate_textual_test_data(item_count * itemset_count)

	for itertools.cycle()


if __name__ == "__main__":
	# textual comparison
	fpath = 'str_distance_test_data.csv'
	count = 1000
	pairs = generate_textual_test_data(count)

	with open(fpath, "wt") as fh:
		fh.write('\n'.join((f'{s1}\t{s2}' for (s1, s2) in pairs)))

	print(f'Test data generated\n\tpath: {fpath}\n\texamples: {count}')


