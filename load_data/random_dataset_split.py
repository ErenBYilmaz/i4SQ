import random


def random_dataset_split(whole_dataset, first_part_fraction) -> tuple:
    """split the given dataset randomly into two parts, where the first part, has the specified fraction of samples"""
    num_first_samples = round(first_part_fraction * len(whole_dataset))
    if num_first_samples > len(whole_dataset):
        raise ValueError("training can not be set larger than dataset")
    if num_first_samples == len(whole_dataset):
        return whole_dataset, []
    if num_first_samples == 0:
        return [], whole_dataset

    second_part_indices = list(range(len(whole_dataset)))
    first_part_indices = []
    for _ in range(num_first_samples):
        i = random.randrange(len(second_part_indices))
        first_part_indices.append(second_part_indices[i])
        del second_part_indices[i]
    for sample in first_part_indices:
        assert sample not in second_part_indices
    first_part = [whole_dataset[idx] for idx in first_part_indices]
    second_part = [whole_dataset[idx] for idx in second_part_indices]
    for sample in first_part:
        if sample in second_part:
            print('WARNING: sample occurs in both parts.')
            print('  ' + str(sample))
    return first_part, second_part