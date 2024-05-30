import copy
import warnings
from abc import ABC, abstractmethod
from typing import Tuple, List, Union, Set, Optional

import cachetools
import numpy

from lib.tuned_cache import TunedMemory


class LabeledCoordinate:
    def __init__(self,
                 position_px: numpy.ndarray,
                 label: int,
                 root_spatial_influence: Union[float, numpy.ndarray]):
        self.position_px = position_px
        self.label = label
        self.root_spatial_influence = root_spatial_influence


class LabelMapGeneratorND(ABC):
    label_map_disk_cache = TunedMemory('.cache/label_map_cache', verbose=0)
    label_map_ram_cache = cachetools.LRUCache(maxsize=159)

    def __init__(self, label_ids: Set[int]):
        self.label_ids = label_ids
        self.generate_with_disk_cache = self.label_map_disk_cache.cache(self.generate_with_disk_cache)

    @abstractmethod
    def generate(self, map_shape: Tuple[int, ...], coordinates: List[LabeledCoordinate]):
        pass

    def generate_with_disk_cache(self, *args, **kwargs):
        self_before = self.ordered_attributes
        result = self.generate(*args, **kwargs)
        assert self.ordered_attributes == self_before  # otherwise the cache will not work
        return result

    def ordered_attributes(self):
        # noinspection PyRedundantParentheses
        return (self.label_ids,)

    @cachetools.cached(cache=label_map_ram_cache, key=lambda self, map_shape, coordinates: (type(self).__name__,
                                                                                            self.ordered_attributes,
                                                                                            map_shape,
                                                                                            tuple(coordinates)))
    def generate_with_ram_cache(self, *args, **kwargs):
        self_before = self.ordered_attributes
        result = self.generate_with_disk_cache(*args, **kwargs)
        assert self.ordered_attributes == self_before  # otherwise the cache will not work
        return result


class Glocker(LabelMapGeneratorND):
    def __init__(self, label_ids: Set[int],
                 hard_labels=True,
                 background_label: Optional[int] = 0,
                 return_sparse=True):
        label_ids = set(label_ids)
        super().__init__(label_ids)
        self.background_label = background_label
        if background_label is not None:
            assert background_label in label_ids
        self.return_sparse = return_sparse
        self.hard_labels = hard_labels
        if label_ids != set(range(min(label_ids), max(label_ids) + 1)):
            raise NotImplementedError

    def use_background(self):
        return self.background_label is not None

    def generate(self, map_shape: Tuple[int, ...], coordinates: List[LabeledCoordinate]):
        coordinates = copy.deepcopy(coordinates)
        coords = numpy.array(numpy.meshgrid(*[range(s) for s in map_shape], indexing='ij'))

        for c in coordinates:
            if len(map_shape) != len(c.position_px):
                raise ValueError
            for p, s in zip(c.position_px, map_shape):
                if p < 0 or p > s - 1:
                    warnings.warn('Coordinates out of bounds. This may or may not be intentional.')
        # coords now has shape (len(map_shape), *map_shape)
        assert coords.shape == (len(map_shape), *map_shape)

        for c in coordinates:
            if c.label not in self.label_ids:
                raise ValueError
            if c.label == self.background_label:
                raise NotImplementedError('Setting a points label explicitly to background is not supported yet.')

        for c in coordinates:
            c.position_px = numpy.array(c.position_px)
            for _ in range(len(map_shape)):
                c.position_px = numpy.expand_dims(c.position_px, axis=-1)
        label_likelihoods = {label: numpy.sum((numpy.exp(-numpy.sum(((c.position_px - coords) / c.root_spatial_influence) ** 2, axis=0))
                                               for c in coordinates
                                               if c.label == label))
                             for label in self.label_ids}

        ll_array = numpy.stack([label_likelihoods[label]
                                if not numpy.isscalar(label_likelihoods[label])
                                else numpy.full(map_shape, fill_value=label_likelihoods[label])
                                for label in sorted(self.label_ids)],
                               axis=0)
        if self.use_background():
            ll_array[self.background_label] = -numpy.inf
            ll_array[self.background_label] = 1 - ll_array.max(axis=0)
        if self.hard_labels:
            result = numpy.array(sorted(self.label_ids), dtype='int8')[numpy.argmax(ll_array, axis=0)]
        else:
            raise NotImplementedError

        # too expensive
        # assert numpy.isin(result, [0,1,2]).all()

        if not self.return_sparse:
            num_classes = len(self.label_ids)
            result = numpy.identity(num_classes, dtype='int8')[result]
            assert result.shape[-1] == len(self.label_ids)

            # too expensive
            # assert (numpy.sum(result, axis=-1) == 1).all()
        else:
            assert result.shape[-1] == 1
        return result

    def ordered_attributes(self):
        return super(Glocker, self).ordered_attributes() + [self.hard_labels, self.background_label, self.return_sparse]


def main():
    map_shape = (4, 5)
    coordinates = [
        # {'position_px': (1, 1), 'label': 1, 'root_spatial_influence': numpy.array([[[1]],[[1]]])},
        {'position_px': (2, 4), 'label': 2, 'root_spatial_influence': numpy.array([[[1]], [[2]]])},
        # {'position_px': (1, 2), 'label': 2, 'root_spatial_influence': numpy.array([[[1]],[[1]]])},
        {'position_px': (1, 2), 'label': 1, 'root_spatial_influence': 1},
    ]
    data = (Glocker({0, 1, 2}, background_label=0, return_sparse=False).generate(map_shape, [LabeledCoordinate(**c) for c in coordinates]))
    print(data)
    print(data.shape)
    print(data.nbytes)


if __name__ == '__main__':
    main()
