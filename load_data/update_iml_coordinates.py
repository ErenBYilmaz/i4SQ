import copy
from typing import Dict, Any

import cachetools
import numpy
import pandas

import hiwi
from hiwi import Object
from load_data.load_image import coordinate_transformer_from_file

COORDINATE_SOURCES = ['ground_truth', 'manual', 'localizer']


class NewCoordinates(ValueError):
    pass


class MissingCoordinates(ValueError):
    pass


class CoordinateCSVNotFound(FileNotFoundError):
    pass


@cachetools.cached(cachetools.LRUCache(maxsize=2000))
def coordinate_annotations(csv_path) -> pandas.DataFrame:
    return pandas.read_csv(
        csv_path,
        sep=';',
        dtype={'studyDescription': str}
    )


@cachetools.cached(cachetools.LRUCache(maxsize=2000),
                   key=lambda study_description, coordinate_sources=None, csv_path=None: (study_description, str(coordinate_sources), csv_path))
def coord_rows_by_vertebra_name(study_description, coordinate_sources=None, csv_path=None) -> Dict[str, Any]:
    from ct_dirs import COORDINATE_CSV
    if csv_path is None:
        csv_path = COORDINATE_CSV
    try:
        coords: pandas.DataFrame = coordinate_annotations(csv_path)
    except FileNotFoundError:
        raise CoordinateCSVNotFound()
    if coordinate_sources is None:
        coordinate_sources = COORDINATE_SOURCES
    coords = coords[coords['studyDescription'] == study_description]
    if coordinate_sources == 'all':
        coordinate_sources = coords['source'].unique()
        if coordinate_sources is None:
            coordinate_sources = []
    vertebra_coordinates = {}
    for source in coordinate_sources:
        for row_idx, row in coords[coords['source'] == source].iterrows():
            name = row['vertebra']
            if name in vertebra_coordinates:
                continue  # only use first found source
            vertebra_coordinates[name] = row[['CenterX', 'CenterY', 'CenterZ']].tolist()

    return vertebra_coordinates


def update_coordinates_from_csv(iml: hiwi.ImageList, csv_path: str, coordinate_sources=None, on_new_coordinates='add', on_missing_coordinates='invalidate'):
    from ct_dirs import COORDINATE_CSV
    for img in iml:
        vertebrae_of_img = img.parts
        csv_vertebra_names = []
        for vertebra_name, coords in coord_rows_by_vertebra_name(coordinate_sources=coordinate_sources, csv_path=csv_path, study_description=img['patient_id']).items():
            csv_vertebra_names.append(vertebra_name)
            world_coords = tuple(float(c) for c in coords)
            voxel_coords = coordinate_transformer_from_file(img.path, img['patient_id']).TransformPhysicalPointToContinuousIndex(world_coords)
            try:
                vertebra = vertebrae_of_img[vertebra_name]
            except KeyError:
                if on_new_coordinates == 'add':
                    img.parts[vertebra_name] = Object()
                    img.parts[vertebra_name].position = voxel_coords
                    img.parts[vertebra_name]['not_in_the_image'] = True
                    continue
                elif on_new_coordinates == 'ignore':
                    continue
                elif on_new_coordinates == 'raise':
                    raise NewCoordinates()
                else:
                    raise ValueError(on_new_coordinates)
            if not numpy.allclose(vertebra.position, voxel_coords, rtol=0, atol=0.1):
                vertebra.position = voxel_coords
        remove_vertebrae = []
        for vertebra_name, vertebra in vertebrae_of_img.items():
            if vertebra_name not in csv_vertebra_names:
                if on_missing_coordinates == 'invalidate':
                    vertebra.position = None
                elif on_missing_coordinates == 'ignore':
                    continue
                elif on_missing_coordinates == 'remove':
                    remove_vertebrae.append(vertebra_name)
                elif on_missing_coordinates == 'raise':
                    raise MissingCoordinates()
                else:
                    raise ValueError(on_missing_coordinates)
        for v in remove_vertebrae:
            del img.parts[v]
    if csv_path != COORDINATE_CSV:
        if hasattr(iml, 'name'):
            iml.name = f'{iml.name}_{csv_path}'
        else:
            iml.name = f'{csv_path}'


def update_iml_coordinates_from_predictions(iml: hiwi.ImageList, tool_name: str, on_new_coordinates='add', on_missing_coordinates='invalidate'):
    for img in iml:
        vertebrae_of_img = img.parts
        predicted_vertebra_names = []
        for vertebra_name in img['tool_outputs'][tool_name]:
            if vertebra_name.startswith('_'):
                continue
            predicted_vertebra_names.append(vertebra_name)
            assert len(img['tool_outputs'][tool_name][vertebra_name]) == 1
            coords = img['tool_outputs'][tool_name][vertebra_name][0]['pos']
            world_coords = tuple(float(c) for c in coords)
            voxel_coords = coordinate_transformer_from_file(img.path, img['patient_id']).TransformPhysicalPointToContinuousIndex(world_coords)
            try:
                vertebra = vertebrae_of_img[vertebra_name]
            except KeyError:
                if on_new_coordinates == 'add':
                    assert vertebra_name not in img.parts
                    img.parts[vertebra_name] = Object()
                    img.parts[vertebra_name].position = voxel_coords
                    img.parts[vertebra_name]['not_in_the_image'] = True
                    continue
                elif on_new_coordinates == 'ignore':
                    continue
                elif on_new_coordinates == 'raise':
                    raise NewCoordinates()
                else:
                    raise ValueError(on_new_coordinates)
            if not numpy.allclose(vertebra.position, voxel_coords, rtol=0, atol=0.1):
                vertebra.position = voxel_coords
        remove_vertebrae = []
        for vertebra_name, vertebra in vertebrae_of_img.items():
            if vertebra_name not in predicted_vertebra_names:
                if on_missing_coordinates == 'invalidate':
                    vertebra.position = None
                    if 'world_coords' in vertebra:
                        del vertebra['world_coords']
                elif on_missing_coordinates == 'ignore':
                    continue
                elif on_missing_coordinates == 'remove':
                    remove_vertebrae.append(vertebra_name)
                elif on_missing_coordinates == 'raise':
                    raise MissingCoordinates()
                else:
                    raise ValueError(on_missing_coordinates)
        for v in remove_vertebrae:
            del img.parts[v]
    if hasattr(iml, 'name'):
        iml.name = f'{iml.name}_{tool_name}'
    else:
        iml.name = f'{tool_name}'


class UsingCSVCoordinates:
    def __init__(self,
                 iml: hiwi.ImageList,
                 csv_path: str,
                 coordinate_sources=None,
                 on_new_coordinates='add',
                 on_missing_coordinates='invalidate', ):
        self.on_missing_coordinates = on_missing_coordinates
        self.on_new_coordinates = on_new_coordinates
        self.coordinate_sources = coordinate_sources
        self.csv_path = csv_path
        self.iml = iml

    def __enter__(self):
        self.iml_before = copy.deepcopy(self.iml)
        update_coordinates_from_csv(self.iml,
                                    csv_path=self.csv_path,
                                    coordinate_sources=self.coordinate_sources,
                                    on_new_coordinates=self.on_new_coordinates,
                                    on_missing_coordinates=self.on_missing_coordinates)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.iml[:] = self.iml_before[:]
        if hasattr(self.iml_before, 'name'):
            self.iml.name = self.iml_before.name


class UsingPredictedCoordinates:
    def __init__(self,
                 iml: hiwi.ImageList,
                 tool_name: str,
                 on_new_coordinates='add',
                 on_missing_coordinates='invalidate', ):
        self.on_missing_coordinates = on_missing_coordinates
        self.on_new_coordinates = on_new_coordinates
        self.tool_name = tool_name
        self.iml = iml

    def __enter__(self):
        self.iml_before = copy.deepcopy(self.iml)
        update_iml_coordinates_from_predictions(self.iml,
                                                tool_name=self.tool_name,
                                                on_new_coordinates=self.on_new_coordinates,
                                                on_missing_coordinates=self.on_missing_coordinates)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.iml[:] = self.iml_before[:]
        if hasattr(self.iml_before, 'name'):
            self.iml.name = self.iml_before.name
