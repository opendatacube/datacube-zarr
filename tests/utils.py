'''Helper functions for the unit tests.'''
from json import load

from zarr_io.zarr_io import ZarrIO


def _save_dataarray(data, uri, name, chunks=None):
    '''Save DataArray to storage.'''
    zio = ZarrIO()
    zio.save_dataarray(uri=uri,
                       dataarray=data.copy(),
                       name=name,
                       chunks=chunks)


def _save_dataset(data, uri, name, chunks=None):
    '''Save Dataset to storage.'''
    zio = ZarrIO()
    zio.save_dataset(uri=uri,
                     dataset=data.copy().to_dataset(name=name),
                     chunks=chunks)


def _load_dataset(uri):
    '''Load Dataset from storage.'''
    zio = ZarrIO()
    return zio.load_dataset(uri=uri)


def _check_zarr_files(data, root, group_name, name, chunks):
    '''Check zarr files in local filesystem.

    Only some metadata and chunk file names are checked, not actual binary content.'''
    assert root.exists(), f'Missing {root} after save'
    # Check chunks in root level metadata
    metadata_path = root / '.zmetadata'
    assert metadata_path.exists(), f'Missing .zmetadata in {root}'
    with metadata_path.open() as fh:
        metadata = load(fh)
    assert metadata['metadata'][f'{group_name}/{name}/.zarray']['chunks'] == \
        chunks['output'], 'Chunks not as set'

    dataset_dir = root / group_name
    assert dataset_dir.exists(), f'Missing {group_name}/ in {root}'
    array_dir = dataset_dir / name
    assert array_dir.exists(), f'Missing {name}/ in {dataset_dir}'

    # Check chunks in array level metadata
    metadata_path = array_dir / '.zarray'
    assert metadata_path.exists(), f'Missing .zarray in {array_dir}'
    with metadata_path.open() as fh:
        metadata = load(fh)
    assert metadata['chunks'] == chunks['output'], 'Chunks not as set'
    assert metadata['shape'] == list(data.shape), 'Data shape not as set'

    # Check chunk files
    chunk_files = sorted([path.name for path in array_dir.glob('?.?')])
    expected_chunk_files = sorted([f'{i}.{j}'
                                   for i in range(chunks['chunks_per_side'])
                                   for j in range(chunks['chunks_per_side'])])
    assert chunk_files == expected_chunk_files, 'Unexpected chunk files'
