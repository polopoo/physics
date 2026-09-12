from pathlib import Path
import argparse
import hashlib
import json
import urllib.request

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from scipy.interpolate import interp1d
from skimage.filters import threshold_otsu
from skimage.measure import marching_cubes
from skimage.metrics import structural_similarity
import trimesh

ROOT = Path(__file__).resolve().parent
SOURCE = 'https://zenodo.org/records/10549081/files/atlas_56.nii.gz?download=1'
MD5 = '84ce61d27fef48cba823c5bcd05be0b1'


def prepare(source, folder, stride):
    """Export actual float-valued 2D arrays; no masks or hidden slices in inputs."""
    image = nib.load(source)
    volume = image.get_fdata(dtype=np.float32)
    folder.mkdir(parents=True, exist_ok=True)
    indices = np.unique(np.r_[np.arange(0, volume.shape[2], stride), volume.shape[2]-1])
    files = []
    for z in indices:
        name = f'slice_{z:03d}.npy'
        np.save(folder / name, volume[:, :, z])
        files.append(name)
    manifest = dict(files=files, z_positions=indices.tolist(), shape=list(volume.shape),
                    affine=image.affine.tolist(), units='atlas voxel units; physical size unknown',
                    source=SOURCE, source_md5=MD5, stride=stride,
                    provenance='2D reslices of a population ultrasound atlas, not independent clinical acquisitions')
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return volume


def reconstruct(folder, method='linear'):
    """Read ONLY the 2D files and geometry in a manifest. Known parallel planes."""
    meta = json.loads((folder / 'manifest.json').read_text())
    planes = np.stack([np.load(folder / f) for f in meta['files']], axis=2)
    z = np.asarray(meta['z_positions'])
    if len(z) < 2 or np.any(np.diff(z) <= 0):
        raise ValueError('At least two ordered distinct plane positions are required')
    if planes.shape[:2] != tuple(meta['shape'][:2]):
        raise ValueError('Inconsistent image dimensions')
    volume = interp1d(z, planes, axis=2, kind=method, bounds_error=True)(np.arange(meta['shape'][2]))
    return volume.astype(np.float32), meta


def segment(volume, threshold=None):
    smooth = ndi.gaussian_filter(volume, sigma=1.0)
    threshold = float(threshold_otsu(smooth)) if threshold is None else float(threshold)
    mask = smooth > threshold
    mask = ndi.binary_closing(mask, iterations=2)
    labels, count = ndi.label(mask)
    if count == 0:
        raise ValueError('No foreground at this threshold')
    sizes = np.bincount(labels.ravel()); sizes[0] = 0
    mask = ndi.binary_fill_holes(labels == sizes.argmax())
    return mask, threshold


def mesh_from_mask(mask):
    field = ndi.gaussian_filter(np.pad(mask.astype(np.float32), 2), sigma=0.7)
    vertices, faces, _, _ = marching_cubes(field, level=0.5, allow_degenerate=False)
    mesh = trimesh.Trimesh(vertices=vertices-2, faces=faces, process=True)
    mesh.fix_normals()
    return mesh


def compare(reference, reconstructed, meta):
    held = np.setdiff1d(np.arange(reference.shape[2]), meta['z_positions'])
    data_range = float(reference.max()-reference.min())
    roi = reference > threshold_otsu(reference)
    held_roi = roi[:, :, held]
    delta = reconstructed[:, :, held]-reference[:, :, held]
    ref_mask, _ = segment(reference)
    pred_mask, _ = segment(reconstructed)
    return dict(held_out_slices=len(held),
                rmse_all=float(np.sqrt(np.mean(delta**2))),
                rmse_foreground=float(np.sqrt(np.mean(delta[held_roi]**2))),
                ssim_heldout_mean=float(np.mean([structural_similarity(reference[:,:,i], reconstructed[:,:,i], data_range=data_range) for i in held])),
                envelope_dice_vs_algorithm_on_full_atlas=float(2*np.sum(ref_mask & pred_mask)/(ref_mask.sum()+pred_mask.sum())),
                measured_plane_max_abs_error=float(np.max(np.abs(reference[:,:,meta['z_positions']]-reconstructed[:,:,meta['z_positions']]))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--input', type=Path, help='Existing folder with 2D .npy files and manifest.json')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs')
    args = parser.parse_args()
    if not 2 <= args.stride <= 64:
        parser.error('stride must be between 2 and 64')
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    source = ROOT/'data'/'atlas_56.nii.gz'
    reference = None
    if args.input is None:
        source.parent.mkdir(parents=True, exist_ok=True)
        if not source.exists():
            urllib.request.urlretrieve(SOURCE, source)
        if hashlib.md5(source.read_bytes()).hexdigest() != MD5:
            raise ValueError('Source checksum mismatch')
        folder = ROOT/'data'/f'slices_step{args.stride}'
        reference = prepare(source, folder, args.stride)
    else:
        folder = args.input
    volume, meta = reconstruct(folder)
    mask, threshold = segment(volume)
    mesh = mesh_from_mask(mask)
    mesh.export(out/'embryonic_brain_envelope.stl')
    reread = trimesh.load_mesh(out/'embryonic_brain_envelope.stl')
    assert reread.is_watertight and reread.is_winding_consistent and reread.volume > 0
    assert len(reread.split()) == 1
    for name, array in [('reconstructed_volume', volume), ('envelope_mask', mask.astype(np.uint8))]:
        nib.save(nib.Nifti1Image(array, np.asarray(meta['affine'])), out/f'{name}.nii.gz')
    stats = dict(input_planes=len(meta['files']), shape=list(volume.shape), threshold=threshold,
                 vertices=len(reread.vertices), faces=len(reread.faces), watertight=bool(reread.is_watertight),
                 winding_consistent=bool(reread.is_winding_consistent), connected_components=1,
                 bounds=reread.bounds.tolist(), dimensions_atlas_units=reread.extents.tolist(),
                 volume_atlas_units_cubed=float(reread.volume), physical_scale='unknown; STL is unitless')
    if reference is not None:
        stats['linear'] = compare(reference, volume, meta)
        nearest, _ = reconstruct(folder, 'nearest')
        stats['nearest_baseline'] = compare(reference, nearest, meta)
        variants = []
        for factor in [0.85, 1.0, 1.15]:
            alternative, _ = segment(volume, threshold*factor)
            variants.append(dict(threshold_factor=factor, foreground_voxels=int(alternative.sum())))
        stats['threshold_sensitivity'] = variants
    (out/'metrics.json').write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
