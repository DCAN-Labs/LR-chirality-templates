"""
Creates a left/right mask from a segmentation file

Usage:
  create_chirality_mask <nifti_input_file_path> <nifti_output_file_path>
  create_chirality_mask -h | --help

Options:
  -h --help     Show this screen.
"""
import os

from docopt import docopt
import nibabel as nib
import shutil
import numpy as np
from nipype.interfaces import fsl

UNKNOWN = 0
LEFT = 1
RIGHT = 2
BILATERAL = 3

def get_id_to_region_mapping(mapping_file_name, separator=None):
    file = open(mapping_file_name, 'r')
    lines = file.readlines()

    id_to_region = {}
    for line in lines:
        line = line.strip()
        if line.startswith('#') or line == '':
            continue
        if separator:
            parts = line.split(separator)
        else:
            parts = line.split()
        region_id = int(parts[0])
        region = parts[1]
        id_to_region[region_id] = region
    return id_to_region
   
def create_initial_mask(nifti_input_file_path, nifti_output_file_path):
    segment_lookup_table='src/FreeSurferColorLUT.txt'
    img = nib.load(nifti_input_file_path)
    data = img.get_data()
    data_shape = img.header.get_data_shape()
    width = data_shape[0]
    height = data_shape[1]
    depth = data_shape[2]
    chirality_mask = data.copy()
    free_surfer_label_to_region = get_id_to_region_mapping(segment_lookup_table)
    for i in range(width):
        for j in range(height):
            for k in range(depth):
                region_id = data[i][j][k]
                if region_id == 0:
                    continue
                region_name = free_surfer_label_to_region[region_id]
                if region_name.startswith('Left-'):
                    chirality_mask[i][j][k] = LEFT
                elif region_name.startswith('Right-'):
                    chirality_mask[i][j][k] = RIGHT
                else:
                    chirality_mask[i][j][k] = BILATERAL
    mask_img = nib.Nifti1Image(chirality_mask, img.affine, img.header)
    nib.save(mask_img, nifti_output_file_path)

def fillh_lr_mask(nifti_output_file_path):
    wd=os.path.join(os.path.dirname(nifti_output_file_path), 'wd')

    if not os.path.exists(wd):
        os.mkdir(wd)

    anatfile = nifti_output_file_path
    maths = fsl.ImageMaths(in_file=anatfile, op_string='-thr 1 -uthr 1',
                           out_file='{}/Lmask.nii.gz'.format(wd))
    maths.run()

    maths = fsl.ImageMaths(in_file=anatfile, op_string='-thr 2 -uthr 2',
                           out_file='{}/Rmask.nii.gz'.format(wd))
    maths.run()

    # dilate, fill, and erode each mask in order to get rid of holes (also binarize L image in order to perform binary operations)
    anatfile = '{}/Lmask.nii.gz'.format(wd)
    maths = fsl.ImageMaths(in_file=anatfile, op_string='-dilM -dilM -dilM -fillh -ero -ero -ero',
                           out_file='{}/L_mask_holes_filled.nii.gz'.format(wd))
    maths.run()

    anatfile = '{}/Rmask.nii.gz'.format(wd)
    maths = fsl.ImageMaths(in_file=anatfile, op_string='-bin -dilM -dilM -dilM -fillh -ero -ero -ero',
                           out_file='{}/R_mask_holes_filled.nii.gz'.format(wd))
    maths.run()

    # Reassign value of 2 to R mask (L mask already a value of 1)
    anatfile = '{}/R_mask_holes_filled.nii.gz'.format(wd)
    maths = fsl.ImageMaths(in_file=anatfile, op_string='-mul 2',
                           out_file='{}/R_mask_holes_filled_label2.nii.gz'.format(wd))
    maths.run()

    # recombine new L & R mask files
    anatfile_left = '{}/L_mask_holes_filled.nii.gz'.format(wd)
    anatfile_right = '{}/R_mask_holes_filled_label2.nii.gz'.format(wd)
    maths = fsl.ImageMaths(in_file=anatfile_left, op_string='-add {}'.format(anatfile_right),
                           out_file='{}/recombined_mask_LR.nii.gz'.format(wd))
    maths.run()

    ## Fix incorrect values resulting from recombining dilated components
    orig_LRmask_img = nib.load(nifti_output_file_path)
    orig_LRmask_data = orig_LRmask_img.get_fdata()

    fill_LRmask_img = nib.load('{}/recombined_mask_LR.nii.gz'.format(wd))
    fill_LRmask_data = fill_LRmask_img.get_fdata()

    data_shape = fill_LRmask_img.header.get_data_shape()
    width = data_shape[0]
    height = data_shape[1]
    depth = data_shape[2]
    for i in range(width):
        for j in range(height):
            for k in range(depth):
                region_id_new = fill_LRmask_data[i][j][k]
                if region_id_new > 2:
                    fill_LRmask_data[i][j][k] = orig_LRmask_data[i][j][k]
     
    # save new numpy array as image
    empty_header = nib.Nifti1Header()
    out_img = nib.Nifti1Image(fill_LRmask_data, orig_LRmask_img.affine, empty_header)
    nib.save(out_img, nifti_output_file_path)

    #remove working directory with intermediate outputs
    shutil.rmtree(wd)

def LR_mask_from_aseg(nifti_input_file_path, nifti_output_file_path):
    create_initial_mask(nifti_input_file_path, nifti_output_file_path)
    fillh_lr_mask(nifti_output_file_path)

if __name__ == '__main__':
    args = docopt(__doc__)
    LR_mask_from_aseg(
        args['<nifti_input_file_path>'], args['<nifti_output_file_path>'])
