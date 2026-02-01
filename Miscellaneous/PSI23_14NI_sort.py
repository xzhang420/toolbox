# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 21:42:25 2020

@author: boillat
"""

def sort_files(list_fname, src_dir, dst_dir, hdf_prefix, base_dir=''):
    
    import pandas as pd
    from img_utils import get_img, write_img
    from tqdm.notebook import tqdm
    import os
    
    flist = pd.read_csv(os.path.join(base_dir, list_fname), sep=';')
    
    nlines = len(flist['name'])
    
    for i in tqdm(range(nlines)):
        
        name = flist['name'][i]
        istart = int(flist['start'][i])
        iend = int(flist['end'][i])
        types = flist['types'][i].split(',')
        ilist_str = str(flist['ignore'][i])
        if ilist_str == '' or ilist_str == 'nan':
            ignore_list = []
        else:
            ignore_list = [int(float(x)) for x in ilist_str.split(',')]
        
        ntypes = len(types)
        
        i_img = [0 for x in types]
        
        nfiles = iend-istart+1
        
        for j in tqdm(range(nfiles)):
            
            src_fname = os.path.join(src_dir, hdf_prefix + '{:06d}'.format(istart+j) + '.hdf')
            
            stack = get_img(src_fname, base_dir)
            
            if len(stack.shape) == 2:
                
                itype = 0
            
                dst_fbase = os.path.join(dst_dir, name, types[itype], (name.split('\\'))[-1])
                    
                dst_fname = dst_fbase + '_{:05d}.fits'.format(i_img[itype])
                    
                dst_name_dir, _ = os.path.split(dst_fname)
                
                if not os.path.exists(dst_name_dir):
                    os.makedirs(os.path.join(dst_name_dir,''))
                
                write_img(stack, dst_fname, base_dir)
                
                i_img[itype] = i_img[itype] + 1 
            
            else:
                
                for k in range(stack.shape[0]):
                    
                    if k not in ignore_list:
                    
                        itype = k % ntypes
                
                        dst_fbase = os.path.join(dst_dir, name, types[itype], (name.split('\\'))[-1])
                        
                        dst_fname = dst_fbase + '_{:05d}.fits'.format(i_img[itype])
                        
                        dst_name_dir, _ = os.path.split(dst_fname)
                        
                        if not os.path.exists(dst_name_dir):
                            os.makedirs(os.path.join(dst_name_dir,''))
                    
                        write_img(stack[k,:,:], dst_fname, base_dir)
                        
                        i_img[itype] = i_img[itype] + 1 
            

