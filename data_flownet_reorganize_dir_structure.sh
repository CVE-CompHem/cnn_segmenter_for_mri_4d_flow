#!/bin/bash


# copy fully sampled images
for N in 1 2 3 4 5 6 7
do

    srcdir="../../data/v1/decrypt/flownet/hpc_predict/v2/inference/"
    srcdir+=$1 # such as 2021-02-11_19-41-32_daint102
    srcdir+="_volN"
    srcdir+=$N
    srcdir+="/output/recon_volN"
    srcdir+=$N
    srcdir+="_vn.mat.h5"
    
    dstdir="../../data/v1/decrypt/segmenter/segmenter_data/v"
    dstdir+=$N
    dstdir+="_R1.h5"
    
    echo $srcdir
    echo $dstdir
    
    cp $srcdir $dstdir
    
done

# copy under sampled reconstructed images
for N in 1 2 3 4 5 6 7
do

    for R in 8 10 12 14 16 18 20 22
    do
    
        srcdir="../../data/v1/decrypt/flownet/hpc_predict/v2/inference/"
	srcdir+=$2 #such as 2021-03-19_15-46-05_daint102
	srcdir+="_volN"
        srcdir+=$N
        srcdir+="_R"
        srcdir+=$R
        srcdir+="/output/kspc_R"
        srcdir+=$R
        srcdir+="_volN"
        srcdir+=$N
        srcdir+="_vn.mat.h5"
        
        dstdir="../../data/v1/decrypt/segmenter/segmenter_data/v"
        dstdir+=$N
        dstdir+="_R"
        dstdir+=$R
        dstdir+=".h5"
        
        echo $srcdir
        echo $dstdir
        
        cp $srcdir $dstdir

    done    
done

# copy random walker segmentations
for N in 1 2 3 4 5 6 7
do

    srcdir="../../data/v1/decrypt/segmenter/random_walker_segmenter/"
    srcdir+=$3 #such as 2021-02-11_20-14-44_daint102
    srcdir+="_volN"
    srcdir+=$N
    srcdir+="/output/recon_volN"
    srcdir+=$N
    srcdir+="_vn_seg_rw.h5"
    
    dstdir="../../data/v1/decrypt/segmenter/segmenter_data/v"
    dstdir+=$N
    dstdir+="_seg_rw.h5"
    
    echo $srcdir
    echo $dstdir
    
    cp $srcdir $dstdir
    
done
