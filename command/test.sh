CUDA_VISIBLE_DEVICES=0,1 python tools/test.py configs/cascade_rcnn_aa_abrm_dcn_x101_32x4d_fpn_1x.py work_dirs/cascade_rcnn_aa_abrm_dcn_x101_32x4d_fpn_1x/epoch_100.pth --out ./result/result.pkl --eval bbox;