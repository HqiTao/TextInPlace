python -W ignore train.py --epochs_num 10 --train_batch_size 64 --infer_batch_size 64 \
  --backbone ResNet50 --aggregation boq --features_dim 16384 \
  --lr 1e-4 --use_amp16 \
  --config-file configs/Bridge/TotalText/R_50_poly.yaml \
  --opts MODEL.WEIGHTS checkpoints/pretrain/BTS/Bridge_tt.pth

python -W ignore train.py train.py --epochs_num 10 --train_batch_size 64 --infer_batch_size 64 \
  --backbone ResNet50 --aggregation boq --features_dim 16384 \
  --lr 5e-5 --use_amp16 --use_indoor_datasets \
  --config-file configs/Bridge/TotalText/R_50_poly.yaml \
  --resume checkpoints/best_model.pth
