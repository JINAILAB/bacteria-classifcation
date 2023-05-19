# python3 main.py --model swinv2_s --data-path ./culture/cul46 --batch-size 32 --epochs 80 --lr 0.001 --opt sgd
# python3 main.py --model resnet18 --data-path ./culture/cul46 --batch-size 32 --epochs 80 --lr 0.001 --opt sgd
# python3 main.py --model swinv2_t --data-path ./culture/cul46 --batch-size 32 --epochs 80 --lr 0.001 --opt sgd
# python3 main.py --model regnet_16gf --data-path ./culture/cul46 --batch-size 32 --epochs 80 --lr 0.001 --opt sgd
# python3 main.py --model effnetv2_s --data-path ./culture/cul46 --batch-size 32 --epochs 80 --lr 0.001 --opt sgd

#python3 main.py --model swinv2_s --data-path ./dacon --batch-size 32 --epochs 100 --lr 0.001 --opt sgd --n-mean 0.485,0.456,0.406 --n-std 0.229,0.224,0.225
#python3 main.py --model resnet18 --data-path ./dacon --batch-size 32 --epochs 120 --lr 0.001 --opt sgd --n-mean 0.485,0.456,0.406 --n-std 0.229,0.224,0.225
#python3 main.py --model swinv2_t --data-path ./dacon --batch-size 16 --epochs 5 --lr 0.001 --opt sgd 

# python3 main.py --model regnet_16gf --data-path ./dacon --batch-size 32 --epochs 120 --lr 0.001 --opt sgd --n-mean 0.485,0.456,0.406 --n-std 0.229,0.224,0.225
# python3 main.py --model effnetv2_s --data-path ./dacon --batch-size 32 --epochs 120 --lr 0.001 --opt sgd --n-mean 0.485,0.456,0.406 --n-std 0.229,0.224,0.225




#python3 main.py --model swinv2_s --data-path ./culture/cul_group --batch-size 32 --epochs 80 --lr 0.001 --opt sgd
# python3 main.py --model resnet18 --data-path ./culture/cul_group --batch-size 32 --epochs 80 --lr 0.001 --opt sgd
# python3 main.py --model swinv2_t --data-path ./culture/cul_group --batch-size 32 --epochs 80 --lr 0.001 --opt sgd
# python3 main.py --model regnet_16gf --data-path ./culture/cul_group --batch-size 32 --epochs 80 --lr 0.001 --opt sgd
# python3 main.py --model effnetv2_s --data-path ./culture/cul_group --batch-size 32 --epochs 80 --lr 0.001 --opt sgd

# #python3 main.py --model swinv2_s --data-path ./culture/cul56 --batch-size 32 --epochs 80 --lr 0.001 --opt sgd
# python3 main.py --model resnet18 --data-path ./culture/cul56 --batch-size 32 --epochs 80 --lr 0.001 --opt sgd
# python3 main.py --model swinv2_t --data-path ./culture/cul56 --batch-size 32 --epochs 80 --lr 0.001 --opt sgd
# python3 main.py --model regnet_16gf --data-path ./culture/cul56 --batch-size 32 --epochs 80 --lr 0.001 --opt sgd
# python3 main.py --model effnetv2_s --data-path ./culture/cul56 --batch-size 32 --epochs 80 --lr 0.001 --opt sgd

# #python3 main.py --model swinv2_s --data-path ./culture/cul45 --batch-size 32 --epochs 80 --lr 0.001 --opt sgd
# python3 main.py --model resnet18 --data-path ./culture/cul45 --batch-size 32 --epochs 80 --lr 0.001 --opt sgd
# python3 main.py --model swinv2_t --data-path ./culture/cul45 --batch-size 32 --epochs 80 --lr 0.001 --opt sgd
# python3 main.py --model regnet_16gf --data-path ./culture/cul45 --batch-size 32 --epochs 80 --lr 0.001 --opt sgd
# python3 main.py --model effnetv2_s --data-path ./culture/cul45 --batch-size 32 --epochs 80 --lr 0.001 --opt sgd

# #python3 main.py --model swinv2_s --data-path ./culture/cul_aeg --batch-size 32 --epochs 80 --lr 0.001 --opt sgd
# python3 main.py --model resnet18 --data-path ./culture/cul_aeg --batch-size 32 --epochs 80 --lr 0.001 --opt sgd
# python3 main.py --model swinv2_t --data-path ./culture/cul_aeg --batch-size 32 --epochs 80 --lr 0.001 --opt sgd
# python3 main.py --model regnet_16gf --data-path ./culture/cul_aeg --batch-size 32 --epochs 80 --lr 0.001 --opt sgd
# python3 main.py --model effnetv2_s --data-path ./culture/cul_aeg --batch-size 32 --epochs 80 --lr 0.001 --opt sgd


#python3 main.py --model resnet18 --data-path ./skin/skin_2 --batch-size 32 --epochs 100 --lr 0.001 --opt sgd --test True
#python3 main.py --model swinv2_t --data-path ./skin/skin_2 --batch-size 32 --epochs 80 --lr 0.001 --opt sgd --test True
#python3 main.py --model regnet_16gf --data-path ./skin/skin_2 --batch-size 32 --epochs 80 --lr 0.001 --opt sgd --test True
# python3 main.py --model effnetv2_s --data-path ./skin/skin_2 --batch-size 32 --epochs 100 --lr 0.001 --opt sgd --test True

# python3 main.py --model resnet18 --data-path ./skin/skin_3 --batch-size 32 --epochs 50 --lr 0.001 --opt sgd --test True
# python3 main.py --model effnetv2_s --data-path ./skin/skin_3 --batch-size 32 --epochs 35 --lr 0.001 --opt sgd --test True

python3 main.py --model effnetv2_s --data-path ./culture/cul3_AE6_unbal --batch-size 32 --epochs 20 --lr 0.001 --opt sgd --test True 
#--resume ./model_log/cul3_AE6_unbalresize_300_model_effnetv2_s_2023_05_09_060535/cul3_AE6_e

#python3 main.py --model resnet18 --data-path ./culture/cul_transfer --batch-size 32 --epochs 6 --lr 0.0001 --opt sgd
#--resume ./model_log/cul3_AE6_unbalresize_300_model_effnetv2_s_2023_05_09_060535/cul3_AE6_e