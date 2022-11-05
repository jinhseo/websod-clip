#!/bin/sh

cd $SLURM_SUBMIT_DIR

echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_VERSION=$CUDA_VERSION"

export NCCL_DEBUG=INFO NCCL_P2P_DISABLE=1 NCCL_P2P_LEVEL=5
#srun -l /bin/hostname
#srun -l /bin/pwd
#srun -l /bin/date

#module purge
#module load postech
ml restore cuda11_0
while [[ $# -gt 1 ]]
    do
        key="$1"
        case $key in
            -c|--config_path)
            CONFIG_PATH="$2"
            shift # past argument
            ;;
            -o|--out_dir)
            OUTPUT_DIR="$2"
            shift # past argument
            ;;
            -g|--gpu_num)
            G="$2"
            shift
            ;;
            *) # unknown option
            ;;
        esac
    shift # past argument or value
    done
echo ${CONFIG_PATH}
echo ${TAG}

SAMPLES_DIR=$HOME/cbs

date

python -m torch.distributed.launch --master_port=$RANDOM --nproc_per_node=${G} \
    $SAMPLES_DIR/tools/train_net.py --config-file "${CONFIG_PATH}" \
    OUTPUT_DIR $HOME/cbs/output/${OUTPUT_DIR}

date
