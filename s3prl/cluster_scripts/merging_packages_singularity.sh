## packages needed for singularity merging####
#!/bin/bash
#module load singularity
cd /home/project/13003821/fabian/projects/merging-text-transformers
#CUDA_VISIBLE_DEVICES=0 singularity exec --nv --bind /home/project/13003821/fabian/projects/multi_distiller/s3prl/s3prl:/workspace/s3prl/s3prl,/home/project/13003821/fabian/corpus/superb:/livingrooms/public/superb,/home/project/13003821/fabian/projects:/home/project/13003821/fabian/projects s3prl_for_sslm_v2.sif bash
source /opt/conda/bin/activate s3prl_old_cuda
BASE_DIR_S3PRL="/workspace/s3prl/s3prl"
cd $BASE_DIR_S3PRL
export PYTHONPATH=/workspace/s3prl:$PYTHONPATH

# Install necessary Python packages
pip install networkx pytorch-nlp transformers datasets==2.14.5 scipy==1.8.* librosa==0.9.* scikit-learn==0.24.2 matplotlib==3.3.4 modelscope==1.11.0 ipdb
pip install evaluate


# Configure Git and pull the latest changes if necessary
cd /workspace/s3prl
git config --global --add safe.directory /workspace/s3prl
cd /workspace/s3prl/s3prl
