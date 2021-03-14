set -e
conda create --name xk python=3.7
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=9.0 -c pytorch
conda install pandas
conda install gensim
conda install tqdm
