# Cross-Domain Pedestrian Trajectory Prediction via Behavioral Pattern-Aware Multi-Instance GCN
## Environment
    python=3.8
    torch==2.3.1+cu118 torchvision==0.18.1+cu118 torchaudio==2.3.1+cu118
    pip install -r requirements.txt
## Train
The default dataset combination is ETH-eth as the source domain and ETH-hotel as the target domain. The data cache and model will be saved to./checkpoints/eth2hotel<br><br>
The source domain and target domain datasets can be manually set according to the comments in train.py and test.py
``` git clone https://github.com/cymdd/PMITra.git
    cd PMITra
    python train.py
  
