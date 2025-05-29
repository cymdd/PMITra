# Cross-Domain Pedestrian Trajectory Prediction via Behavioral Pattern-Aware Multi-Instance GCN
## Environment
    python=3.8
    torch==2.3.1+cu118 torchvision==0.18.1+cu118 torchaudio==2.3.1+cu118
    pip install -r requirements.txt
## Train
The default dataset combination is ETH-eth as the source domain and ETH-hotel as the target domain. The data cache and model will be saved to./checkpoints/eth2hotel<br><br>
The source domain and target domain datasets can be manually set according to the comments in train.py and test.py
```
train.py, test.py
#domains = ['eth', 'hotel', 'zara01', 'zara02', 'students001', 'students003', 'uni_examples', 'zara03']
#Select the source domain and target domain based on domains:
#For example:  source_domain = 0   ->  source domain: eth
#              target_domain = 1   -> target domain: hotel
```
```
git clone https://github.com/cymdd/PMITra.git
cd PMITra
python train.py
```  
