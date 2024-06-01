from torchreid.reid.utils import FeatureExtractor
import torch
extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    model_path='osnet_x1_0_custom.pth.tar-25',
    device='cpu'
)
print(extractor.model)
torch.save(extractor.model.state_dict(), 'osnet_x1_0.pt')