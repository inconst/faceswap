import sys
sys.path.append('./arcface/')
from arcface.model import Backbone
from arcface.mtcnn import *

detector = MTCNN()

arcface_model = Backbone(50, 0.6, 'ir_se').to(device)
arcface_model.eval()
arcface_model.load_state_dict(torch.load('saved_models/model_ir_se50.pth', map_location=device), strict=False)