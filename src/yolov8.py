from src.utils import *

class YOLOv8(nn.Module):
    def __init__(self, features, num_bboxes=2, num_classes=4, bn=True):
        super(YOLOv8, self).__init__()
        self.backbone = resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.feature_size = 7
        self.num_bboxes = num_bboxes
        self.num_classes = num_classes

        self.features = features
        self.fc_layers = self._make_fc_layers()


    def _make_fc_layers(self):
        S, B, C = self.feature_size, self.num_bboxes, self.num_classes

        net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 4096),  
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, S * S * (5 * B + C)),
            nn.Sigmoid()
        )

        return net

    def forward(self, x):
        S, B, C = self.feature_size, self.num_bboxes, self.num_classes
        x=self.backbone(x)
        x = self.features(x)
        x = self.fc_layers(x)

        x = x.view(-1, S, S, 5 * B + C)
        return x
    
def collate_fn(batch):
    return tuple(zip(*batch))


