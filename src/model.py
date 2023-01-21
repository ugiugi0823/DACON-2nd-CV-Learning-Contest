# model.py
"""
Here every model to be used for pretraining/training is defined.
"""
class PlainResnet50(nn.Module):
    def __init__(self):
        super(PlainResnet50, self).__init__()
        
        base_model = resnet50()
        self.block = nn.Sequential(
            base_model,
            nn.Linear(1000, 26),
        )
        
        nn.init.xavier_normal_(self.block[1].weight)
        
    def forward(self, x):
        out = self.block(x)
        return out

    
class PlainEfficientnetB4(nn.Module):
    def __init__(self):
        super(PlainEfficientnetB4, self).__init__()
        
        base_model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=26)
        self.block = nn.Sequential(
            base_model
        )
        
        nn.init.xavier_normal_(self.block[0]._fc.weight)
        
    def forward(self, x):
        out = self.block(x)
        return out


class PlainEfficientnetB5(nn.Module):
    def __init__(self):
        super(PlainEfficientnetB5, self).__init__()
        
        base_model = EfficientNet.from_pretrained('efficientnet-b5', num_classes=26)
        self.block = nn.Sequential(
            base_model
        )
        
        nn.init.xavier_normal_(self.block[0]._fc.weight)
        
    def forward(self, x):
        out = self.block(x)
        return out


class PlainEfficientnetB7(nn.Module):
    def __init__(self):
        super(PlainEfficientnetB7, self).__init__()
        
        base_model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=26)
        self.block = nn.Sequential(
            base_model
        )
        
        nn.init.xavier_normal_(self.block[0]._fc.weight)
        
    def forward(self, x):
        out = self.block(x)
        return out
