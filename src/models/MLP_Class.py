from torch import nn

class MLPClassifier(nn.Module):
    def __init__(self, in_features, hidden_units, out_classes):
        super(MLPClassifier, self).__init__() 
        self.hidden_layer = nn.Linear(in_features, hidden_units)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(hidden_units, out_classes)
        
        
    def forward(self,x):
        hidden_representation = self.hidden_layer(x)
        hidden_representation = self.activation(hidden_representation)
        scores = self.output_layer(hidden_representation)
        return scores