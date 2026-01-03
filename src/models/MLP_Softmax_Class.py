from torch import nn

class MLP_Softmax_Classifier(nn.Module):
    def __init__(self, in_features, hidden_units, out_classes):
        super(MLP_Softmax_Classifier, self).__init__() 
        self.hidden_layer = nn.Linear(in_features, hidden_units)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(hidden_units, out_classes)
        self.softmax = nn.LogSoftmax(dim = 1) # Scegliamo LogSoftmax per stabilità numerica, infatti con nn.Softmax si possono avere problemi di underflow/overflow, infatti se faccio l'esponenziale di valori troppo piccoli/grandi il computer arrotonda a 0 o infinito.
        # Usando LogSoftmax, lavoriamo nel dominio del logaritmo, che è più stabile numericamente, dato che trasformiamo moltiplicazioni in somme e divisioni in sottrazioni.

        
    def forward(self,x):
        hidden_representation = self.hidden_layer(x)
        hidden_representation = self.activation(hidden_representation)
        scores = self.output_layer(hidden_representation)
        return self.softmax(scores)