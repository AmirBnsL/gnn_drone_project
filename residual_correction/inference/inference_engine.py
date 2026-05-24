import torch

class InferenceEngine:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def predict(self, graph):
        graph = graph.to(self.device)
        out = self.model(graph)
        return out.cpu()