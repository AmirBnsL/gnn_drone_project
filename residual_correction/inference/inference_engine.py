import torch

class InferenceEngine:
    def __init__(self, model, device="cuda"):
        self.device = device
        self.model = model.to(device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, graph):
        graph = graph.to(self.device)
        output = self.model(graph)
        return output.cpu()