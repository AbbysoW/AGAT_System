from pathlib import Path

import torch
import numpy as np

from .embaddings_model.model import Model as EmbaddingsModel


class Model:
    bias = -0.15

    def __init__(self):
        self.pos_prototypes = []
        self.neg_prototypes = []
        self.__load_prototypes()

        self.embaddings = EmbaddingsModel()


    def __call__(self, text: str) -> float:
        vector = self._get_embaddings(text)

        print(text)
        print(vector[:5])

        max_pos_similarity = max(self._get_similarity(vector, proto) for proto in self.pos_prototypes)
        max_neg_similarity = max(self._get_similarity(vector, proto) for proto in self.neg_prototypes)

        print(f"Positive similarity: {max_pos_similarity}")
        print(f"Negative similarity: {max_neg_similarity - self.bias}")

        return float(max_pos_similarity > max_neg_similarity - self.bias)

    def _get_embaddings(self, text: str) -> torch.Tensor:
        return self.embaddings(text) # shape [d_model]


    def _get_similarity(self, vector: torch.Tensor, prototype: torch.Tensor) -> float:
        return torch.cosine_similarity(vector.unsqueeze(0), prototype.unsqueeze(0)).item()

    def __load_prototypes(self):
        try:
            base_path = Path(__file__).parent
            self.load_prototypes(
                pos_prototypes_path=base_path / "pos_prototypes.npy",
                neg_prototypes_path=base_path / "neg_prototypes.npy"
            )
        except FileNotFoundError:
            print("Prototype files not found. Please make prototypes first.")

    def load_prototypes(self, pos_prototypes_path: str, neg_prototypes_path: str):
        pos = np.load(pos_prototypes_path)
        neg = np.load(neg_prototypes_path)

        self.pos_prototypes = [torch.tensor(p, dtype=torch.float32) for p in pos]
        self.neg_prototypes = [torch.tensor(n, dtype=torch.float32) for n in neg]

    def make_prototypes(self, pos_class_text: list[str], neg_class_text: list[str], num_prototypes: int = 3):
        from sklearn.cluster import KMeans    

        pos_embeddings = torch.stack([self._get_embaddings(text) for text in pos_class_text])

        kmeans = KMeans(n_clusters=num_prototypes)
        kmeans.fit(pos_embeddings)

        pos_prototypes = kmeans.cluster_centers_

        neg_embeddings = torch.stack([self._get_embaddings(text) for text in neg_class_text])
        kmeans.fit(neg_embeddings)
        neg_prototypes = kmeans.cluster_centers_

        self.pos_prototypes = [torch.tensor(p, dtype=torch.float32) for p in pos_prototypes]
        self.neg_prototypes = [torch.tensor(n, dtype=torch.float32) for n in neg_prototypes]

        # Save prots in own file as np array
        base_path = Path(__file__).parent
        np.save(base_path / "pos_prototypes.npy", 
                np.stack([p.numpy() for p in self.pos_prototypes]))
        np.save(base_path / "neg_prototypes.npy", 
                np.stack([n.numpy() for n in self.neg_prototypes]))


if __name__ == '__main__':
    import csv
    
    classifier = Model()

    # # Open csv as list
    # with open('/mnt/windows/Programs/Datasets/SA_Binary_Classifier/SSU/positive_class.csv', mode='r') as file:
    #     reader = csv.reader(file, delimiter=';')
    #     # Assuming you want the first column (index 0)
    #     pos_list = [row[0] for row in reader]

    # with open('/mnt/windows/Programs/Datasets/SA_Binary_Classifier/SSU/neutral_class.csv', mode='r') as file:
    #     reader = csv.reader(file, delimiter=';')
    #     # Assuming you want the first column (index 0)
    #     neg_list = [row[0] for row in reader]

    # # Make prototipes
    # classifier.make_prototypes(
    #     pos_class_text=pos_list,
    #     neg_class_text=neg_list
    # )

    # input("Press Enter to continue...")

    print(classifier("Тестовое сообщение")) # False
    print(classifier("Помолчи пожалуйста")) # True
    print(classifier("Хватит")) # True
    print(classifier("Привет Агат")) # False
    print(classifier("ПЕНИС")) # False

    print(classifier("Агат, открой браузер и перейди на YouTube")) # False
    print(classifier("Замолкни, пожалуйста")) # True

    print(classifier("Agate, open the browser and go to YouTube")) # False
    print(classifier("Please be quiet")) # True

