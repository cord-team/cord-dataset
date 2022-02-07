import torch

from .objects import Ontology
from .transforms import TransformOutput


def get_collate_top_level_object_classes(ontology: Ontology):
    ids = sorted([o.id for o in ontology.objects])
    classes = {id: i for i, id in enumerate(ids)}

    def collate_fn(objects: TransformOutput):
        imgs = torch.stack([t.img for t in objects], dim=0)
        labels = []
        for t in objects:
            cls = next(
                c.ontology_id for c in t.attributes.classes if c.ontology_id in classes
            )
            labels.append(classes[cls])

        labels = torch.tensor(labels, dtype=torch.long)
        return imgs, labels

    return collate_fn
