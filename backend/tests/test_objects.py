from typing import Any

from brutus.vision.models import LabelInstance
from brutus.vision.objects import person_boxes, split_detections
from brutus.vision.parse import parse_label_instances
from tests.rekognition_fixtures import dog, label, labels_response, person

DOG_BOX = (0.5, 0.5, 0.3, 0.3)


def instances_of(*labels: dict[str, Any]) -> tuple[LabelInstance, ...]:
    return parse_label_instances(labels_response(*labels))


class TestSplitDetections:
    def test_given_dog_pet_and_animal_on_same_box_when_split_then_one_dog(
        self,
    ) -> None:
        instances = instances_of(
            dog(DOG_BOX),
            label(
                "Pet", (DOG_BOX,), parents=("Animal",), categories=("Animals and Pets",)
            ),
            label(
                "Animal", ((0.501, 0.5, 0.3, 0.3),), categories=("Animals and Pets",)
            ),
        )

        animals, objects = split_detections(instances, dedupe_iou=0.8)

        assert [(a.number, a.name) for a in animals] == [(1, "Dog")]
        assert objects == ()

    def test_given_person_related_labels_when_split_then_excluded(self) -> None:
        instances = instances_of(
            person((0.1, 0.1, 0.3, 0.8)),
            label("Man", ((0.1, 0.1, 0.3, 0.8),), parents=("Adult", "Male", "Person")),
            label("Face", ((0.2, 0.1, 0.1, 0.1),), categories=("Person Description",)),
        )

        assert split_detections(instances, dedupe_iou=0.8) == ((), ())

    def test_given_two_chairs_and_dog_when_split_then_numbered_left_to_right(
        self,
    ) -> None:
        instances = instances_of(
            label("Chair", ((0.7, 0.5, 0.1, 0.2), (0.1, 0.5, 0.1, 0.2))),
            dog(DOG_BOX),
        )

        animals, objects = split_detections(instances, dedupe_iou=0.8)

        assert [(o.number, o.name, o.box.left) for o in objects] == [
            (1, "Chair", 0.1),
            (2, "Chair", 0.7),
        ]
        assert [a.name for a in animals] == ["Dog"]

    def test_given_animal_without_category_when_split_then_still_animal(self) -> None:
        instances = instances_of(label("Cat", (DOG_BOX,), parents=("Animal", "Pet")))

        animals, objects = split_detections(instances, dedupe_iou=0.8)

        assert [a.name for a in animals] == ["Cat"]
        assert objects == ()

    def test_given_worn_clothing_when_split_then_not_an_object(self) -> None:
        instances = instances_of(
            person((0.3, 0.1, 0.3, 0.8)),
            label(
                "Jeans",
                ((0.35, 0.5, 0.2, 0.4),),
                parents=("Clothing", "Pants"),
                categories=("Apparel and Accessories",),
            ),
        )

        assert split_detections(instances, dedupe_iou=0.8) == ((), ())

    def test_given_wheels_inside_car_when_split_then_only_car(self) -> None:
        vehicles = ("Vehicles and Automotive",)
        instances = instances_of(
            label("Car", ((0.05, 0.3, 0.75, 0.55),), categories=vehicles),
            label(
                "Wheel",
                ((0.05, 0.5, 0.08, 0.25), (0.2, 0.6, 0.1, 0.25)),
                parents=("Machine",),
                categories=vehicles,
            ),
        )

        _, objects = split_detections(instances, dedupe_iou=0.8)

        assert [(o.number, o.name) for o in objects] == [(1, "Car")]

    def test_given_cup_on_table_when_split_then_both_kept(self) -> None:
        instances = instances_of(
            label("Table", ((0.1, 0.4, 0.8, 0.5),), categories=("Furniture",)),
            label(
                "Cup",
                ((0.4, 0.45, 0.05, 0.08),),
                categories=("Food and Beverage",),
            ),
        )

        _, objects = split_detections(instances, dedupe_iou=0.8)

        assert sorted(o.name for o in objects) == ["Cup", "Table"]


class TestPersonBoxes:
    def test_given_person_instances_when_extracted_then_boxes(self) -> None:
        instances = instances_of(
            person((0.6, 0.1, 0.2, 0.8), (0.1, 0.1, 0.2, 0.8)), dog(DOG_BOX)
        )

        assert [b.left for b in person_boxes(instances)] == [0.6, 0.1]
