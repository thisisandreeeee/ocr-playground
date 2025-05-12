from transformers import AutoProcessor, TableTransformerForObjectDetection
from PIL import Image
import torch
import matplotlib.pyplot as plt
import os

COLORS = [
    [0.000, 0.447, 0.741],
    [0.850, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
    [0.301, 0.745, 0.933],
]


def visualize_results(
    image: Image.Image,
    scores: torch.Tensor,
    labels: torch.Tensor,
    boxes: torch.Tensor,
    id_to_label: dict,
):
    """Visualizes the detected objects on the image."""
    plt.figure(figsize=(16, 10))
    plt.imshow(image)
    ax = plt.gca()
    num_colors = len(COLORS)
    for score, label, (xmin, ymin, xmax, ymax), color in zip(
        scores.tolist(),
        labels.tolist(),
        boxes.tolist(),
        COLORS * (len(scores) // num_colors + 1),
    ):
        rect = plt.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=color, linewidth=3
        )
        ax.add_patch(rect)
        text = f"{id_to_label[label]}: {score:0.2f}"
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor="yellow", alpha=0.5))
    plt.axis("off")
    plt.show()


def crop_and_save_tables(
    image: Image.Image, results: dict, output_dir: str, id_to_label: dict
):
    """Crops detected tables and saves them to the specified directory."""
    os.makedirs(output_dir, exist_ok=True)
    for i, (box, label_id, score) in enumerate(
        zip(results["boxes"], results["labels"], results["scores"])
    ):
        x0, y0, x1, y1 = map(int, box.tolist())
        cropped_table = image.crop((x0, y0, x1, y1))
        crop_path = os.path.join(output_dir, f"table_{i+1}.png")
        cropped_table.save(crop_path)
        label = id_to_label[label_id.item()]
        confidence = round(score.item(), 3)
        print(
            f"Saved cropped {label} {i+1} with confidence {confidence} to {crop_path} at {box.tolist()}"
        )


def process_image(
    image_path: str,
    processor,
    model,
    output_dir: str = "cropped_tables",
    show: bool = False,
):
    """Loads, processes an image to detect tables, visualizes results, and saves cropped tables."""
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, threshold=0.5, target_sizes=target_sizes
    )[0]

    if show:
        visualize_results(
            image,
            results["scores"],
            results["labels"],
            results["boxes"],
            model.config.id2label,
        )
    crop_and_save_tables(image, results, output_dir, model.config.id2label)


if __name__ == "__main__":
    # Configuration
    model_name = "microsoft/table-transformer-detection"
    image_file = "table.png"

    # Load resources
    processor = AutoProcessor.from_pretrained(model_name)
    model = TableTransformerForObjectDetection.from_pretrained(model_name)

    # Process the image
    process_image(image_file, processor, model)
