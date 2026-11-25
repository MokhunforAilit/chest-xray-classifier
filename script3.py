from pathlib import Path

import torch
from PIL import Image, ImageOps
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# Settings you can change
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.0001

# Files are stored next to main.py
PROJECT = Path(__file__).resolve().parent
DATA = PROJECT / "data"
MODEL_FILE = PROJECT / "models" / "best.pt"

CLASSES = ["normal", "pneumonia", "tuberculosis"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def create_folders():
    for split in ["train", "val", "test"]:
        for label in CLASSES:
            (DATA / split / label).mkdir(parents=True, exist_ok=True)

    MODEL_FILE.parent.mkdir(exist_ok=True)


def open_xray(path):
    with Image.open(path) as image:
        if image.format not in {"JPEG", "PNG"}:
            raise ValueError(f"Please use a JPG or PNG image: {path}")

        if image.mode not in {"L", "RGB", "RGBA", "P"}:
            raise ValueError(f"Please use an 8-bit grayscale or RGB image: {path}")

        # ResNet takes three channels. Repeat the grayscale image across all three.
        image = ImageOps.exif_transpose(image).convert("L").convert("RGB")

        # Fit the entire X-ray into a square without stretching or cropping it.
        return ImageOps.pad(
            image, (224, 224),
            method=Image.Resampling.BILINEAR,
            color="black",
        )


def is_xray_file(path):
    return Path(path).suffix.lower() in {".jpg", ".jpeg", ".png"}


def make_loader(split):
    dataset = datasets.ImageFolder(
        root=DATA / split,
        loader=open_xray,
        transform=normalize,
        is_valid_file=is_xray_file,
    )

    if dataset.classes != CLASSES:
        raise ValueError(f"Expected these class folders in {DATA / split}: {CLASSES}")

    print(f"\n{split.upper()} images")
    for number, name in enumerate(CLASSES):
        print(f"  {name}: {dataset.targets.count(number)}")

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=(split == "train"),
    )


def build_model(pretrained):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # Replace the original output layer with one score for each X-ray class.
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model.to(DEVICE)


@torch.inference_mode()
def get_predictions(model, loader):
    model.eval()
    actual = []
    predicted = []

    for images, labels in loader:
        scores = model(images.to(DEVICE))
        predicted.extend(scores.argmax(dim=1).cpu().tolist())
        actual.extend(labels.tolist())

    return actual, predicted


def train_model():
    torch.manual_seed(42)
    train_loader = make_loader("train")
    val_loader = make_loader("val")

    model = build_model(pretrained=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()
    best_f1 = -1.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            scores = model(images)
            loss = loss_function(scores, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(images)

        actual, predicted = get_predictions(model, val_loader)
        val_f1 = f1_score(actual, predicted, average="macro", zero_division=0)
        average_loss = total_loss / len(train_loader.dataset)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Training loss: {average_loss:.3f} | "
            f"Validation F1: {val_f1:.3f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            temporary = MODEL_FILE.with_suffix(".tmp")
            torch.save(
                {"weights": model.state_dict(), "classes": CLASSES},
                temporary,
            )
            temporary.replace(MODEL_FILE)
            print(f"Saved best model to {MODEL_FILE}")


def load_trained_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError("No trained model yet. Run option 1 first.")

    saved = torch.load(MODEL_FILE, map_location=DEVICE, weights_only=True)

    if saved["classes"] != CLASSES:
        raise ValueError("The saved model uses different class labels.")

    model = build_model(pretrained=False)
    model.load_state_dict(saved["weights"])
    model.eval()
    return model


@torch.inference_mode()
def predict_one_image():
    model = load_trained_model()

    from tkinter import Tk, filedialog

    window = Tk()
    window.withdraw()
    path = filedialog.askopenfilename(
        title="Choose one frontal chest X-ray",
        filetypes=[("X-ray images", "*.jpg *.jpeg *.png")],
    )
    window.destroy()

    if not path:
        print("No image selected.")
        return

    image = normalize(open_xray(path)).unsqueeze(0).to(DEVICE)
    scores = model(image)
    label = CLASSES[scores.argmax(dim=1).item()]

    print(f"\nImage: {Path(path).name}")
    print(f"Predicted dataset label: {label}")


def test_model():
    model = load_trained_model()
    test_loader = make_loader("test")
    actual, predicted = get_predictions(model, test_loader)
    label_numbers = list(range(len(CLASSES)))

    print("\nTest results")
    print(classification_report(
        actual, predicted,
        labels=label_numbers,
        target_names=CLASSES,
        zero_division=0,
    ))

    print("Confusion matrix: rows = actual, columns = predicted")
    print("Class order:", ", ".join(CLASSES))
    print(confusion_matrix(actual, predicted, labels=label_numbers))


def main():
    create_folders()

    if DEVICE.type == "cuda":
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU. Training will be slower.")

    print("Dataset folder:", DATA)
    print("\n1. Train the model")
    print("2. Choose one X-ray and predict its label")
    print("3. Evaluate the model on the test dataset")
    choice = input("Enter 1, 2, or 3: ").strip()

    if choice == "1":
        train_model()
    elif choice == "2":
        predict_one_image()
    elif choice == "3":
        test_model()
    else:
        print("Folders are ready. Run again and choose 1, 2, or 3.")


if __name__ == "__main__":
    main()