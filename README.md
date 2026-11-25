# Chest X-ray classifier

An educational Streamlit demo that assigns a frontal chest X-ray one of three dataset labels: normal, pneumonia, or tuberculosis. The ResNet-18 model was trained using PyTorch. It always picks one label and can misclassify other conditions or unrelated images. The output is unsuitable for diagnosis. Use de-identified sample images.

## Run the web app

Use Python 3.12 on Linux with the CPU requirements in this repository.

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Prototype results

On the project test set, the saved model classified 294 of 300 images correctly (98%). Correct counts were 96/100 normal, 99/100 pneumonia and 99/100 tuberculosis. These figures describe this test set only.
