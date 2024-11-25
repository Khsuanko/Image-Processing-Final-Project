- Training:
```bash
python train.py --dataroot "datasets\font" --model font_translator_gan --name TRAIN_MODEL_OUTPUT_FOLDER_NAME --no_dropout
```
- Testing:
```bash
python test.py --dataroot "datasets\font"  --model font_translator_gan  --eval --name TEST_OUTPUT_FOLDER_NAME --no_dropout
```

- Change batch size: `model\font_translator_gan_model.py > Line 13`

- Change train options: `options\base_options.py` & `options\train_options.py`
