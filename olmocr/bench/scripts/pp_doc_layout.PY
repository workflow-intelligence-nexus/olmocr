# conda install -c conda-forge gcc=12.1.0
# python -m pip install paddlepaddle-gpu==2.6.1.post120 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
# pip install paddlex==3.0.0b2
# pip install https://paddle-model-ecology.bj.bcebos.com/paddlex/whl/paddlex-3.0.0rc0-py3-none-any.whl



# from paddlex import create_model
# model = create_model(model_name='PP-DocLayout-L', model_dir='/data/input/amanr/models')

# from paddlex as pdx
# model = pdx.create_model(model_dir='/data/input/amanr/models/PP-DocLayout-L')

from paddlex.inference import PaddlePredictorOption
model = PaddlePredictorOption(model_dir='/data/input/amanr/models/PP-DocLayout-L')


# import paddlex as pdx
# from paddlex.inference import Config

# config = Config(model_dir='/data/input/amanr/models/PP-DocLayout-L')
# model = pdx.inference.create_predictor(config)


# import paddlex as pdx

# model_path = "/root/.paddlex/official_models/PP-DocLayout-L"
# model = pdx.load_model(model_path)  # Instead of just 'PP-DocLayout-L'
# print("Model loaded successfully!")


# output = model.predict("layout.jpg", batch_size=1, layout_nms=True)

# for res in output:
#     res.print()
#     res.save_to_img(save_path="./output/")
#     res.save_to_json(save_path="./output/res.json")