from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer as _InputLayer

def InputLayer(*args, **kwargs):
    kwargs.pop("batch_shape", None)
    return _InputLayer(*args, **kwargs)

model = load_model(
    "final_model.h5",
    compile=False,
    custom_objects={"InputLayer": InputLayer}
)

model.save("final_model_stream.h5")
print("✅ Saved final_model_stream.h5")
