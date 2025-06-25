# convert_model.py

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer as _InputLayer

# Patch InputLayer to ignore 'batch_shape' kwarg
def InputLayer(*args, **kwargs):
    kwargs.pop("batch_shape", None)
    return _InputLayer(*args, **kwargs)

# Load with our patched InputLayer
model = load_model(
    "final_model.h5",
    compile=False,
    custom_objects={"InputLayer": InputLayer}
)

# Re-save in pure HDF5 (Keras 2.13 format)
model.save("final_model_streamlit.h5")
print("✅ Saved final_model_streamlit.h5")
