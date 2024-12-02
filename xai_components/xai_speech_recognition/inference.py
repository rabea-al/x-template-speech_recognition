from xai_components.base import InArg, OutArg, InCompArg, Component, xai_component

#------------------------------------------------------------------------------
#                    Xircuits Component : LoadModel
#------------------------------------------------------------------------------
@xai_component
class LoadModel(Component):
    model_path: InArg[str]
        
    def execute(self, ctx):
        import tensorflow as tf
        import numpy as np
        
        model = tf.keras.models.load_model(self.model_path.value)
        
        ctx.update({'model':model})
        self.done = True
        

#------------------------------------------------------------------------------
#                    Xircuits Component : LoadAudioFile
#------------------------------------------------------------------------------
@xai_component
class LoadAudioFile(Component):
    audio_file: InArg[str]
    classes: InArg[list]
    
    audio_data: OutArg[any]
        
    def execute(self, ctx):
        import tensorflow as tf
        import numpy as np
        audio_file = self.audio_file.value
        
        AUTOTUNE = tf.data.AUTOTUNE
        audio_data = tf.data.Dataset.from_tensor_slices([audio_file])

        # Convert audio to waveforms
        def audioToTensor(filepath):
            audio_binary = tf.io.read_file(filepath)
            audio, _ = tf.audio.decode_wav(audio_binary)
            audio = tf.squeeze(audio, axis=-1)
            return audio
        
        waveform_ds = audio_data.map(map_func=audioToTensor, num_parallel_calls=AUTOTUNE)  
        
        # Convert waveforms to spectrogram
        def get_spectrogram(waveform):
            # Zero-padding for an audio waveform with less than 16,000 samples.
            input_len = 16000
            # audio = waveform[0]
            waveform = waveform[:input_len]
            zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)
            # Cast the waveform tensors' dtype to float32.
            waveform = tf.cast(waveform, dtype=tf.float32)
            # Concatenate the waveform with `zero_padding`, which ensures all audio
            # clips are of the same length.
            equal_length = tf.concat([waveform, zero_padding], 0)
            # Convert the waveform to a spectrogram via a STFT.
            spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128)
            # Obtain the magnitude of the STFT.
            spectrogram = tf.abs(spectrogram)
            # Add a `channels` dimension, so that the spectrogram can be used
            # as image-like input data with convolution layers (which expect
            # shape (`batch_size`, `height`, `width`, `channels`).
            spectrogram = spectrogram[..., tf.newaxis]
            return spectrogram
        
        spectrogram_ds = waveform_ds.map(map_func=get_spectrogram, num_parallel_calls=AUTOTUNE) 
        
        # Prepare data for prediction
        test_data = []
        for i in spectrogram_ds:
            test_data.append(i.numpy())
        test_data = np.array(test_data)

        ctx.update({'commands':self.classes.value})
        self.audio_data.value = test_data
        
        self.done = True
        
#------------------------------------------------------------------------------
#                    Xircuits Component : PredictSpeech
#------------------------------------------------------------------------------
@xai_component
class PredictSpeech(Component):
    spectrogram_data: InArg[any]
        
    def execute(self, ctx):
        import tensorflow as tf
        import matplotlib.pyplot as plt
        
        spectrogram_data = self.spectrogram_data.value
        model = ctx['model']
        commands = ctx['commands']
        
        prediction = model.predict(spectrogram_data)
        plt.bar(commands, tf.nn.softmax(prediction[0]))
        plt.show()