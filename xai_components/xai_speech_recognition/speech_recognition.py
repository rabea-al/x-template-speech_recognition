from xai_components.base import InArg, OutArg, InCompArg, Component, xai_component

#------------------------------------------------------------------------------
#                    Xircuits Component : DownloadDataset
#------------------------------------------------------------------------------
@xai_component
class DownloadDataset(Component):
    dataset_url: InArg[str]
    save_dataset_path: InArg[str]
    
    data_dir: OutArg[any]

        
    def execute(self, ctx):
        import pathlib
        import tensorflow as tf
        import os

        DATASET_PATH = self.save_dataset_path.value
        fname = os.path.basename(DATASET_PATH) + '.zip'
        data_dir = pathlib.Path(DATASET_PATH)

        if not data_dir.exists():
            tf.keras.utils.get_file(
              fname,
              origin=self.dataset_url.value,
              extract=True,
              cache_dir='.', cache_subdir='data')
        
        self.data_dir.value = data_dir
        self.done = True
        
#------------------------------------------------------------------------------
#                    Xircuits Component : ExtractAudioFilesAndLabels
#------------------------------------------------------------------------------
@xai_component
class ExtractAudioFilesAndLabels(Component):
    data_dir: InArg[str]
    dataset_files: OutArg[any]

    def execute(self, ctx):
        import os
        import tensorflow as tf

        data_dir = self.data_dir.value

        filenames = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.wav'):
                    filenames.append(os.path.join(root, file))

        if not filenames:
            raise ValueError(f"No .wav files found in directory: {data_dir}")

        filenames = tf.random.shuffle(filenames).numpy()
        filenames = [f.decode('utf-8') for f in filenames]

        num_samples = len(filenames)

        commands = sorted(set(os.path.basename(os.path.dirname(f)) for f in filenames))

        print(f'Number of total examples: {num_samples}')
        print(f'Commands: {commands}')

        ctx.update({'commands': commands, 'dataset_size': num_samples})
        self.dataset_files.value = filenames
        self.done = True

#------------------------------------------------------------------------------
#                    Xircuits Component : AudioToTensors
#------------------------------------------------------------------------------
@xai_component
class AudioToTensors(Component):
    dataset_files: InArg[any]
    waveform_data: OutArg[any]

    def execute(self, ctx):
        import tensorflow as tf
        import os

        dataset_files = self.dataset_files.value

        AUTOTUNE = tf.data.AUTOTUNE
        datasets = tf.data.Dataset.from_tensor_slices(dataset_files)

        def get_waveform_and_label(file_path):
            try:
                label = tf.strings.split(file_path, os.path.sep)[-2]
                audio_binary = tf.io.read_file(file_path)
                audio, _ = tf.audio.decode_wav(audio_binary)
                waveform = tf.squeeze(audio, axis=-1)
                return waveform, label
            except Exception as e:
                print(f"Skipping file {file_path}: {e}")
                return None, None

        waveform_ds = datasets.map(map_func=get_waveform_and_label, num_parallel_calls=AUTOTUNE)
        self.waveform_data.value = waveform_ds
        self.done = True

        
#------------------------------------------------------------------------------
#                    Xircuits Component : WaveformsToSpectrograms
#------------------------------------------------------------------------------
@xai_component
class WaveformsToSpectrograms(Component):
    waveform_data: InArg[any]
    
    spectrogram_data: OutArg[any]
        
    def execute(self, ctx):
        import tensorflow as tf
        waveform_data = self.waveform_data.value
        commands=ctx['commands']
        
        # Convert waveforms to spectrogram
        # time domain --> time-frequency domain
        # 1. Compute STFT for conversion
        # 2. Spectrogram will be fed into the neural network
        
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
        
        def get_spectrogram_and_label_id(audio, label):
            spectrogram = get_spectrogram(audio)
            label_id = tf.math.argmax(label == commands)
            return spectrogram, label_id
        
        AUTOTUNE = tf.data.AUTOTUNE
        spectrogram = waveform_data.map(map_func=get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)
        
        self.spectrogram_data.value = spectrogram
        
        self.done = True
        
#------------------------------------------------------------------------------
#                    Xircuits Component : PlotSpectrogram
#------------------------------------------------------------------------------
@xai_component
class PlotSpectrogram(Component):
    spectrogram_data: InArg[any]
    
    def execute(self, ctx):
        import matplotlib.pyplot as plt
        import numpy as np
        
        spectogram_data = self.spectrogram_data.value
        commands = ctx['commands']
        
        def plot_spectrogram(spectrogram, ax):
            if len(spectrogram.shape) > 2:
                assert len(spectrogram.shape) == 3
                spectrogram = np.squeeze(spectrogram, axis=-1)
            # Convert the frequencies to log scale and transpose, so that the time is
            # represented on the x-axis (columns).
            # Add an epsilon to avoid taking a log of zero.
            log_spec = np.log(spectrogram.T + np.finfo(float).eps)
            height = log_spec.shape[0]
            width = log_spec.shape[1]
            X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
            Y = range(height)
            ax.pcolormesh(X, Y, log_spec)
            
        rows = 2
        cols = 2
        n = rows*cols
        fig, axes = plt.subplots(rows, cols, figsize=(8, 8))

        for i, (spectrogram, label_id) in enumerate(spectogram_data.take(n)):
            r = i // cols
            c = i % cols
            ax = axes[r][c]
            plot_spectrogram(spectrogram.numpy(), ax)
            ax.set_title(commands[label_id.numpy()])
            ax.axis('off')

        plt.show()
        
        self.done = True
        
#------------------------------------------------------------------------------
#                    Xircuits Component : SplitData
#------------------------------------------------------------------------------
@xai_component
class SplitData(Component):
    spectrogram_data: InArg[any]
    train_size: InArg[float]
        
    def execute(self, ctx):
        import numpy as np
        import tensorflow as tf
        spectrogram = self.spectrogram_data.value
        DATASET_SIZE = ctx['dataset_size']
        
        train_percentage = self.train_size.value
        val_percentage = (1 - train_percentage)/2
        
        train_size = int(train_percentage * DATASET_SIZE)
        val_size = int(val_percentage * DATASET_SIZE)
        test_size = int(val_percentage * DATASET_SIZE)

        train_dataset = spectrogram.take(train_size)
        test_files = spectrogram.skip(train_size)
        val_dataset = test_files.skip(val_size)
        test_dataset = test_files.take(test_size)

        print('Training set size', len(train_dataset))
        print('Validation set size', len(val_dataset))
        print('Test set size', len(test_dataset))
        
        batch_size = 64
        AUTOTUNE = tf.data.AUTOTUNE
        
        train_dataset = train_dataset.batch(batch_size)
        val_dataset = val_dataset.batch(batch_size)

        train_dataset = train_dataset.cache().prefetch(AUTOTUNE)
        val_dataset = val_dataset.cache().prefetch(AUTOTUNE)
        
        ctx.update({'train_dataset':train_dataset, 'val_dataset':val_dataset})
        
        # self.train_dataset.value = train_dataset
        # self.val_dataset.value = val_dataset
        ctx.update({'test_dataset':test_dataset})
        
        self.done = True
        
#------------------------------------------------------------------------------
#                    Xircuits Component : BuildSpeechModel
#------------------------------------------------------------------------------
@xai_component
class BuildSpeechModel(Component):
    model: OutArg[any]
  
    def execute(self, ctx):
        from tensorflow.keras import layers, models
        
        train_dataset = ctx['train_dataset']
        commands = ctx['commands']
        
        for spectrogram, _ in train_dataset.take(1):
            input_shape = spectrogram[0].shape
        print('Input shape:', input_shape)
        num_labels = len(commands)

        norm_layer = layers.Normalization()
        # Fit the state of the layer to the spectrograms with `Normalization.adapt`.
        norm_layer.adapt(data=train_dataset.map(map_func=lambda spec, label: spec))

        model = models.Sequential([
            layers.Input(shape=input_shape),
            # Downsample the input.
            layers.Resizing(32, 32),
            # Normalize.
            norm_layer,
            layers.Conv2D(32, 3, activation='relu'),
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_labels),
        ])

        model.summary()
        self.model.value = model
        self.done = True
        
#------------------------------------------------------------------------------
#                    Xircuits Component : CompileSpeechModel
#------------------------------------------------------------------------------
@xai_component
class CompileSpeechModel(Component):
    model: InArg[any]
    optimizer: InArg[str]
    
    compiled_model: OutArg[any]
        
    def execute(self, ctx):
        import tensorflow as tf
        model = self.model.value
        
        model.compile(
            optimizer=self.optimizer.value,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        
        self.compiled_model.value = model
        self.done = True
        
#------------------------------------------------------------------------------
#                    Xircuits Component : TrainSpeechModel
#------------------------------------------------------------------------------
@xai_component
class TrainSpeechModel(Component):
    compiled_model: InArg[any]
    epochs: InArg[int]
    
    training_metrics: OutArg[dict]
        
    def execute(self, ctx):
        import tensorflow as tf
        
        model = self.compiled_model.value
        train_dataset = ctx['train_dataset']
        val_dataset = ctx['val_dataset']
        
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.epochs.value,
            callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2)
        )
        ctx.update({'trained_model':model})
        self.training_metrics.value = history
        
        self.done = True
        
#------------------------------------------------------------------------------
#                    Xircuits Component : PlotSpeechMetrics
#------------------------------------------------------------------------------
@xai_component
class PlotSpeechMetrics(Component):
    training_metrics: InArg[dict]

    def execute(self, ctx) -> None:
        import matplotlib.pyplot as plt
        hist = self.training_metrics.value 
        history = hist.history
        
        acc = history['accuracy']
        val_acc = history['val_accuracy']

        loss = history['loss']
        val_loss = history['val_loss']
        
        epochs = range(1, len(acc)+1)

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(epochs, acc, label='Training Accuracy')
        plt.plot(epochs, val_acc, label='Validation Accuracy')
        plt.ylim([0, 1])
        plt.xticks(epochs)
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(epochs, loss, label='Training Loss')
        plt.plot(epochs, val_loss, label='Validation Loss')
        plt.ylim([0, 1.0])
        plt.xticks(epochs)
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.xlabel('epochs')
        plt.show()

        self.done = True

#------------------------------------------------------------------------------
#                    Xircuits Component : EvaluateSpeechModel
#------------------------------------------------------------------------------
@xai_component
class EvaluateSpeechModel(Component):
    confusion_matrix: InArg[bool]
    
    def execute(self, ctx):
        import numpy as np
        import matplotlib.pyplot as plt
        import tensorflow as tf
        import seaborn as sns
        
        test_dataset = ctx['test_dataset']
        model = ctx['trained_model']
        commands = ctx['commands']
        test_audio = []
        test_labels = []

        for audio, label in test_dataset:
            test_audio.append(audio.numpy())
            test_labels.append(label.numpy())

        test_audio = np.array(test_audio)
        test_labels = np.array(test_labels)
        
        y_pred = np.argmax(model.predict(test_audio), axis=1)
        y_true = test_labels

        test_acc = sum(y_pred == y_true) / len(y_true)
        print(f'Test set accuracy: {test_acc:.0%}')
        
        if self.confusion_matrix.value:
            confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(confusion_mtx,
                        xticklabels=commands,
                        yticklabels=commands,
                        annot=True, fmt='g')
            plt.xlabel('Prediction')
            plt.ylabel('Label')
            plt.show()
        
        self.done = True
        
#------------------------------------------------------------------------------
#                    Xircuits Component : SaveSpeechModel
#------------------------------------------------------------------------------
@xai_component
class SaveSpeechModel(Component):
    save_model_path: InArg[str]
    keras_format: InArg[bool]
    
    def execute(self, ctx):
        import os
        model = ctx['trained_model']
        model_name = self.save_model_path.value

        dirname = os.path.dirname(model_name)
        
        if len(dirname):
            os.makedirs(dirname, exist_ok=True)
        
        if self.keras_format.value:
            model_name = model_name + '.h5'
        else:
            model_name = model_name
            
        model.save(model_name)
        print(f"Saving model at: {model_name}")
        ctx.update({'saved_model_path': model_name})
        self.done = True
        
#------------------------------------------------------------------------------
#                    Xircuits Component : ConvertSpeechTFModelToOnnx
#------------------------------------------------------------------------------
@xai_component
class ConvertSpeechTFModelToOnnx(Component):
    output_onnx_path: InArg[str]
    
    def execute(self, ctx):
        import os
        saved_model = ctx['saved_model_path']
        onnx_path = self.output_onnx_path.value
        dirname = os.path.dirname(onnx_path)
        if len(dirname):
            os.makedirs(dirname, exist_ok=True)
            
        os.system(f"python -m tf2onnx.convert --saved-model {saved_model} --opset 11 --output {onnx_path}.onnx")
        print(f'Converted {saved_model} TF model to {onnx_path}.onnx')
        
        self.done = True