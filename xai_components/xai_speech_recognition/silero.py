from xai_components.base import InArg, OutArg, InCompArg, Component, xai_component

#------------------------------------------------------------------------------
#                    Xircuits Component : SileroModelInference
#------------------------------------------------------------------------------
@xai_component
class SileroModelInference(Component):
    audio_file: InCompArg[str]
    language: InArg[str]
        
    def execute(self, ctx):
        import torch
        import zipfile
        import os
        import torchaudio
        from glob import glob
        
        audio_file = self.audio_file.value
        dst = os.path.basename(audio_file)
        
        lang = self.language.value if self.language.value else 'en'
        
        device = torch.device('cpu')  
        model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                               model='silero_stt',
                                               language=lang, # also available 'de', 'es'
                                               device=device)
        (read_batch, split_into_batches, read_audio, prepare_model_input) = utils  # see function signature for details

        test_files = glob(audio_file) 
        batches = split_into_batches(test_files, batch_size=10)
        input = prepare_model_input(read_batch(batches[0]), device=device)

        output = model(input)
        for example in output:
            print(f'Text: {decoder(example.cpu())}')