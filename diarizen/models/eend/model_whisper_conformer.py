#!/usr/bin/env python3

# Licensed under the MIT license.
# Copyright 2020 CNRS (author: Herve Bredin, herve.bredin@irit.fr)
# Copyright 2024 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)


import os
import torch
import torch.nn as nn
import copy 
from functools import lru_cache

from pyannote.audio.core.model import Model as BaseModel
from pyannote.audio.utils.receptive_field import (
    multi_conv_num_frames, 
    multi_conv_receptive_field_size, 
    multi_conv_receptive_field_center
)

from diarizen.models.module.conformer import ConformerEncoder
from transformers import WhisperModel, AutoFeatureExtractor

class Model(BaseModel):
    def __init__(
        self,
        whisper_src: str = "openai/whisper_small",
        whisper_layer_num: int = 13,
        whisper_feat_dim: int = 768,
        attention_in: int = 256,
        ffn_hidden: int = 1024,
        num_head: int = 4,
        num_layer: int = 4,
        kernel_size: int = 31,
        dropout: float = 0.1,
        use_posi: bool = False,
        output_activate_function: str = False,
        max_speakers_per_chunk: int = 4,
        max_speakers_per_frame: int = 2,
        chunk_size: int = 5,
        num_channels: int = 8,
        selected_channel: int = 0,
        sample_rate: int = 16000,
    ):
        super().__init__(
            num_channels=num_channels,
            duration=chunk_size,
            max_speakers_per_chunk=max_speakers_per_chunk,
            max_speakers_per_frame=max_speakers_per_frame
        )
        
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.selected_channel = selected_channel

        # whisper 
        self.encoder_model = self.load_whisper(whisper_src)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_src)
        self.weight_sum = nn.Linear(whisper_layer_num, 1, bias=False)

        self.proj = nn.Linear(whisper_feat_dim, attention_in)
        self.lnorm = nn.LayerNorm(attention_in)

        self.conformer = ConformerEncoder(
            attention_in=attention_in,
            ffn_hidden=ffn_hidden,
            num_head=num_head,
            num_layer=num_layer,
            kernel_size=kernel_size,
            dropout=dropout,
            use_posi=use_posi,
            output_activate_function=output_activate_function
        )

        self.classifier = nn.Linear(attention_in, self.dimension)
        self.activation = self.default_activation()

    def non_encoder_parameters(self):
        return [
            *self.weight_sum.parameters(),
            *self.proj.parameters(),
            *self.lnorm.parameters(),
            *self.conformer.parameters(),
            *self.classifier.parameters(),
        ]

    @property
    def dimension(self) -> int:
        """Dimension of output"""
        if isinstance(self.specifications, tuple):
            raise ValueError("PyanNet does not support multi-tasking.")

        if self.specifications.powerset:
            return self.specifications.num_powerset_classes
        else:
            return len(self.specifications.classes)

    @lru_cache
    def num_frames(self, num_samples: int) -> int:
        """Compute number of output frames

        Parameters
        ----------
        num_samples : int
            Number of input samples.

        Returns
        -------
        num_frames : int
            Number of output frames.
        """
        # Whisper encoder: log-mel (n_fft=400, hop=160) + 2 conv1d layers
        kernel_size = [400, 3, 3]  # mel spectrogram window, conv1, conv2
        stride = [160, 1, 2]        # mel hop length, conv1 stride, conv2 stride
        padding = [0, 1, 1]         # no padding for mel, padding=1 for both convs
        dilation = [1, 1, 1]

        return multi_conv_num_frames(
            num_samples,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    def receptive_field_size(self, num_frames: int = 1) -> int:
        """Compute size of receptive field

        Parameters
        ----------
        num_frames : int, optional
            Number of frames in the output signal

        Returns
        -------
        receptive_field_size : int
            Receptive field size.
        """
        # Whisper encoder: log-mel (n_fft=400, hop=160) + 2 conv1d layers
        kernel_size = [400, 3, 3]  # mel spectrogram window, conv1, conv2
        stride = [160, 1, 2]        # mel hop length, conv1 stride, conv2 stride
        dilation = [1, 1, 1]

        return multi_conv_receptive_field_size(
            num_frames,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )

    def receptive_field_center(self, frame: int = 0) -> int:
        """Compute center of receptive field

        Parameters
        ----------
        frame : int, optional
            Frame index

        Returns
        -------
        receptive_field_center : int
            Index of receptive field center.
        """
        # Whisper encoder: log-mel (n_fft=400, hop=160) + 2 conv1d layers
        kernel_size = [400, 3, 3]  # mel spectrogram window, conv1, conv2
        stride = [160, 1, 2]        # mel hop length, conv1 stride, conv2 stride
        padding = [0, 1, 1]         # no padding for mel, padding=1 for both convs
        dilation = [1, 1, 1]

        return multi_conv_receptive_field_center(
            frame,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
    
    @property
    def get_rf_info(self):     
        """Return receptive field info to dataset
        """

        receptive_field_size = self.receptive_field_size(num_frames=1)
        receptive_field_step = (
            self.receptive_field_size(num_frames=2) - receptive_field_size
        )
        num_frames = self.num_frames(self.chunk_size * self.sample_rate)
        duration = receptive_field_size / self.sample_rate
        step=receptive_field_step / self.sample_rate
        return num_frames, duration, step

    def load_whisper(self, source: str):
        """
        Load a Whisper model from either a config name or a checkpoint file.

        Parameters
        ----------
        source : str
            - If `source` is a config name (e.g., "whisper_small"), 
            the model will be initialized using predefined configuration.
            - If `source` is a file path, the model will be loaded from the checkpoint.

        Returns
        -------
        model : nn.Module
            Initialized Whisper model.
        """
        model = WhisperModel.from_pretrained(
            source,
            output_hidden_states=True
        )
        self.embed_positions = copy.deepcopy(model.encoder.embed_positions.weight)
        model.encoder.embed_positions = model.encoder.embed_positions.from_pretrained(self.embed_positions[:400])
        model.encoder.embed_positions.requires_grad = False

        return model


    def wav2whisper(self, in_wav, model):
        """
        transform wav to whisper features
        """
        # Convert tensor to numpy for feature extraction
        in_wav_np = in_wav.cpu().numpy() if in_wav.is_cuda else in_wav.numpy()
        
        # Extract features using Whisper's feature extractor
        features = self.feature_extractor(
            in_wav_np,
            return_tensors="pt", 
            sampling_rate=self.sample_rate,
            max_length=in_wav.shape[-1],
            padding=False
        )
        
        # Move input_features to the same device as the model
        input_features = features.input_features.to(in_wav.device)
        
            
        # Use encoder only (we don't need decoder for feature extraction)
        outputs = model.encoder(input_features, output_hidden_states=True)
        layer_reps = outputs.hidden_states
        return torch.stack(layer_reps, dim=-1)
    
    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        waveforms : (batch, sample) or (batch, channel, sample)

        Returns
        -------
        scores : (batch, frame, classes)
        """
        assert waveforms.dim() == 3
        waveforms = waveforms[:, self.selected_channel, :]

        whisper_feat = self.wav2whisper(waveforms, self.encoder_model)
        whisper_feat = self.weight_sum(whisper_feat)
        whisper_feat = torch.squeeze(whisper_feat, -1)

        outputs = self.proj(whisper_feat)
        outputs = self.lnorm(outputs)
        
        outputs = self.conformer(outputs)

        outputs = self.classifier(outputs)
        outputs = self.activation(outputs)
        outputs = outputs[:, :-1, :]  # remove last frame to match target length
        outputs = outputs.contiguous()
        return outputs


if __name__ == '__main__':
    whisper_src = 'openai/whisper-medium'
    model = Model(whisper_src=whisper_src, whisper_layer_num=25, whisper_feat_dim=1024)
    print(model)
    x = torch.randn(2, 1, 8*16000)
    y = model(x)
    print(f'y: {y.shape}')