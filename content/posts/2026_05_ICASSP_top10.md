---
author: "Arjun Bahuguna"
title: "Top 5 Takeaways from ICASSP 2026"
date: "2026-05-09"
description: ""
tags: ["conference takeaways", "music information retrieval", "large audio language models"]
ShowToc: true
TocOpen: false
---

I was at the IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP) in Barcelona this year 🟨🟥🟨🟥. Here are my five main takeaways from the papers, tutorials and workshops I went to.

## 1. Researchers are probing how audio LLMs work

At least 14 papers looked inside large audio language models to understand how they behave. Morais & Fuentes applied Shapley-value decomposition (MM-SHAP) to measure how much each modality contributes in music tasks, and found that audio and text contribute unevenly across subtasks. Chen used variational inference to calibrate uncertainty in audio question answering. Tam & Chen ran a controlled study showing that audio LLMs make biased clinical decisions when the speaker's voice characteristics change. Kim et al. released an auditory illusion benchmark that exposed systematic failure modes in these models. Sridhar et al. (Audiocards) found that injecting structured metadata at inference time significantly improves text-audio retrieval, which suggests these models rely heavily on linguistic scaffolding.

## 2. Flow matching is taking over from diffusion

Flow matching came up in 25 papers. Continuous normalizing flows with optimal transport trajectories now appear in all the places where diffusion appeared two years ago. Welker et al. demonstrated real-time streaming mel vocoding with it, Hsieh & Braun used it for speech restoration under hard latency budgets, and Vosoughi et al. used latent rectified flow matching to synthesize multimodal room impulse responses. In text-to-speech, Pankov et al. (PFluxTTS) combined flow matching with inference-time model fusion for cross-lingual voice cloning, and Chen et al. (F5E-TTS) aligned semantic representations in a diffusion-transformer backbone. Zhang et al. (AnyAccomp) applied it to accompaniment generation through a quantized melodic bottleneck. Most of these are latency-sensitive tasks, so my read is that inference efficiency is what is driving the switch.

Stanley Chan's tutorial on Diffusion Models for Imaging and Vision was a good grounding in what flow matching is replacing. He traced diffusion from VAEs to DDPMs to score-based methods, covering score matching and Langevin dynamics along the way, and then unified all of them through SDEs. If I could get the recording of one Day 1 session to revisit, it would be this one. He annotated his slides with a stylus as he talked and made dense math easy to follow.

## 3. Neural audio codecs are becoming shared infrastructure

Eleven papers treated codecs as general-purpose learned representations for many tasks beyond compression. Zhang et al. (STACodec) used distillation to separate semantic and acoustic tokens, which lets you control the tradeoff between fidelity and semantics at decode time. Wang et al. (SwitchCodec) used residual-expert sparse quantization for variable-bitrate coding. Aihara et al. (Sunac) built source separation and speech enhancement directly into the codec, and Chen et al. (AUV) proposed a single nested codebook as a unified audio representation across domains. Several TTS and generation pipelines now use codec tokens as their main intermediate representation, so a choice made in the codec architecture carries through to everything built on top of it.

The Low-resource Audio Codec Workshop was all about this. Its challenge asked teams to build neural audio codecs under a low bitrate budget of up to 6 kbps and an ultra-low budget of up to 1 kbps. I may be biased, since I supervised a paper accepted to the workshop ("Attention-guided Audio Compression for Multimodal LLMs" by Rane et al.), but the poster session had some fantastic ideas. Ziqian Wu from ByteDance won Track 1 of the LRAC Challenge with a system that combined SnakeBeta activations, Conv2FormerBlocks and eight loss terms, and reached a MUSHRA score of 82/100 on clean speech at the low bitrate. If you're into neural audio codecs, there is a lot more to explore on the [LRAC Challenge](https://lnkd.in/esHdD6Sx) page.

## 4. Music analysis now runs on learned representations

The tutorial on Current Approaches to Computational Analysis of Music Audio Signals, given by Xavier Serra, Martín Rocamora, Dmitry Bogdanov, Andrea Poltronieri and Pablo Alonso Jiménez, focused on representation learning for music analysis. It followed the field from handcrafted features through deep learning to self-supervised approaches that produce large-scale, transferable audio representations. The second half covered multi-modal methods that combine audio, symbolic and textual data for music understanding and cross-modal retrieval. The [code](https://lnkd.in/eXbWVJ7v) is available.

## 5. Causal representation learning is worth watching

Ali Tajer's tutorial on Causal Representation Learning (CRL) introduced a framework for recovering interpretable, mechanistic representations from high-dimensional data. He went through disentanglement, causal inference and identifiability, and then methods such as interventional CRL, which uses controlled perturbations to recover causal structure. He also compared unsupervised CRL with self-supervised JEPA on a robotics task, and suggested that CRL could be applied to blind source separation to improve generalization and interpretability, which I found exciting. His [slides](https://lnkd.in/eSPPvUdx) are online.

## Beyond the talks

The welcome reception had seafood, castells and an incredible live band, and I honestly can't pick a favorite. ICASSP kicked off with a bang.

If you want to explore more of these trends, Hieu-Thi Luong's [paper theme map](https://lnkd.in/eD9tx_Ze) is a good place to start. I'm happy to go deeper on any of these topics, and I'd like to hear what your own takeaways from ICASSP 2026 were.
