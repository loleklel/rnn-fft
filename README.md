# rnn-fft (work in progress)

An experiment with RNNs applied to audio in the frequency domain, to try to generate new audio.

## Running

Note that if you do not have GPUs, you will need to edit the script to install `tensorflow` 
instead of `tensorflow-gpu`, and change `gpu_count` to 0.

```bash
python3 rnn-fft.py
```

This will start model training and then produce output showing the audio in the frequency domain,
and then a plot of some of the generated audio in time domain, and also the frequency domain.

It will output generated audio as `out.wav`.

## Status?

Doesn't really work, but produces some agreeable noise.