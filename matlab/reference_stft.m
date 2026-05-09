function [mag, phase] = reference_stft(x, fs)
%REFERENCE_STFT MATLAB reference frontend for EdgeSpeech-RT validation.
%   [MAG, PHASE] = reference_stft(X, FS) computes a 512-point Hann-windowed
%   STFT with 320-sample hop, matching the default 16 kHz / 20 ms SDK setup.

if nargin < 2
    fs = 16000;
end
if fs ~= 16000
    warning('EdgeSpeech-RT defaults are tuned for 16 kHz audio.');
end

nfft = 512;
hop = 320;
win = hann(nfft, 'symmetric');
overlap = nfft - hop;
[s, ~, ~] = stft(x(:), fs, 'Window', win, 'OverlapLength', overlap, 'FFTLength', nfft);
mag = abs(s);
phase = angle(s);
end
