import numpy as np
from scipy.interpolate import interp1d


class LSOFDMChannelEstimator:
    """Small LS-OFDM estimator reused for DeepMIMO generated channels.

    Expected channel shape per sample is `(n_tap, n_rx, n_tx)`.
    """

    def __init__(self, n_tap, n_rx, n_tx, n_subcarriers=1024, pilot_spacing=4, snr_db=0.0):
        self.n_tap = n_tap
        self.n_rx = n_rx
        self.n_tx = n_tx
        self.n_subcarriers = n_subcarriers
        self.pilot_spacing = pilot_spacing
        self.snr_db = snr_db
        self.pilot_positions = np.arange(0, n_subcarriers, pilot_spacing)

    def time_to_frequency(self, h_time):
        h_padded = np.zeros((self.n_subcarriers, self.n_rx, self.n_tx), dtype=np.complex128)
        h_padded[: self.n_tap] = h_time
        return np.fft.fft(h_padded, axis=0)

    def frequency_to_time(self, h_freq):
        return np.fft.ifft(h_freq, axis=0)[: self.n_tap]

    def estimate_one(self, true_channel):
        h_true_freq = self.time_to_frequency(true_channel)
        signal_power = np.mean(np.abs(h_true_freq) ** 2)
        noise_power = signal_power / (10 ** (self.snr_db / 10))
        noise_std = np.sqrt(noise_power / 2)

        h_est_freq = np.zeros_like(h_true_freq)
        all_positions = np.arange(self.n_subcarriers)

        for rx in range(self.n_rx):
            for tx in range(self.n_tx):
                pilots = h_true_freq[self.pilot_positions, rx, tx]
                noise = noise_std * (
                    np.random.randn(len(self.pilot_positions))
                    + 1j * np.random.randn(len(self.pilot_positions))
                )
                pilots_noisy = pilots + noise

                mag_interp = interp1d(
                    self.pilot_positions,
                    np.abs(pilots_noisy),
                    kind="linear",
                    fill_value="extrapolate",
                )
                phase_interp = interp1d(
                    self.pilot_positions,
                    np.unwrap(np.angle(pilots_noisy)),
                    kind="linear",
                    fill_value="extrapolate",
                )
                h_est_freq[:, rx, tx] = mag_interp(all_positions) * np.exp(
                    1j * phase_interp(all_positions)
                )

        return self.frequency_to_time(h_est_freq)

    def estimate_batch(self, channels):
        estimates = np.empty_like(channels)
        for idx in range(len(channels)):
            estimates[idx] = self.estimate_one(channels[idx])
        return estimates
