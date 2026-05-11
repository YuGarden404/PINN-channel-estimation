import numpy as np
from scipy.interpolate import interp1d


class LSOFDMChannelEstimator:
    """Pilot-limited LS-OFDM estimator for DeepMIMO tap-domain channels.

    Each input sample is expected to have shape ``(n_tap, n_rx, n_tx)``. The
    estimator transforms taps to the frequency domain, observes uniformly spaced
    noisy pilot tones, interpolates magnitude and unwrapped phase across all
    subcarriers, and maps the result back to the first ``n_tap`` delay taps.
    """

    def __init__(
        self,
        n_tap,
        n_rx,
        n_tx,
        n_subcarriers=1024,
        pilot_spacing=4,
        snr_db=0.0,
        seed=42,
    ):
        if n_tap <= 0 or n_rx <= 0 or n_tx <= 0:
            raise ValueError("n_tap, n_rx, and n_tx must be positive.")
        if n_subcarriers < n_tap:
            raise ValueError("n_subcarriers must be greater than or equal to n_tap.")
        if pilot_spacing <= 0:
            raise ValueError("pilot_spacing must be positive.")

        self.n_tap = int(n_tap)
        self.n_rx = int(n_rx)
        self.n_tx = int(n_tx)
        self.n_subcarriers = int(n_subcarriers)
        self.pilot_spacing = int(pilot_spacing)
        self.snr_db = float(snr_db)
        self.rng = np.random.default_rng(seed)
        self.pilot_positions = np.arange(0, self.n_subcarriers, self.pilot_spacing)
        if self.pilot_positions[-1] != self.n_subcarriers - 1:
            self.pilot_positions = np.unique(
                np.append(self.pilot_positions, self.n_subcarriers - 1)
            )

    @property
    def pilot_fraction(self):
        return float(len(self.pilot_positions) / self.n_subcarriers)

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
                    self.rng.standard_normal(len(self.pilot_positions))
                    + 1j * self.rng.standard_normal(len(self.pilot_positions))
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
        channels = np.asarray(channels)
        expected_shape = (self.n_tap, self.n_rx, self.n_tx)
        if channels.ndim != 4 or tuple(channels.shape[1:]) != expected_shape:
            raise ValueError(
                f"Expected channels with shape (n, {expected_shape}), got {channels.shape}."
            )

        estimates = np.empty_like(channels, dtype=np.complex64)
        for idx in range(len(channels)):
            estimates[idx] = self.estimate_one(channels[idx]).astype(np.complex64)
        return estimates
