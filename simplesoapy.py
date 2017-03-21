#!/usr/bin/env python3

import sys, math, logging

import SoapySDR
import numpy

__version__ = '1.3.0'
logger = logging.getLogger(__name__)


def closest(num_list, num):
    """Return number closest to supplied number from list of numbers"""
    return min(num_list, key=lambda x: abs(x - num))


def detect_devices(soapy_args='', as_string=False):
    """Detect connected SoapySDR devices"""
    devices = [dict(d) for d in SoapySDR.Device.enumerate(soapy_args)]

    if not as_string:
        return devices
    else:
        devices_str = []
        for d in devices:
            d_str = []
            d_str.append('driver={}'.format(d['driver']))
            if d['driver'] == 'remote':
                d_str.append('remote:driver={}'.format(d['remote:driver']))
                d_str.append('remote={}'.format(d['remote']))
            if 'serial' in d:
                d_str.append('serial={}'.format(d['serial']))
            if 'device_id' in d:
                d_str.append('device_id={}'.format(d['device_id']))
            if 'rtl' in d:
                d_str.append('rtl={}'.format(d['rtl']))
            d_str.append('label={}'.format(d['label']))
            devices_str.append(', '.join(d_str))
        return devices_str


class SoapyDevice:
    """Simple wrapper for SoapySDR"""
    default_buffer_size = 8192

    def __init__(self, soapy_args='', sample_rate=0, bandwidth=0, corr=0, gain=0, auto_gain=False,
                 channel=0, antenna='', settings=None, force_sample_rate=False, force_bandwidth=False):
        self.device = SoapySDR.Device(soapy_args)
        self.buffer = None
        self.buffer_size = None
        self.buffer_overflow_count = 0
        self.stream = None
        self.stream_args = None
        self.stream_timeout = None

        self._hardware = self.device.getHardwareKey()
        self._channel = None
        self._freq = None
        self._sample_rate = None
        self._bandwidth = None
        self._corr = None
        self._gain = None
        self._auto_gain = None
        self._antenna = None

        self.channel = channel
        self.force_sample_rate = force_sample_rate
        self.force_bandwidth = force_bandwidth

        self._fix_hardware_quirks()

        if sample_rate:
            self.sample_rate = sample_rate

        if bandwidth:
            self.bandwidth = bandwidth

        if corr:
            self.corr = corr

        if gain and isinstance(gain, dict):
            for amp_name, value in gain.items():
                self.set_gain(amp_name, value)
        elif gain:
            self.gain = gain

        if auto_gain:
            self.auto_gain = auto_gain

        if antenna:
            self.antenna = antenna

        if settings:
            for setting_name, value in settings.items():
                self.set_setting(setting_name, value)

    def _fix_hardware_quirks(self):
        """Apply some settings to fix quirks of specific hardware"""
        if self.hardware == 'RTLSDR':
            logger.debug('Applying fixes for RTLSDR quirks...')
            # Fix buffer overflows when reading too much samples too quickly
            # (maybe using bigger buffer, set up dynamicaly based on number of samples, would be better?)
            if not self.stream_args:
                self.stream_args = {'buffers': '100'}  # default is 15 buffers
        elif self.hardware == 'SDRPlay':
            logger.debug('Applying fixes for SDRPlay quirks...')
            # Don't use base buffer size returned by getStreamMTU(), it's too big
            self.buffer_size = self.default_buffer_size
        elif self.hardware == 'LimeSDR-USB':
            logger.debug('Applying fixes for LimeSDR-USB quirks...')
            # LimeSDR driver doesn't provide list of allowed sample rates and bandwidths
            self.force_sample_rate = True
            self.force_bandwidth = True

    @property
    def hardware(self):
        """Type of SDR hardware (read-only)"""
        return self._hardware

    @property
    def is_streaming(self):
        """Has been start_stream() already called? (read-only)"""
        return bool(self.stream)

    @property
    def channel(self):
        """RX channel number"""
        return self._channel

    @channel.setter
    def channel(self, channel):
        """Set RX channel number"""
        if channel in self.list_channels():
            self._channel = channel
        else:
            logger.warning('Incorrect RX channel number, using channel 0 instead!')
            self._channel = 0

    @property
    def freq(self):
        """Center frequency [Hz]"""
        return self.device.getFrequency(SoapySDR.SOAPY_SDR_RX, self._channel, 'RF')

    @freq.setter
    def freq(self, freq):
        """Set center frequency [Hz]"""
        freq_range = self.device.getFrequencyRange(SoapySDR.SOAPY_SDR_RX, self._channel, 'RF')[0]
        if freq < freq_range.minimum() or freq > freq_range.maximum():
            raise ValueError('Center frequency out of range ({}, {})!'.format(
                freq_range.minimum(), freq_range.maximum()
            ))

        self._freq = freq
        self.device.setFrequency(SoapySDR.SOAPY_SDR_RX, self._channel, 'RF', freq)

    @property
    def sample_rate(self):
        """Sample rate [Hz]"""
        return self.device.getSampleRate(SoapySDR.SOAPY_SDR_RX, self._channel)

    @sample_rate.setter
    def sample_rate(self, sample_rate):
        """Set sample rate [Hz]"""
        if self.force_sample_rate:
            real_sample_rate = sample_rate
        else:
            real_sample_rate = closest(self.list_sample_rates(), sample_rate)
            if sample_rate != real_sample_rate:
                logger.warning('Sample rate {} Hz is not supported, setting it to {} Hz!'.format(
                    sample_rate, real_sample_rate
                ))

        self._sample_rate = real_sample_rate
        self.device.setSampleRate(SoapySDR.SOAPY_SDR_RX, self._channel, real_sample_rate)

    @property
    def bandwidth(self):
        """Filter bandwidth [Hz]"""
        return self.device.getBandwidth(SoapySDR.SOAPY_SDR_RX, self._channel)

    @bandwidth.setter
    def bandwidth(self, bandwidth):
        """Set filter bandwidth [Hz]"""
        if self.force_bandwidth:
            real_bandwidth = bandwidth
        else:
            bandwidths = self.list_bandwidths()
            if not bandwidths:
                logger.warning('Device does not support setting filter bandwidth!')
                return

            real_bandwidth = closest(bandwidths, bandwidth)
            if bandwidth != real_bandwidth:
                logger.warning('Filter bandwidth {} Hz is not supported, setting it to {} Hz!'.format(
                    bandwidth, real_bandwidth
                ))

        self._bandwidth = real_bandwidth
        self.device.setBandwidth(SoapySDR.SOAPY_SDR_RX, self._channel, real_bandwidth)

    @property
    def gain(self):
        """Gain [dB]"""
        return self.device.getGain(SoapySDR.SOAPY_SDR_RX, self._channel)

    @gain.setter
    def gain(self, gain):
        """Set gain [dB]"""
        gain_range = self.device.getGainRange(SoapySDR.SOAPY_SDR_RX, self._channel)
        if gain < gain_range.minimum() or gain > gain_range.maximum():
            raise ValueError('Gain out of range ({}, {})!'.format(
                gain_range.minimum(), gain_range.maximum()
            ))

        self._gain = gain
        self.device.setGain(SoapySDR.SOAPY_SDR_RX, self._channel, gain)

    @property
    def auto_gain(self):
        """Automatic Gain Control"""
        return self.device.getGainMode(SoapySDR.SOAPY_SDR_RX, self._channel)

    @auto_gain.setter
    def auto_gain(self, auto_gain):
        """Set Automatic Gain Control"""
        if not self.device.hasGainMode(SoapySDR.SOAPY_SDR_RX, self._channel):
            logger.warning('Device does not support Automatic Gain Control!')
            return

        self._auto_gain = auto_gain
        self.device.setGainMode(SoapySDR.SOAPY_SDR_RX, self._channel, auto_gain)

    @property
    def antenna(self):
        """Selected antenna"""
        return self.device.getAntenna(SoapySDR.SOAPY_SDR_RX, self._channel)

    @antenna.setter
    def antenna(self, antenna):
        """Set the selected antenna"""
        antennas = self.list_antennas()
        if not antennas:
            logger.warning('Device does not support setting selected antenna!')
            return

        if antenna not in antennas:
            logger.warning('Unknown antenna {}!'.format(antenna))
            return

        self._antenna = antenna
        self.device.setAntenna(SoapySDR.SOAPY_SDR_RX, self._channel, antenna)

    @property
    def corr(self):
        """Frequency correction [ppm]"""
        try:
            return self.device.getFrequency(SoapySDR.SOAPY_SDR_RX, self._channel, 'CORR')
        except RuntimeError:
            return 0

    @corr.setter
    def corr(self, corr):
        """Set frequency correction [ppm]"""
        if 'CORR' not in self.list_frequencies():
            logger.warning('Device does not support frequency correction!')
            return

        corr_range = self.device.getFrequencyRange(SoapySDR.SOAPY_SDR_RX, self._channel, 'CORR')[0]
        if corr < corr_range.minimum() or corr > corr_range.maximum():
            raise ValueError('Frequency correction out of range ({}, {})!'.format(
                corr_range.minimum(), corr_range.maximum()
            ))

        self._corr = corr
        self.device.setFrequency(SoapySDR.SOAPY_SDR_RX, self._channel, 'CORR', corr)

    def list_channels(self):
        """List available RX channels"""
        return list(range(self.device.getNumChannels(SoapySDR.SOAPY_SDR_RX)))

    def list_sample_rates(self):
        """List allowed sample rates"""
        return self.device.listSampleRates(SoapySDR.SOAPY_SDR_RX, self._channel)

    def list_bandwidths(self):
        """List allowed bandwidths"""
        return self.device.listBandwidths(SoapySDR.SOAPY_SDR_RX, self._channel)

    def list_antennas(self):
        """List available antennas"""
        return self.device.listAntennas(SoapySDR.SOAPY_SDR_RX, self._channel)

    def list_gains(self):
        """List available amplification elements"""
        return self.device.listGains(SoapySDR.SOAPY_SDR_RX, self._channel)

    def list_frequencies(self):
        """List available tunable elements"""
        return self.device.listFrequencies(SoapySDR.SOAPY_SDR_RX, self._channel)

    def list_settings(self):
        """List available device settings, their default values and description"""
        settings = {
            s.key: {'value': s.value, 'name': s.name, 'description': s.description}
            for s in self.device.getSettingInfo()
        }
        return settings

    def get_gain(self, amp_name):
        """Get gain of given amplification element"""
        if amp_name not in self.list_gains():
            raise ValueError('Unknown amplification element!')
        return self.device.getGain(SoapySDR.SOAPY_SDR_RX, self._channel, amp_name)

    def set_gain(self, amp_name, value):
        """Set gain of given amplification element"""
        if amp_name not in self.list_gains():
            raise ValueError('Unknown amplification element!')
        self.device.setGain(SoapySDR.SOAPY_SDR_RX, self._channel, amp_name, value)

    def get_frequency(self, tunable_name):
        """Get frequency of given tunable element"""
        if tunable_name not in self.list_frequencies():
            raise ValueError('Unknown tunable element!')
        return self.device.getFrequency(SoapySDR.SOAPY_SDR_RX, self._channel, tunable_name)

    def set_frequency(self, tunable_name, value):
        """Set frequency of given tunable element"""
        if tunable_name not in self.list_frequencies():
            raise ValueError('Unknown tunable element!')
        self.device.setFrequency(SoapySDR.SOAPY_SDR_RX, self._channel, tunable_name, value)

    def get_setting(self, setting_name):
        """Get value of given device setting"""
        if setting_name not in self.list_settings():
            raise ValueError('Unknown device setting!')
        return self.device.readSetting(setting_name)

    def set_setting(self, setting_name, value):
        """Set value of given device setting"""
        if setting_name not in self.list_settings():
            raise ValueError('Unknown device setting!')
        self.device.writeSetting(setting_name, value)

    def start_stream(self, buffer_size=0, stream_args=None):
        """Start streaming samples"""
        if self.is_streaming:
            raise RuntimeError('Streaming has been already initialized!')

        if stream_args or self.stream_args:
            logger.debug('SoapySDR stream - args: {}'.format(stream_args or self.stream_args))
            self.stream = self.device.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32, [self._channel],
                                                  stream_args or self.stream_args)
        else:
            self.stream = self.device.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32, [self._channel])
        self.device.activateStream(self.stream)

        buffer_size = buffer_size or self.buffer_size
        if not buffer_size:
            try:
                buffer_size = self.device.getStreamMTU(self.stream)
            except AttributeError:
                logger.warning('getStreamMTU not implemented! Using default value: {}'.format(
                    self.default_buffer_size
                ))
                buffer_size = self.default_buffer_size

        self.buffer = numpy.empty(buffer_size, numpy.complex64)
        self.buffer_overflow_count = 0
        self.stream_timeout = 0.1 + (buffer_size / self.sample_rate)
        logger.debug('SoapySDR stream - buffer size: {}'.format(buffer_size))
        logger.debug('SoapySDR stream - read timeout: {:.6f}'.format(self.stream_timeout))

        return self.buffer

    def stop_stream(self):
        """Stop streaming samples"""
        if not self.is_streaming:
            raise RuntimeError('Streaming is not initialized, you must run start_stream() first!')

        self.device.deactivateStream(self.stream)
        self.device.closeStream(self.stream)
        self.stream = None
        self.buffer = None

    def read_stream(self, stream_timeout=0):
        """Read samples into buffer"""
        if not self.is_streaming:
            raise RuntimeError('Streaming is not initialized, you must run start_stream() first!')

        buffer_size = len(self.buffer)
        if stream_timeout or self.stream_timeout:
            res = self.device.readStream(self.stream, [self.buffer], buffer_size,
                                         timeoutUs=math.ceil((stream_timeout or self.stream_timeout) * 1e6))
        else:
            res = self.device.readStream(self.stream, [self.buffer], buffer_size)
        if res.ret > 0 and res.ret < buffer_size:
            logger.warning('readStream returned only {} samples, but buffer size is {}!'.format(
                res.ret, buffer_size
            ))
        return res

    def read_stream_into_buffer(self, output_buffer):
        """Read samples into supplied output_buffer (blocks until output_buffer is full)"""
        output_buffer_size = len(output_buffer)
        ptr = 0
        while True:
            res = self.read_stream()
            if res.ret > 0:
                output_buffer[ptr:ptr + res.ret] = self.buffer[:min(res.ret, output_buffer_size - ptr)]
                ptr += res.ret
            elif res.ret == -4:
                self.buffer_overflow_count += 1
                logger.debug('Buffer overflow error in readStream ({:d})!'.format(self.buffer_overflow_count))
                logger.debug('Value of ptr when overflow happened: {}'.format(ptr))
            else:
                raise RuntimeError('Unhandled readStream() error: {} ({})'.format(
                    res.ret, SoapySDR.errToStr(res.ret)
                ))

            if ptr >= len(output_buffer):
                return


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)s: %(message)s'
    )

    devices = detect_devices(as_string=True)
    if not devices:
        logger.error('No SoapySDR devices detected!')
        sys.exit(1)

    logger.info('Detected SoapySDR devices:')
    for i, d in enumerate(devices):
        logger.info('  {}'.format(d))
