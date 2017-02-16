#!/usr/bin/env python3

import sys, math, logging

import SoapySDR
import numpy

__version__ = '1.0.0'
logger = logging.getLogger(__name__)


def closest(num_list, num):
    """Return number closest to supplied number from list of numbers"""
    return min(num_list, key=lambda x: abs(x - num))


def detect_devices():
    """Detect connected SoapySDR devices"""
    devices = SoapySDR.Device.enumerate()
    return [{'driver': d['driver'], 'label': d['label']} for d in devices]


class SoapyDevice:
    """Simple wrapper for SoapySDR"""
    default_buffer_size = 8192

    def __init__(self, soapy_args='', sample_rate=2.00e6, bandwidth=0, corr=0, gain=20.7, auto_gain=False):
        self.device = SoapySDR.Device(soapy_args)
        self.buffer = None
        self.buffer_size = None
        self.stream_args = None

        self._hardware = self.device.getHardwareKey()
        self._stream = None
        self._stream_timeout = None
        self._freq = None
        self._sample_rate = None
        self._bandwidth = None
        self._corr = None
        self._gain = None
        self._auto_gain = None

        self.sample_rate = sample_rate
        self.bandwidth = bandwidth or sample_rate
        if corr:
            self.corr = corr
        self.gain = gain
        self.auto_gain = auto_gain

        self._fix_hardware_quirks()

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

    @property
    def hardware(self):
        """Type of SDR hardware (read-only)"""
        return self._hardware

    @property
    def is_streaming(self):
        """Has been start_stream() already called? (read-only)"""
        return bool(self._stream)

    @property
    def freq(self):
        """Center frequency [Hz]"""
        return self._freq

    @freq.setter
    def freq(self, freq):
        """Set center frequency [Hz]"""
        freq_range = self.device.getFrequencyRange(SoapySDR.SOAPY_SDR_RX, 0, 'RF')[0]
        if freq < freq_range.minimum() or freq > freq_range.maximum():
            raise ValueError('Center frequency out of range ({}, {})!'.format(
                freq_range.minimum(), freq_range.maximum()
            ))

        self._freq = freq
        self.device.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, 'RF', freq)

    @property
    def sample_rate(self):
        """Sample rate [Hz]"""
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, sample_rate):
        """Set sample rate [Hz]"""
        real_sample_rate = closest(self.list_sample_rates(), sample_rate)
        if sample_rate != real_sample_rate:
            logger.warning('Sample rate {} Hz is not supported, setting it to {} Hz!'.format(
                sample_rate, real_sample_rate
            ))

        self._sample_rate = real_sample_rate
        self.device.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, real_sample_rate)

    @property
    def bandwidth(self):
        """Filter bandwidth [Hz]"""
        return self._bandwidth

    @bandwidth.setter
    def bandwidth(self, bandwidth):
        """Set filter bandwidth [Hz]"""
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
        self.device.setBandwidth(SoapySDR.SOAPY_SDR_RX, 0, real_bandwidth)

    @property
    def gain(self):
        """Gain [dB]"""
        return self._gain

    @gain.setter
    def gain(self, gain):
        """Set gain [dB]"""
        gain_range = self.device.getGainRange(SoapySDR.SOAPY_SDR_RX, 0)
        if gain < gain_range.minimum() or gain > gain_range.maximum():
            raise ValueError('Gain out of range ({}, {})!'.format(
                gain_range.minimum(), gain_range.maximum()
            ))

        self._gain = gain
        self.device.setGain(SoapySDR.SOAPY_SDR_RX, 0, gain)

    @property
    def auto_gain(self):
        """Automatic Gain Control"""
        return self._auto_gain

    @auto_gain.setter
    def auto_gain(self, auto_gain):
        """Set Automatic Gain Control"""
        if not self.device.hasGainMode(SoapySDR.SOAPY_SDR_RX, 0):
            logger.warning('Device does not support Automatic Gain Control!')
            return

        self._auto_gain = auto_gain
        self.device.setGainMode(SoapySDR.SOAPY_SDR_RX, 0, auto_gain)

    @property
    def corr(self):
        """Frequency correction [ppm]"""
        return self._corr

    @corr.setter
    def corr(self, corr):
        """Set frequency correction [ppm]"""
        if 'CORR' not in self.device.listFrequencies(SoapySDR.SOAPY_SDR_RX, 0):
            logger.warning('Device does not support frequency correction!')
            return

        corr_range = self.device.getFrequencyRange(SoapySDR.SOAPY_SDR_RX, 0, 'CORR')[0]
        if corr < corr_range.minimum() or corr > corr_range.maximum():
            raise ValueError('Frequency correction out of range ({}, {})!'.format(
                corr_range.minimum(), corr_range.maximum()
            ))

        self._corr = corr
        self.device.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, 'CORR', corr)

    def list_sample_rates(self):
        """List allowed sample rates"""
        return self.device.listSampleRates(SoapySDR.SOAPY_SDR_RX, 0)

    def list_bandwidths(self):
        """List allowed bandwidths"""
        return self.device.listBandwidths(SoapySDR.SOAPY_SDR_RX, 0)

    def start_stream(self, buffer_size=0, stream_args=None):
        """Start streaming samples"""
        if self.is_streaming:
            raise RuntimeError('Streaming has been already initialized!')

        if stream_args or self.stream_args:
            logger.debug('SoapySDR stream - args: {}'.format(stream_args or self.stream_args))
            self._stream = self.device.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32, [0],
                                                   stream_args or self.stream_args)
        else:
            self._stream = self.device.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
        self.device.activateStream(self._stream)

        buffer_size = buffer_size or self.buffer_size
        if not buffer_size:
            try:
                buffer_size = self.device.getStreamMTU(self._stream)
            except AttributeError:
                logger.warning('getStreamMTU() not implemented! Using default value: {}'.format(
                    self.default_buffer_size
                ))
                buffer_size = self.default_buffer_size

        self.buffer = numpy.empty(buffer_size, numpy.complex64)
        self._stream_timeout = 0.1 + (buffer_size / self.sample_rate)
        logger.debug('SoapySDR stream - buffer size: {}'.format(buffer_size))
        logger.debug('SoapySDR stream - read timeout: {:.6f}'.format(self._stream_timeout))

        return self.buffer

    def stop_stream(self):
        """Stop streaming samples"""
        if not self.is_streaming:
            raise RuntimeError('Streaming is not initialized, you must run start_stream() first!')

        self.device.deactivateStream(self._stream)
        self.device.closeStream(self._stream)
        self._stream = None
        self.buffer = None

    def read_stream(self, stream_timeout=0):
        """Read samples into buffer"""
        if not self.is_streaming:
            raise RuntimeError('Streaming is not initialized, you must run start_stream() first!')

        buffer_size = len(self.buffer)
        if stream_timeout or self._stream_timeout:
            res = self.device.readStream(self._stream, [self.buffer], buffer_size,
                                         timeoutUs=math.ceil((stream_timeout or self._stream_timeout) * 1e6))
        else:
            res = self.device.readStream(self._stream, [self.buffer], buffer_size)
        if res.ret > 0 and res.ret < buffer_size:
            logger.warning('readStream() returned only {} samples, but buffer size is {}!'.format(
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
                logger.warning('Buffer overflow error in readStream()!')
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

    devices = detect_devices()
    if not devices:
        logger.error('No SoapySDR devices detected!')
        sys.exit(1)

    logger.info('Detected SoapySDR devices:')
    for i, d in enumerate(devices):
        logger.info('  {} ... driver={}, label={}'.format(i + 1, d['driver'], d['label']))
