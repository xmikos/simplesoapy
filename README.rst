SimpleSoapy
===========

Simple pythonic wrapper for SoapySDR library

Requirements
------------

- `Python 3 <https://www.python.org>`_
- `NumPy <http://www.numpy.org>`_
- `SoapySDR <https://github.com/pothosware/SoapySDR>`_

Limitations
-----------

Only receiving is implemented. Transmission may be implemented in future.

Example
-------
::

    import simplesoapy
    import numpy
    
    # List all connected SoapySDR devices
    print(simplesoapy.detect_devices(as_string=True))
    
    # Initialize SDR device
    sdr = simplesoapy.SoapyDevice('driver=rtlsdr')
    
    # Set sample rate
    sdr.sample_rate = 2.56e6
    
    # Set center frequency
    sdr.freq = 88e6
    
    # Setup base buffer and start receiving samples. Base buffer size is determined
    # by SoapySDR.Device.getStreamMTU(). If getStreamMTU() is not implemented by driver,
    # SoapyDevice.default_buffer_size is used instead
    sdr.start_stream()
    
    # Create numpy array for received samples
    samples = numpy.empty(len(sdr.buffer) * 100, numpy.complex64)
    
    # Receive all samples
    sdr.read_stream_into_buffer(samples)
    
    # Stop receiving
    sdr.stop_stream()
