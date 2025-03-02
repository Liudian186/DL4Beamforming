# File:       PlaneWaveData.py
# Author:     Dongwoon Hyun (dongwoon.hyun@stanford.edu)
# Created on: 2020-04-03
import numpy as np
import h5py
import os
from scipy.signal import hilbert, convolve
import glob


class PlaneWaveData:
    """A template class that contains the plane wave data.

    PlaneWaveData is a container or dataclass that holds all of the information
    describing a plane wave acquisition. Users should create a subclass that reimplements
    __init__() according to how their data is stored.

    The required information is:
    idata       In-phase (real) data with shape (nangles, nchans, nsamps)
    qdata       Quadrature (imag) data with shape (nangles, nchans, nsamps)
    angles      List of angles [radians]
    ele_pos     Element positions with shape (N,3) [m]
    fc          Center frequency [Hz]
    fs          Sampling frequency [Hz]
    fdemod      Demodulation frequency, if data is demodulated [Hz]
    c           Speed of sound [m/s]
    time_zero   List of time zeroes for each acquisition [s]

    Correct implementation can be checked by using the validate() method. See the
    PICMUSData class for a fully implemented example.
    """

    def __init__(self):
        """Users must re-implement this function to load their own data."""
        # Do not actually use PlaneWaveData.__init__() as is.
        raise NotImplementedError

        # We provide the following as a visual example for a __init__() method.
        nangles, nchans, nsamps = 2, 3, 4
        # Initialize the parameters that *must* be populated by child classes.
        self.idata = np.zeros((nangles, nchans, nsamps), dtype="float32")
        self.qdata = np.zeros((nangles, nchans, nsamps), dtype="float32")
        self.angles = np.zeros((nangles,), dtype="float32")
        self.ele_pos = np.zeros((nchans, 3), dtype="float32")
        self.fc = 5e6
        self.fs = 20e6
        self.fdemod = 0
        self.c = 1540
        self.time_zero = np.zeros((nangles,), dtype="float32")

    def validate(self):
        """Check to make sure that all information is loaded and valid."""
        # Check size of idata, qdata, angles, ele_pos
        assert self.idata.shape == self.qdata.shape
        assert self.idata.ndim == self.qdata.ndim == 3
        nangles, nchans, nsamps = self.idata.shape
        assert self.angles.ndim == 1 and self.angles.size == nangles
        assert self.ele_pos.ndim == 2 and self.ele_pos.shape == (nchans, 3)
        # Check frequencies (expecting more than 0.1 MHz)
        assert self.fc > 1e5
        assert self.fs > 1e5
        assert self.fdemod > 1e5 or self.fdemod == 0
        # Check speed of sound (should be between 1000-2000 for medical imaging)
        assert 1000 <= self.c <= 2000
        # Check that a separate time zero is provided for each transmit
        assert self.time_zero.ndim == 1 and self.time_zero.size == nangles


class PICMUSData(PlaneWaveData):
    """PICMUSData - Demonstration of how to use PlaneWaveData to load PICMUS data

    PICMUSData is a subclass of PlaneWaveData that loads the data from the PICMUS
    challenge from 2016 (https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016).
    PICMUSData re-implements the __init__() function of PlaneWaveData.
    """

    def __init__(self, database_path, acq, target, dtype):
        """Load PICMUS dataset as a PlaneWaveData object."""
        # Make sure the selected dataset is valid
        assert any([acq == a for a in ["simulation", "experiments"]])
        assert any([target == t for t in ["contrast_speckle", "resolution_distorsion"]])
        assert any([dtype == d for d in ["rf", "iq"]])

        # Load PICMUS dataset
        fname = os.path.join(
            database_path,
            acq,
            target,
            "%s_%s_dataset_%s.hdf5" % (target, acq[:4], dtype),
        )
        f = h5py.File(fname, "r")["US"]["US_DATASET0000"]
        self.idata = np.array(f["data"]["real"], dtype="float32")
        self.qdata = np.array(f["data"]["imag"], dtype="float32")
        self.angles = np.array(f["angles"])
        self.fc = 5208000.0  # np.array(f["modulation_frequency"]).item()
        self.fs = np.array(f["sampling_frequency"]).item()
        self.c = np.array(f["sound_speed"]).item()
        self.time_zero = np.array(f["initial_time"])
        self.ele_pos = np.array(f["probe_geometry"]).T
        self.fdemod = self.fc if dtype == "iq" else 0

        # If the data is RF, use the Hilbert transform to get the imag. component.
        if dtype == "rf":
            iqdata = hilbert(self.idata, axis=-1)
            self.qdata = np.imag(iqdata)

        # Make sure that time_zero is an array of size [nangles]
        if self.time_zero.size == 1:
            self.time_zero = np.ones_like(self.angles) * self.time_zero

        # Validate that all information is properly included
        super().validate()


class MYOData(PlaneWaveData):
    """Load data from Mayo Clinic."""

    def __init__(self, database_path, acq):
        # Make sure the selected dataset is valid
        fname = os.path.join(
            "%s/%s.hdf5" % (database_path, "MYO{:03d}".format(acq)),
        )
        assert fname, "File not found."

        # Load dataset
        f = h5py.File(fname, "r")

        # Phantom-specific parameters
        if acq == 1:
            sound_speed = 1580
        elif acq == 2:
            sound_speed = 1583
        elif acq == 3:
            sound_speed = 1578
        elif acq == 4:
            sound_speed = 1572
        elif acq == 5:
            sound_speed = 1562
        else:
            sound_speed = 1581

        # Get data
        self.idata = np.array(f["channel_data"], dtype="float32")
        self.qdata = np.imag(hilbert(self.idata, axis=-1))
        self.angles = np.array(f["angles"])
        self.fc = np.array(f["modulation_frequency"]).item()
        self.fs = np.array(f["sampling_frequency"]).item()
        self.c = sound_speed  # np.array(f["sound_speed"]).item()
        self.time_zero = np.zeros((len(self.angles),), dtype="float32")
        self.fdemod = 0

        # Make the element positions based on L11-4v geometry
        pitch = 0.3e-3
        nelems = self.idata.shape[1]
        xpos = np.arange(nelems) * pitch
        xpos -= np.mean(xpos)
        self.ele_pos = np.stack([xpos, 0 * xpos, 0 * xpos], axis=1)

        # For this dataset, time zero is the center point
        for i, a in enumerate(self.angles):
            self.time_zero[i] = self.ele_pos[-1, 0] * np.abs(np.sin(a)) / self.c

        # Validate that all information is properly included
        super().validate()


class INSData(PlaneWaveData):
    """Load data from INS."""

    def __init__(self, database_path, acq):
        # Make sure the selected dataset is valid
        fname = os.path.join(
            "%s/%s.hdf5" % (database_path, "INS{:03d}".format(acq)),
        )
        assert fname, "File not found."

        # Load dataset
        f = h5py.File(fname, "r")

        # Phantom-specific parameters
        # Phantom-specific parameters
        if acq == 1:
            sound_speed = 1521
        elif acq == 2:
            sound_speed = 1517
        elif acq == 3:
            sound_speed = 1506
        elif acq == 4:
            sound_speed = 1501
        elif acq == 5:
            sound_speed = 1506
        elif acq == 6:
            sound_speed = 1509
        elif acq == 7:
            sound_speed = 1490
        elif acq == 8:
            sound_speed = 1504
        elif acq == 9:
            sound_speed = 1473
        elif acq == 10:
            sound_speed = 1502
        elif acq == 11:
            sound_speed = 1511
        elif acq == 12:
            sound_speed = 1535
        elif acq == 13:
            sound_speed = 1453
        elif acq == 14:
            sound_speed = 1542
        elif acq == 15:
            sound_speed = 1539
        elif acq == 16:
            sound_speed = 1466
        elif acq == 17:
            sound_speed = 1462
        elif acq == 18:
            sound_speed = 1479
        elif acq == 19:
            sound_speed = 1469
        elif acq == 20:
            sound_speed = 1464
        elif acq == 21:
            sound_speed = 1508
        elif acq == 22:
            sound_speed = 1558
        elif acq == 23:
            sound_speed = 1463
        elif acq == 24:
            sound_speed = 1547
        elif acq == 25:
            sound_speed = 1477
        elif acq == 26:
            sound_speed = 1497
        else:
            sound_speed = 1540

        # Get data
        self.idata = np.array(f["channel_data"], dtype="float32")
        self.qdata = np.imag(hilbert(self.idata, axis=-1))
        self.angles = np.linspace(-16, 16, self.idata.shape[0]) * np.pi / 180
        self.fc = np.array(f["modulation_frequency"]).item()
        self.fs = np.array(f["sampling_frequency"]).item()
        self.c = sound_speed  # np.array(f["sound_speed"]).item()
        self.time_zero = -1 * np.array(f["start_time"], dtype="float32")[0]
        self.fdemod = 0
        self.ele_pos = np.array(f["element_positions"], dtype="float32").T
        self.ele_pos[:, 0] -= np.mean(self.ele_pos[:, 0])

        # For this dataset, time zero is the center point
        for i, a in enumerate(self.angles):
            self.time_zero[i] += self.ele_pos[-1, 0] * np.abs(np.sin(a)) / self.c

        # Validate that all information is properly included
        super().validate()


class TSHData(PlaneWaveData):
    """Load data from Tshinghua Univ."""

    def __init__(self, database_path, acq):
        # Make sure the selected dataset is valid
        fname = os.path.join(
            "%s/%s.hdf5" % (database_path, "TSH{:03d}".format(acq)),
        )
        assert fname, "File not found."

        # Load dataset
        f = h5py.File(fname, "r")

        # Get data
        self.angles = np.array(f["angles"])
        self.idata = np.array(f["channel_data"], dtype="float32")
        self.idata = np.reshape(self.idata, (128, len(self.angles), -1))
        self.idata = np.transpose(self.idata, (1, 0, 2))
        self.qdata = np.imag(hilbert(self.idata, axis=-1))
        self.fc = np.array(f["modulation_frequency"]).item()
        self.fs = np.array(f["sampling_frequency"]).item()
        self.c = 1540  # np.array(f["sound_speed"]).item()
        self.time_zero = np.zeros((len(self.angles),), dtype="float32")
        self.fdemod = 0

        # Make the element positions based on L11-4v geometry
        pitch = 0.3e-3
        nelems = self.idata.shape[1]
        xpos = np.arange(nelems) * pitch
        xpos -= np.mean(xpos)
        self.ele_pos = np.stack([xpos, 0 * xpos, 0 * xpos], axis=1)

        # For this dataset, time zero is the center point
        for i, a in enumerate(self.angles):
            self.time_zero[i] = self.ele_pos[-1, 0] * np.abs(np.sin(a)) / self.c

        # Validate that all information is properly included
        super().validate()


class UFLData(PlaneWaveData):
    """Load data from UFL."""

    def __init__(self, database_path, acq):
        # Make sure the selected dataset is valid
        fname = os.path.join(
            "%s/%s.hdf5" % (database_path, "UFL{:03d}".format(acq)),
        )
        assert fname, "File not found."

        # Load dataset
        f = h5py.File(fname, "r")

        # Phantom-specific parameters
        if acq == 1:
            sound_speed = 1526
        elif acq == 2 or acq == 4 or acq == 5:
            sound_speed = 1523
        else:
            sound_speed = 1525

        # Get data
        self.idata = np.array(f["channel_data"], dtype="float32")
        self.qdata = np.imag(hilbert(self.idata, axis=-1))
        self.angles = np.array(f["angles"]) * np.pi / 180
        self.fc = np.array(f["modulation_frequency"]).item()
        self.fs = np.array(f["channel_data_sampling_frequency"]).item()
        self.c = sound_speed  # np.array(f["sound_speed"]).item()
        self.time_zero = -1 * np.array(f["channel_data_t0"], dtype="float32")
        self.fdemod = self.fc

        # Make the element positions based on LA533 geometry
        pitch = 0.245e-3
        nelems = self.idata.shape[1]
        xpos = np.arange(nelems) * pitch
        xpos -= np.mean(xpos)
        self.ele_pos = np.stack([xpos, 0 * xpos, 0 * xpos], axis=1)

        # Make sure that time_zero is an array of size [nangles]
        if self.time_zero.size == 1:
            self.time_zero = np.ones_like(self.angles) * self.time_zero

        # Demodulate data and low-pass filter
        data = self.idata + 1j * self.qdata
        phase = np.reshape(np.arange(self.idata.shape[2], dtype="float"), (1, 1, -1))
        phase *= self.fdemod / self.fs
        data *= np.exp(-2j * np.pi * phase)
        dsfactor = int(np.floor(self.fs / self.fc))
        kernel = np.ones((1, 1, dsfactor), dtype="float") / dsfactor
        data = convolve(data, kernel, "same")
        data = data[:, :, ::dsfactor]
        self.fs /= dsfactor

        self.idata = np.real(data)
        self.qdata = np.imag(data)

        # Validate that all information is properly included
        super().validate()


class EUTData(PlaneWaveData):
    """Load data from EUT."""

    def __init__(self, database_path, acq):
        # Make sure the selected dataset is valid
        fname = os.path.join(
            "%s/%s.hdf5" % (database_path, "EUT{:03d}".format(acq)),
        )
        assert fname, "File not found."

        # Load dataset
        f = h5py.File(fname, "r")

        # Phantom-specific parameters
        if acq == 1:
            sound_speed = 1603
        elif acq == 2:
            sound_speed = 1618
        elif acq == 3:
            sound_speed = 1607
        elif acq == 4:
            sound_speed = 1614
        elif acq == 5:
            sound_speed = 1495
        else:
            sound_speed = 1479

        # Get data
        self.idata = np.array(f["channel_data"], dtype="float32")
        self.qdata = np.imag(hilbert(self.idata, axis=-1))
        self.angles = np.array(f["transmit_direction"])[:, 0]
        self.fc = np.array(f["modulation_frequency"]).item()
        self.fs = np.array(f["sampling_frequency"]).item()
        self.c = sound_speed
        self.time_zero = np.array(f["start_time"], dtype="float32")[0]
        self.fdemod = 0
        self.ele_pos = np.array(f["element_positions"], dtype="float32").T
        self.ele_pos[:, 0] -= np.mean(self.ele_pos[:, 0])

        # For this dataset, time zero is the center point
        for i, a in enumerate(self.angles):
            self.time_zero[i] = self.ele_pos[-1, 0] * np.abs(np.sin(a)) / self.c

        # Seems to be some offset
        self.time_zero += 10 / self.fc

        # Validate that all information is properly included
        super().validate()


class OSLData(PlaneWaveData):
    """Load data from EUT."""

    def __init__(self, database_path, acq):
        # Make sure the selected dataset is valid
        fname = os.path.join(
            "%s/%s.hdf5" % (database_path, "UFL{:03d}".format(acq)),
        )
        assert fname, "File not found."

        # Load dataset
        f = h5py.File(fname, "r")

        # Phantom-specific parameters
        if acq == 2:
            sound_speed = 1536
        elif acq == 3:
            sound_speed = 1543
        elif acq == 4:
            sound_speed = 1538
        elif acq == 5:
            sound_speed = 1539
        elif acq == 6:
            sound_speed = 1541
        elif acq == 7:
            sound_speed = 1540
        else:
            sound_speed = 1540

        # Get data
        self.idata = np.array(f["channel_data"], dtype="float32")
        self.qdata = np.imag(hilbert(self.idata, axis=-1))
        self.angles = np.array(f["transmit_direction"][0], dtype="float32")
        self.fc = np.array(f["modulation_frequency"]).item()
        self.fs = np.array(f["sampling_frequency"]).item()
        self.c = sound_speed  # np.array(f["sound_speed"]).item()
        self.time_zero = -1 * np.array(f["start_time"], dtype="float32")[0]
        self.fdemod = 0
        self.ele_pos = np.array(f["element_positions"], dtype="float32").T
        self.ele_pos[:, 0] -= np.mean(self.ele_pos[:, 0])

        # Validate that all information is properly included
        super().validate()


class JHUData(PlaneWaveData):
    """Load data from JHU."""

    def __init__(self, database_path, acq):
        # Make sure the selected dataset is valid
        fname = os.path.join(
            "%s/%s.hdf5" % (database_path, "UFL{:03d}".format(acq)),
        )
        assert fname, "File not found."

        # Load dataset
        f = h5py.File(fname, "r")

        # Get data
        self.idata = np.array(f["channel_data"], dtype="float32")
        self.qdata = np.imag(hilbert(self.idata, axis=-1))
        self.angles = np.array(f["angles"])
        self.fc = np.array(f["modulation_frequency"]).item()
        self.fs = np.array(f["sampling_frequency"]).item()
        self.c = np.array(f["sound_speed"]).item()
        self.time_zero = -1 * np.array(f["time_zero"], dtype="float32")
        self.fdemod = 0

        xpos = np.array(f["element_positions"], dtype="float32").T
        self.ele_pos = np.stack([xpos, 0 * xpos, 0 * xpos], axis=1)
        self.zlims = np.array([0e-3, self.idata.shape[2] * self.c / self.fs / 2])
        self.xlims = np.array([self.ele_pos[0, 0], self.ele_pos[-1, 0]])

        # For this dataset, time zero is the center point
        # self.time_zero = np.zeros((len(self.angles),), dtype="float32")
        # for i, a in enumerate(self.angles):
        #     self.time_zero[i] = self.ele_pos[-1, 0] * np.abs(np.sin(a)) / self.c

        # Seems to be some offset
        # self.time_zero -= 10 / self.fc

        # Validate that all information is properly included
        super().validate()
